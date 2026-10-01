# SPDX-License-Identifier: Apache-2.0
"""Exercise real pinned DMA and consumer ownership on an explicitly leased HPU."""

import os

import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory
from vllm_gaudi.ops.deepseek_v41_host import EngramHost, _C1Packet, _TransferSlot, host_native

pytestmark = pytest.mark.skipif(os.getenv("DSV41_TEST_HPU") != "1", reason="An HPU module lease is required")


@pytest.mark.parametrize("heads", [3, 6, 12])
@torch.inference_mode()
def test_staging_views_preserve_all_bytes_and_capacity(heads):
    import habana_frameworks.torch.core  # noqa: F401

    slot = _TransferSlot(6, heads, 256, "hpu")
    slot.host.fill_(197)
    pointers = slot.decode_host.data_ptr(), slot.decode_device.data_ptr()
    for count in (1, 6, 1, 2, 1):
        weights = np.arange(6 * heads * 256, dtype=np.uint8).reshape(6 * heads, 256)
        scales = (255 - np.arange(6 * heads * 8, dtype=np.uint8)).reshape(6 * heads, 8)
        np.copyto(slot.gather.weights, weights)
        np.copyto(slot.gather.scales, scales)
        before = slot.host[count:].clone()
        host, device = slot.fill(count)
        expected = np.concatenate(
            (weights[: count * heads].reshape(count, heads, 256), scales[: count * heads].reshape(count, heads, 8)),
            axis=-1,
        )
        assert np.array_equal(host.numpy(), expected)
        assert torch.equal(slot.host[count:], before)
        assert host.is_pinned("hpu")
        if count == 1:
            assert (host.data_ptr(), device.data_ptr()) == pointers


@pytest.mark.parametrize("tp_size,tp_rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("preparation", ["compat", "native", "packet", "direct"])
@torch.inference_mode()
def test_two_layer_dma_ring_reuse_and_request_reset(
    tmp_path, tp_size, tp_rank, preparation, capacity=6, deferred=False
):
    import habana_frameworks.torch.core  # noqa: F401

    native = host_native()
    HostRows, NativeC1Prepare = native.HostRows, native.NativeC1Prepare
    heads = 24 // tp_size

    host = EngramHost.__new__(EngramHost)
    host.tensor_parallel_size = tp_size
    host.layout = EngramHashLayout.from_config(
        {
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [10000, 10000],
            "engram_max_ngram_size": 4,
            "engram_n_heads": 8,
            "engram_compressed_vocab_size": 8,
            "engram_vocab_size": 5,
            "engram_pad_token_id": 2,
            "engram_head_dim": 256,
        }
    )
    host.history = EngramTokenHistory(host.layout, np.arange(16) % 8)
    host.tables, host.shards, host.slots = {}, {}, {}
    host.histories, host.table_page_counts = {}, {}
    host.stream = torch.hpu.Stream()
    host.generation, host.pending, host.closed = 0, None, False
    host.ready_ticket, host.profile_records, host.batches = None, None, None
    host.native_c1 = None
    host.c1_abi = 2 if preparation != "compat" else None
    host.device_c1 = host.device_rows = host.device_history = None
    host.device_history_parity = 0
    host.device_request = host.device_position = host.device_pending = None
    host.c1_packets = None
    host.max_tokens, host.ring_size = capacity, 3
    host.audit = dict(gathers=0, major_faults=0, dma_bytes=0, generations=0)
    tables = {}
    for layer, rows in ((1, 10000), (14, 10000)):
        weights = np.arange(rows * 256, dtype=np.uint8).reshape(rows, 256)
        weights ^= np.arange(rows, dtype=np.uint8)[:, None]
        scales = np.arange(rows * 8, dtype=np.uint8).reshape(rows, 8)
        path = tmp_path / f"layer-{layer}.bin"
        path.write_bytes(weights.tobytes() + scales.tobytes())
        shard = host.layout.head_shard(layer, tp_rank, tp_size)
        first, last = shard["row_start"], shard["row_stop"]
        host.tables[layer] = HostRows(
            str(path), first * 256, str(path), weights.size + first * 8, first, last, 256, True, False
        )
        host.shards[layer] = shard
        host.slots[layer] = [_TransferSlot(capacity, heads, 256, "hpu") for _ in range(3)]
        tables[layer] = weights, scales
    if preparation != "compat":
        layers = host.layout.layer_ids
        if preparation in ("packet", "direct"):
            first = _C1Packet([heads, heads], 256, "hpu")
            destination = first.device if preparation == "direct" else None
            host.c1_packets = [first] + [
                _C1Packet([heads, heads], 256, "hpu", destination=destination) for _ in range(2)
            ]
            targets = [packet.targets for packet in host.c1_packets]
            for packet in host.c1_packets:
                assert packet.host.is_pinned("hpu")
                assert not np.shares_memory(*packet.targets)
        else:
            targets = [[host.slots[layer][slot].decode_host.numpy() for layer in layers] for slot in range(3)]
        host.native_c1 = NativeC1Prepare(
            host.history.token_map,
            host.layout.multipliers,
            host.layout.primes,
            host.layout.offsets,
            np.array([host.shards[layer]["head_start"] for layer in layers], dtype=np.int64),
            np.array([host.shards[layer]["head_stop"] for layer in layers], dtype=np.int64),
            host.history.pad_id,
            [host.tables[layer] for layer in layers],
            targets,
        )
    pending_results = []
    consumer = torch.hpu.Stream()
    try:
        for request in ("first", "second"):
            host.reset(request)
            for step, count in enumerate((1, capacity, 1, 6 if deferred else 2, 1, capacity, 1, 1, 1)):
                tokens = [(step + i) % 16 for i in range(count)]
                with torch.hpu.stream(consumer):
                    ticket = host.prepare(
                        request,
                        tokens,
                        [i == 1 or step == 6 for i in range(count)],
                        defer_wait=deferred and count == 16384,
                    )
                    inputs = host.defer_prefill(ticket) if deferred and count == 16384 else ticket.buffers
                    assert ticket.packet == (preparation in ("packet", "direct") and count == 1)
                    if ticket.packet:
                        packet = host.c1_packets[ticket.slot]
                        with pytest.raises(RuntimeError, match="pending"):
                            packet.reuse()
                        with pytest.raises(RuntimeError, match="Stale"):
                            packet.complete(ticket.generation - 1, consumer)
                    assert torch.hpu.current_stream() == consumer
                    with pytest.raises(RuntimeError, match="pending"):
                        host.prepare(request, [1])
                    for index, layer in enumerate(host.layout.layer_ids):
                        value = inputs[index]
                        shard = host.shards[layer]
                        ids = ticket.batch.hash_ids[:, index, shard["head_start"] : shard["head_stop"]]
                        w, s = tables[layer]
                        expected = np.concatenate((w[ids], s[ids]), axis=-1)
                        # Retain device consumers without a per-step CPU read;
                        # wrapping the ring must preserve all pending outputs.
                        pending_results.append((value.clone(), torch.from_numpy(expected)))
                    host.complete(ticket, count if count == 1 else count - 1)
                    with pytest.raises(RuntimeError, match="Stale"):
                        host.complete(ticket, count)
        for actual, expected in pending_results:
            assert torch.equal(actual.cpu(), expected)
        assert host.audit["gathers"] == 36
        assert host.audit["generations"] == 18
        assert host.audit.get("native_c1", 0) == (12 if preparation != "compat" else 0)
        assert host.audit.get("c1_packets", 0) == (12 if preparation in ("packet", "direct") else 0)
    finally:
        host.close()


@pytest.mark.parametrize("tp_rank", [0, 3])
@torch.inference_mode()
def test_16k_tp4_engram_staging_transition_and_request_reuse(tmp_path, tp_rank):
    test_two_layer_dma_ring_reuse_and_request_reset(tmp_path, 4, tp_rank, "packet", capacity=16384)


@pytest.mark.parametrize("tp_rank", [0, 3])
@torch.inference_mode()
def test_deferred_16k_ring_reuse_then_c1_and_c6(tmp_path, tp_rank):
    test_two_layer_dma_ring_reuse_and_request_reset(tmp_path, 4, tp_rank, "packet", capacity=16384, deferred=True)


@torch.inference_mode()
def test_native_packet_completion_preserves_queued_consumers_and_ring_reuse():
    from vllm_gaudi.ops.deepseek_v41_completion import _control_bridge

    bridge = _control_bridge()
    packets = [_C1Packet([6, 6], 256, "hpu") for _ in range(3)]
    for packet in packets:
        packet.prepare_native_completion(bridge.record_native_completion)
    pending = []
    previous_contents = [None] * len(packets)
    stream = torch.hpu.current_stream()
    assert stream.stream_id == 0
    for generation in range(1, 19):
        slot = (generation - 1) % len(packets)
        packet = packets[slot]
        packet.reuse()
        packet.host.fill_((generation * 17) % 256)
        # Alternate full-packet and late-only uploads as production does at
        # request/prefill boundaries. Preserve the layer-1 bytes in late mode.
        late = generation > 3 and generation % 2 == 0
        expected = previous_contents[slot].clone() if late else packet.host.clone()
        if late:
            expected[6 * 264 :] = packet.host[6 * 264 :]
        previous_contents[slot] = expected
        packet.upload(generation, late_only=late)
        # Retain real asynchronous device consumers across ring wrap. Reading
        # only the producer or event would not establish their ownership.
        pending.append((packet.device.clone(), expected))
        packet.complete(generation, stream)
    for actual, expected in pending:
        assert torch.equal(actual.cpu(), expected)
    for packet in packets:
        packet.reuse()
