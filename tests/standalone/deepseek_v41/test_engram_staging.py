# SPDX-License-Identifier: Apache-2.0
"""Exercise real pinned DMA and consumer ownership on an explicitly leased HPU."""
import os

import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory
from vllm_gaudi.ops.deepseek_v41_host import EngramHost, _C1Packet, _TransferSlot

pytestmark = pytest.mark.skipif(os.getenv("DSV41_TEST_HPU") != "1", reason="An HPU module lease is required")


@pytest.mark.parametrize("heads", [3, 12])
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
            (weights[:count * heads].reshape(count, heads, 256), scales[:count * heads].reshape(count, heads, 8)),
            axis=-1)
        assert np.array_equal(host.numpy(), expected)
        assert torch.equal(slot.host[count:], before)
        assert host.is_pinned("hpu")
        if count == 1:
            assert (host.data_ptr(), device.data_ptr()) == pointers


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("preparation", ["compat", "native", "packet"])
@torch.inference_mode()
def test_two_layer_dma_ring_reuse_and_request_reset(tmp_path, tp_rank, preparation):
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.lib.dsv41_host_gather import HostRows, NativeC1Prepare

    host = EngramHost.__new__(EngramHost)
    host.layout = EngramHashLayout.from_config({
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [10000, 10000],
        "engram_max_ngram_size": 4,
        "engram_n_heads": 8,
        "engram_compressed_vocab_size": 8,
        "engram_vocab_size": 5,
        "engram_pad_token_id": 2,
        "engram_head_dim": 256
    })
    host.history = EngramTokenHistory(host.layout, np.arange(16) % 8)
    host.tables, host.shards, host.slots = {}, {}, {}
    host.stream = torch.hpu.Stream()
    host.generation, host.pending, host.closed = 0, None, False
    host.native_c1 = None
    host.c1_packets = None
    host.max_tokens, host.ring_size = 6, 3
    host.audit = dict(gathers=0, major_faults=0, dma_bytes=0, generations=0)
    tables = {}
    for layer, rows in ((1, 10000), (14, 10000)):
        weights = np.arange(rows * 256, dtype=np.uint8).reshape(rows, 256)
        weights ^= np.arange(rows, dtype=np.uint8)[:, None]
        scales = np.arange(rows * 8, dtype=np.uint8).reshape(rows, 8)
        path = tmp_path / f"layer-{layer}.bin"
        path.write_bytes(weights.tobytes() + scales.tobytes())
        shard = host.layout.head_shard(layer, tp_rank)
        first, last = shard["row_start"], shard["row_stop"]
        host.tables[layer] = HostRows(str(path), first * 256, str(path), weights.size + first * 8, first, last, 256,
                                      True, False)
        host.shards[layer] = shard
        host.slots[layer] = [_TransferSlot(6, 12, 256, "hpu") for _ in range(3)]
        tables[layer] = weights, scales
    if preparation != "compat":
        layers = host.layout.layer_ids
        if preparation == "packet":
            host.c1_packets = [_C1Packet([12, 12], 256, "hpu") for _ in range(3)]
            targets = [packet.targets for packet in host.c1_packets]
            for packet in host.c1_packets:
                assert packet.host.is_pinned("hpu")
                assert not np.shares_memory(*packet.targets)
        else:
            targets = [[host.slots[layer][slot].decode_host.numpy() for layer in layers] for slot in range(3)]
        host.native_c1 = NativeC1Prepare(
            host.history.token_map, host.layout.multipliers, host.layout.primes, host.layout.offsets,
            np.array([host.shards[layer]["head_start"] for layer in layers], dtype=np.int64),
            np.array([host.shards[layer]["head_stop"] for layer in layers], dtype=np.int64), host.history.pad_id,
            [host.tables[layer] for layer in layers], targets)
    pending_results = []
    consumer = torch.hpu.Stream()
    try:
        for request in ("first", "second"):
            host.reset(request)
            for step, count in enumerate((1, 6, 1, 2, 1, 6, 1, 1, 1)):
                tokens = [(step + i) % 16 for i in range(count)]
                with torch.hpu.stream(consumer):
                    ticket = host.prepare(request, tokens, [i == 1 or step == 6 for i in range(count)])
                    assert ticket.packet == (preparation == "packet" and count == 1)
                    if ticket.packet:
                        packet = host.c1_packets[ticket.slot]
                        with pytest.raises(RuntimeError, match="pending"):
                            packet.reuse()
                        with pytest.raises(RuntimeError, match="Stale"):
                            packet.complete(ticket.generation - 1, consumer)
                    assert torch.hpu.current_stream() == consumer
                    with pytest.raises(RuntimeError, match="pending"):
                        host.prepare(request, [1])
                    for index, (layer, value) in enumerate(zip(host.layout.layer_ids, ticket.buffers)):
                        shard = host.shards[layer]
                        ids = ticket.batch.hash_ids[:, index, shard["head_start"]:shard["head_stop"]]
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
        assert host.audit.get("c1_packets", 0) == (12 if preparation == "packet" else 0)
    finally:
        host.close()
