# SPDX-License-Identifier: Apache-2.0
"""Request-isolated hash → host shard → pinned DMA → device consumer."""
import os

import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory
from vllm_gaudi.ops.deepseek_v41_host import EngramHost, _TransferSlot

pytestmark = pytest.mark.skipif(os.getenv("DSV41_TEST_HPU") != "1", reason="An HPU module lease is required")


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("direct,native_c1", [(False, False), (True, False), (True, True)])
@torch.inference_mode()
def test_request_batch_dma_reuse(tmp_path, tp_rank, direct, native_c1, monkeypatch):
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_host import host_native
    from vllm_gaudi import envs

    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_FUSED_STAGE_IO", False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_ENGRAM_PACKED_STAGING", direct)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BATCH_C1_PREPARE", native_c1)
    host = EngramHost.__new__(EngramHost)
    host.layout = EngramHashLayout.from_config({
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [10000, 10000],
        "engram_max_ngram_size": 4,
        "engram_n_heads": 8,
        "engram_compressed_vocab_size": 8,
        "engram_vocab_size": 5,
        "engram_pad_token_id": 2,
        "engram_head_dim": 256,
    })
    host.history = EngramTokenHistory(host.layout, np.arange(16) % 8)
    host.tables, host.shards, host.slots, host.histories = {}, {}, {}, {}
    host.stream = torch.hpu.Stream()
    host.generation, host.pending, host.closed = 0, None, False
    host.ready_ticket, host.profile_records, host.batches = None, None, None
    host.native_c1 = host.device_c1 = host.c1_packets = None
    host.c1_abi = 2
    host.max_tokens, host.ring_size = 64, 2
    host.audit = dict(gathers=0, major_faults=0, dma_bytes=0, generations=0)
    tables, outputs = {}, []
    for layer in host.layout.layer_ids:
        weights = np.arange(10000 * 256, dtype=np.uint8).reshape(10000, 256)
        weights ^= np.arange(10000, dtype=np.uint8)[:, None]
        scales = np.arange(10000 * 8, dtype=np.uint8).reshape(10000, 8)
        path = tmp_path / f"layer-{layer}.bin"
        path.write_bytes(weights.tobytes() + scales.tobytes())
        shard = host.layout.head_shard(layer, tp_rank)
        first, last = shard["row_start"], shard["row_stop"]
        host.tables[layer] = host_native().HostRows(str(path), first * 256, str(path), weights.size + first * 8, first,
                                                    last, 256, True, False)
        host.shards[layer] = shard
        host.slots[layer] = [_TransferSlot(64, 12, 256, "hpu") for _ in range(2)]
        tables[layer] = weights, scales
    if native_c1:
        layers = host.layout.layer_ids
        host.native_c1 = host_native().NativeC1Prepare(
            host.history.token_map, host.layout.multipliers, host.layout.primes, host.layout.offsets,
            np.array([host.shards[layer]["head_start"] for layer in layers], dtype=np.int64),
            np.array([host.shards[layer]["head_stop"] for layer in layers],
                     dtype=np.int64), host.history.pad_id, [host.tables[layer] for layer in layers],
            [[host.slots[layer][slot].decode_host.numpy() for layer in layers] for slot in range(2)])
    consumer = torch.hpu.Stream()
    starts = {}
    try:
        for step, count in enumerate((1, 7, 64, 3, 16, 1, 64, 1, 1)):
            requests = [f"r{i}" for i in reversed(range(count))]
            spans = [(rid, starts.get(rid, 0), [(step + i) % 16], [i == 2 or step == 7])
                     for i, rid in enumerate(requests)]
            capacity = 1 << (count - 1).bit_length()
            with torch.hpu.stream(consumer):
                ticket = host.prepare_batch(spans, capacity=capacity, defer_wait=True)
                with pytest.raises(RuntimeError, match="idle"):
                    host.prepare_batch(spans, capacity=capacity)
                host.wait(ticket)
                for index, (layer, value) in enumerate(zip(host.layout.layer_ids, ticket.buffers)):
                    shard = host.shards[layer]
                    ids = ticket.batch.hash_ids[:, index, shard["head_start"]:shard["head_stop"]]
                    w, s = tables[layer]
                    expected = np.zeros((capacity, 12, 264), dtype=np.uint8)
                    expected[:count] = np.concatenate((w[ids], s[ids]), axis=-1)
                    outputs.append((value.clone(), torch.from_numpy(expected)))
                committed = [0 if step in (3, 7) and i == 0 else 1 for i in range(count)]
                host.complete_batch(ticket, committed)
                for rid, kept in zip(requests, committed):
                    starts[rid] = starts.get(rid, 0) + kept
                with pytest.raises(RuntimeError, match="Stale"):
                    host.complete_batch(ticket, committed)
        for actual, expected in outputs:
            assert torch.equal(actual.cpu(), expected)
        assert host.audit["request_batches"] == 9
        assert host.audit["gathers"] == 18
        assert host.audit.get("native_request_c1", 0) == (4 if native_c1 else 0)
        for rid, expected in starts.items():
            assert host.histories[rid].position == expected
    finally:
        host.close()
