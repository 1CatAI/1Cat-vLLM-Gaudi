# SPDX-License-Identifier: Apache-2.0
"""Query/head ownership, sinks and TP group routing for the exchange helper."""
import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops import deepseek_v41_prefill_sequence as sequence


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("chunk", (8, 16))
@pytest.mark.parametrize("fused_layout", (False, True))
@pytest.mark.parametrize("retire_chunks", (False, True))
def test_exchange_preserves_query_head_and_sink_ownership(monkeypatch, rank, chunk, fused_layout, retire_chunks):
    group = object()
    tokens, heads, width = 16, 16, 512
    rows = chunk // 4
    monkeypatch.setattr(sequence, "SEQUENCE_QUERY_CHUNK", chunk)
    base = torch.arange(tokens * heads * width).reshape(tokens, heads, width).remainder(31)
    queries = [(base + peer * 32).bfloat16() for peer in range(4)]
    sinks = torch.arange(64).float() / 8
    indices = torch.arange(tokens, dtype=torch.int32)[:, None].repeat(1, 3)
    expected = [
        (q.float() + sinks[p * heads:(p + 1) * heads][None, :, None] + indices[:, :1, None].float() / 16).bfloat16()
        for p, q in enumerate(queries)
    ]
    calls, destinations = [], []
    drained = []
    monkeypatch.setattr(sequence, "_retire_stream", lambda: drained.append(len(calls)))

    def world_size(actual):
        assert actual is group
        return 4

    def local_rank(actual):
        assert actual is group
        return rank

    def exchange(output, value, *, group):
        assert group is expected_group
        begin = len(calls) // 2 * chunk
        if len(calls) % 2 == 0:
            assert torch.equal(value, queries[rank][begin:begin + chunk])
            output.copy_(torch.cat([q[begin + rank * rows:begin + (rank + 1) * rows] for q in queries]))
        else:
            # Each destination owns one original head shard. Check the data
            # before simulating its peers' return packets, so a wrong forward
            # and inverse transpose cannot cancel and conceal an error.
            packed = torch.cat([q[begin + rank * rows:begin + (rank + 1) * rows] for q in expected])
            assert torch.equal(value, packed)
            assert output.is_contiguous()
            destinations.append((output.untyped_storage().data_ptr(), output.storage_offset()))
            output.copy_(expected[rank][begin:begin + chunk])
        calls.append(value.shape)

    def consumer(query, cache, selected, sink):
        del cache
        begin = len(calls) // 2 * chunk
        assert torch.equal(selected, indices[begin + rank * rows:begin + (rank + 1) * rows])
        return (query.float() + sink[None, :, None] + selected[:, :1, None].float() / 16).bfloat16()

    expected_group = group
    monkeypatch.setattr(sequence.dist, "get_world_size", world_size)
    monkeypatch.setattr(sequence.dist, "get_rank", local_rank)
    monkeypatch.setattr(sequence.dist, "all_to_all_single", exchange)
    monkeypatch.setattr(sequence, "_compiled_mla64", lambda signature: consumer)
    # Exercise the actual layout producer/consumer around a CPU arithmetic
    # stand-in. This catches reversed peer/head/query order independently.
    monkeypatch.setattr(sequence, "_mla64", consumer)
    monkeypatch.setattr(sequence, "_compiled_exchanged_mla64", lambda signature: sequence._exchanged_mla64)
    if not retire_chunks and tokens > chunk:
        with pytest.raises(ValueError, match="one exchange chunk"):
            sequence.sequence_prefill_mla(queries[rank],
                                          torch.zeros(1, 512).bfloat16(),
                                          indices,
                                          sinks,
                                          rank,
                                          group=group,
                                          fused_layout=fused_layout,
                                          retire_chunks=False)
        assert not calls and not drained
        return
    result = sequence.sequence_prefill_mla(queries[rank],
                                           torch.zeros(1, 512).bfloat16(),
                                           indices,
                                           sinks,
                                           rank,
                                           group=group,
                                           fused_layout=fused_layout,
                                           retire_chunks=retire_chunks)
    assert len(calls) == 2 * (tokens // chunk) and torch.equal(result, expected[rank])
    assert drained == (list(range(2, len(calls) + 1, 2)) if retire_chunks else [])
    assert destinations == [(result.untyped_storage().data_ptr(), start * heads * width)
                            for start in range(0, tokens, chunk)]


def test_wrong_group_fails_before_communication(monkeypatch):
    monkeypatch.setattr(sequence.dist, "get_world_size", lambda group: 8)
    with pytest.raises(ValueError, match="four-rank TP group"):
        sequence.sequence_prefill_mla(torch.zeros(4, 16, 512),
                                      torch.zeros(1, 512),
                                      torch.zeros(4, 1),
                                      torch.zeros(64),
                                      0,
                                      group=object())


@pytest.mark.parametrize("tokens,search", ((4096, 16384), (16384, 16384), (16384, 32768)))
def test_serving_uses_tp_group_and_caches_only_immutable_sinks(monkeypatch, tokens, search):
    import vllm.distributed
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
    monkeypatch.setenv("VLLM_HPU_DSV41_FLASHINFER_PREFILL", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE", "1")
    group = SimpleNamespace(rank_in_group=2, device_group=object())
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: group)
    calls, gathers = [], []

    def gather(sink, dim):
        gathers.append((sink, dim))
        return torch.zeros(64, device="meta")

    def exchanged(query, cache, indices, sink, rank, **kwargs):
        assert rank == 2 and kwargs["group"] is group.device_group
        assert kwargs["retire_chunks"] is False
        calls.append(sink)
        return query

    monkeypatch.setattr(sequence, "sequence_prefill_mla", exchanged)
    owner = SimpleNamespace(tensor_parallel_size=4,
                            search_length=search,
                            _prefill_sequence_sink=None,
                            weights=SimpleNamespace(attn_sink=torch.zeros(16, device="meta")),
                            gather=gather)
    q = torch.zeros(tokens, 16, 512, device="meta")
    cache = torch.zeros(16384, 512, device="meta")
    ids = torch.zeros(tokens, 640, device="meta", dtype=torch.int32)
    for _ in range(2):
        assert PagedCSA2Attention._prefill_sparse(owner, q, cache, ids) is q
    assert len(gathers) == 1 and calls[0] is calls[1]


@pytest.mark.parametrize("tokens,heads,columns,tp,search", [
    (16384, 16, 128, 4, 16384),
    (8192, 16, 640, 4, 16384),
    (4096, 16, 640, 4, 32768),
    (16384, 16, 640, 4, 65536),
    (16384, 32, 640, 2, 16384),
])
def test_unqualified_shapes_keep_ordinary_mla(monkeypatch, tokens, heads, columns, tp, search):
    from vllm_gaudi.ops import deepseek_v41_paged_attention as attention
    monkeypatch.setenv("VLLM_HPU_DSV41_FLASHINFER_PREFILL", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_MLA_SEQUENCE", "1")
    called = []

    def ordinary(query, cache, indices, sink):
        called.append(query.shape[0])
        return query

    monkeypatch.setattr(attention, "compiled_flash_prefill_mla", lambda signature: ordinary)
    owner = SimpleNamespace(tensor_parallel_size=tp,
                            search_length=search,
                            layer=0,
                            weights=SimpleNamespace(attn_sink=torch.zeros(heads, device="meta")))
    q = torch.zeros(tokens, heads, 512, device="meta")
    cache = torch.zeros(16384, 512, device="meta")
    ids = torch.zeros(tokens, columns, device="meta", dtype=torch.int32)
    result = attention.PagedCSA2Attention._prefill_sparse(owner, q, cache, ids)
    assert result.shape == q.shape and sum(called) == tokens
