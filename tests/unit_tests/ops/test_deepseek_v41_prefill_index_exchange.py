# SPDX-License-Identifier: Apache-2.0
"""Query ownership through packed exchange, original scoring and state publication."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops import deepseek_v41_prefill_index_exchange as exchange
from vllm_gaudi.ops import deepseek_v41_prefill_index_scores as scores


def fixture(tokens):
    generator = torch.Generator().manual_seed(293)
    query = [torch.randint(-3, 4, (tokens, 8, 128), generator=generator).bfloat16() for _ in range(4)]
    weights = [torch.randint(-2, 3, (tokens, 8), generator=generator).bfloat16() for _ in range(4)]
    return query, weights


def route(monkeypatch, query, weights, rank):
    tokens = query[0].shape[0]
    rows = (tokens + 3) // 4
    group = object()
    packets = [
        F.pad(torch.cat((q, w[..., None]), -1), (0, 0, 0, 0, 0, rows * 4 - tokens)) for q, w in zip(query, weights)
    ]
    calls = []
    monkeypatch.setattr(exchange.dist, "get_world_size", lambda actual: 4 if actual is group else 0)
    monkeypatch.setattr(exchange.dist, "get_rank", lambda actual: rank if actual is group else -1)

    def collective(output, value, *, group):
        assert group is expected_group
        assert torch.equal(value, packets[rank])
        output.copy_(torch.cat([p[rank * rows:(rank + 1) * rows] for p in packets]))
        calls.append(value.numel())

    expected_group = group
    monkeypatch.setattr(exchange.dist, "all_to_all_single", collective)
    result = exchange.exchange_prefill_index_queries(query[rank], weights[rank], rank, group=group)
    assert len(calls) == 1
    return result


@pytest.mark.parametrize("tokens", (1, 5, 8, 13, 1024))
@pytest.mark.parametrize("rank", range(4))
def test_direct_route_equals_gather_then_slice_with_tail_padding(monkeypatch, tokens, rank):
    query, weights = fixture(tokens)
    result = route(monkeypatch, query, weights, rank)
    rows = (tokens + 3) // 4
    all_q = F.pad(torch.cat(query, 1), (0, 0, 0, 0, 0, rows * 4 - tokens))
    all_w = F.pad(torch.cat(weights, 1), (0, 0, 0, rows * 4 - tokens))
    assert torch.equal(result.query, all_q[rank * rows:(rank + 1) * rows])
    assert torch.equal(result.weights, all_w[rank * rows:(rank + 1) * rows])
    assert result.query.is_contiguous() and result.weights.is_contiguous()
    result.validate(tokens, rank)
    with pytest.raises(ValueError, match="ownership"):
        result.validate(tokens, (rank + 1) % 4)


@pytest.mark.parametrize("tokens", (5, 12))
@pytest.mark.parametrize("tied", (False, True))
def test_routed_inputs_preserve_full_index_topk_and_candidate_publication(monkeypatch, tokens, tied):
    query, weights = fixture(tokens)
    if tied:
        weights = [torch.zeros_like(w) for w in weights]
    keys = torch.randint(-2, 3, (512, 128), generator=torch.Generator().manual_seed(997)).bfloat16()
    positions = (torch.arange(tokens, dtype=torch.int32) * 127) % 512

    def scorer(q, w, packed, table, p, rows, ratio, local_heads):
        assert local_heads == 8
        values = scores._weighted_index_scores(q, w, keys[rows.long()], local_heads, False)
        return values.masked_fill(rows[None] >= ((p + 1) // ratio)[:, None], -torch.inf)

    monkeypatch.setattr(scores, "full_prefill_sram_scores", scorer)
    expected, blocks = scores.full_prefill_index_selection(torch.cat(query, 1),
                                                           torch.cat(weights, 1),
                                                           None,
                                                           None,
                                                           positions,
                                                           1,
                                                           512,
                                                           True,
                                                           tensor_parallel_size=4)
    reference = torch.cat((expected, blocks), -1)
    rows = (tokens + 3) // 4
    padded = F.pad(reference, (0, 0, 0, rows * 4 - tokens), value=-1)
    for rank in range(4):
        local = route(monkeypatch, query, weights, rank)

        def gather(packet, dim, rank=rank):
            assert dim == 0
            assert torch.equal(packet, padded[rank * rows:(rank + 1) * rows])
            return padded

        actual, candidates = scores.tp_full_prefill_index_selection(local.query,
                                                                    local.weights,
                                                                    None,
                                                                    None,
                                                                    positions,
                                                                    1,
                                                                    512,
                                                                    True,
                                                                    rank,
                                                                    gather,
                                                                    tensor_parallel_size=4,
                                                                    query_partitioned=True)
        assert torch.equal(actual, expected) and torch.equal(candidates, blocks)


@pytest.mark.parametrize("tokens", (5, 12))
def test_reindex_keeps_candidate_slots_and_causal_positions_local(monkeypatch, tokens):
    query, weights = fixture(tokens)
    positions = torch.arange(tokens, dtype=torch.int32) + 100
    blocks = torch.arange(tokens * 4, dtype=torch.int32).reshape(tokens, 4)
    blocks[:, 1] = blocks[:, 0]
    blocks[:, 2] = -1
    rows = (tokens + 3) // 4
    all_q = F.pad(torch.cat(query, 1), (0, 0, 0, 0, 0, rows * 4 - tokens))
    all_w = F.pad(torch.cat(weights, 1), (0, 0, 0, rows * 4 - tokens))
    all_p = F.pad(positions, (0, rows * 4 - tokens), value=-1)
    all_b = F.pad(blocks, (0, 0, 0, rows * 4 - tokens), value=-1)
    selected = torch.arange(rows * 4 * 512, dtype=torch.int32).reshape(rows * 4, 512)
    for rank in range(4):
        local = route(monkeypatch, query, weights, rank)
        sl = slice(rank * rows, (rank + 1) * rows)

        def consumer(q, w, packed, table, p, b, ratio, source, native_gather, native_scores, tp, sl=sl):
            assert tp == 4 and ratio == 1 and source == 16384
            assert torch.equal(q, all_q[sl]) and torch.equal(w, all_w[sl])
            assert torch.equal(p, all_p[sl]) and torch.equal(b, all_b[sl])
            return selected[sl]

        def gather(packet, dim, sl=sl):
            assert dim == 0 and torch.equal(packet, selected[sl])
            return selected

        monkeypatch.setattr(scores, "prefill_reindex_selection", consumer)
        actual = scores.tp_prefill_reindex_selection(local.query,
                                                     local.weights,
                                                     None,
                                                     None,
                                                     positions,
                                                     blocks,
                                                     1,
                                                     16384,
                                                     rank,
                                                     gather,
                                                     tensor_parallel_size=4,
                                                     query_partitioned=True)
        assert torch.equal(actual, selected[:tokens])


def test_wrong_group_and_dtype_fail_before_collective(monkeypatch):
    query, weights = fixture(8)
    monkeypatch.setattr(exchange.dist, "get_world_size", lambda group: 2)
    with pytest.raises(ValueError, match="four-rank"):
        exchange.exchange_prefill_index_queries(query[0], weights[0], 0, group=object())
    with pytest.raises(ValueError, match="BF16"):
        exchange.exchange_prefill_index_queries(query[0], weights[0].float(), 0, group=object())


def test_normal_prefill_entry_routes_queries_without_all_gather(monkeypatch):
    import vllm.distributed
    from vllm_gaudi.ops import deepseek_v41_paged_attention as attention
    tokens = 1024
    q_weight, w_weight = object(), object()
    owner = SimpleNamespace(index_heads=8,
                            tensor_parallel_size=4,
                            search_length=16384,
                            ratio=1,
                            prefill_tp_rank=2,
                            weights=SimpleNamespace(indexer=SimpleNamespace(wq_b=q_weight, weights_proj=w_weight)),
                            linear=lambda x, w: torch.ones(tokens, 1024 if w is q_weight else 8).bfloat16(),
                            _rope=lambda q, p: q)
    owner.gather = lambda *args: pytest.fail("Partitioned prefill must not all-gather index Q")
    group = SimpleNamespace(rank_in_group=2, device_group=object())
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: group)
    monkeypatch.setattr(attention, "fp4_roundtrip", lambda q, group: q)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP", "1")
    sentinel = object()

    def routed(q, w, rank, *, group):
        assert rank == 2 and group is expected_group
        assert q.shape == (tokens, 8, 128)
        assert torch.equal(w, torch.full_like(w, 1 / 64))
        return sentinel

    expected_group = group.device_group
    monkeypatch.setattr(exchange, "exchange_prefill_index_queries", routed)
    assert attention.PagedCSA2Attention._prepare_index_queries(owner,
                                                               torch.empty(tokens, 1),
                                                               torch.empty(tokens, 1),
                                                               torch.arange(tokens),
                                                               prefill=True) is sentinel
