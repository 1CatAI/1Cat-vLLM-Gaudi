# SPDX-License-Identifier: Apache-2.0
"""Cold projection replication must preserve shard arithmetic and local views."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
from vllm_gaudi.ops.deepseek_v41_tp4_selection import (
    invalidate_local_index_queries, local_index_query_projections, prepare_local_index_queries)


def owner_with_shards(rank):
    generator = torch.Generator().manual_seed(615)
    query = torch.randn(4, 1024, 1280, generator=generator).bfloat16()
    score = torch.randn(4, 8, 5120, generator=generator).bfloat16()
    indexer = SimpleNamespace(wq_b=SimpleNamespace(weight=query[rank].clone(), scale=torch.empty(0)),
                              weights_proj=SimpleNamespace(weight=score[rank].clone()))
    calls = []

    def gather(value, dim):
        assert dim == 0
        calls.append(tuple(value.shape))
        shards = query if value.shape == (1024, 1280) else score
        return shards.flatten(0, 1).clone()

    owner = SimpleNamespace(tensor_parallel_size=4, owns_index=True, index_heads=8, prefill_tp_rank=rank,
                            weights=SimpleNamespace(indexer=indexer), gather=gather,
                            tp4_local_index_queries=False)
    return owner, query, score, calls


@pytest.mark.parametrize("rank", range(4))
def test_prepare_preserves_local_geometry_without_duplicate_storage(rank):
    owner, query, score, calls = owner_with_shards(rank)
    prepare_local_index_queries(owner, release_local=True)
    assert calls == [(1024, 1280), (8, 5120)]
    assert owner.tp4_local_index_queries
    assert torch.equal(owner.weights.indexer.wq_b.weight, query[rank])
    assert torch.equal(owner.weights.indexer.weights_proj.weight, score[rank])
    assert owner.weights.indexer.wq_b.weight is getattr(owner, f"_tp4_index_query_shard_{rank}")
    assert owner.weights.indexer.weights_proj.weight is getattr(owner, f"_tp4_index_score_shard_{rank}")
    for kind in ("query", "score"):
        tensors = [getattr(owner, f"_tp4_index_{kind}_shard_{i}") for i in range(4)]
        assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == 4
    with pytest.raises(ValueError, match="invalidated"):
        prepare_local_index_queries(owner)


@pytest.mark.parametrize("tokens", [1, 6])
def test_projections_equal_original_rank_order_and_activation_rounding(tokens):
    owner, query, score, _ = owner_with_shards(2)
    prepare_local_index_queries(owner)
    generator = torch.Generator().manual_seed(33)
    value = torch.randn(tokens, 5120, generator=generator).bfloat16()
    qr = torch.randn(tokens, 1280, generator=generator).bfloat16()
    actual_q, actual_w = local_index_query_projections(owner, value, qr)
    expected_q = torch.cat([F.linear(quantize_activation(qr), shard).reshape(tokens, 8, 128)
                            for shard in query], dim=1)
    expected_w = torch.cat([F.linear(value, shard) for shard in score], dim=1)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_w, expected_w)


def test_prepare_rejects_unsupported_owner():
    owner, _, _, _ = owner_with_shards(0)
    owner.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="TP4"):
        prepare_local_index_queries(owner)


@pytest.mark.parametrize("rank", range(4))
def test_invalidation_releases_replicas_and_reload_uses_new_weights(rank):
    owner, query, score, calls = owner_with_shards(rank)
    prepare_local_index_queries(owner, release_local=True)
    previous_local = owner.weights.indexer.wq_b.weight
    invalidate_local_index_queries(owner)
    assert not owner.tp4_local_index_queries
    assert all(getattr(owner, f"_tp4_index_{kind}_shard_{i}") is None
               for kind in ("query", "score") for i in range(4))
    # The ordinary shard remains valid until the checkpoint loader replaces it.
    assert owner.weights.indexer.wq_b.weight is previous_local
    query.add_(1)
    score.add_(2)
    owner.weights.indexer.wq_b.weight = query[rank].clone()
    owner.weights.indexer.weights_proj.weight = score[rank].clone()
    prepare_local_index_queries(owner, release_local=True)
    assert calls == [(1024, 1280), (8, 5120)] * 2
    assert owner.weights.indexer.wq_b.weight is not previous_local
    for i in range(4):
        assert torch.equal(getattr(owner, f"_tp4_index_query_shard_{i}"), query[i])
        assert torch.equal(getattr(owner, f"_tp4_index_score_shard_{i}"), score[i])
