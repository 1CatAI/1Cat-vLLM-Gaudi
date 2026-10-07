# SPDX-License-Identifier: Apache-2.0
"""Cold rank ordering, lifecycle and C1-only gain exchange elimination."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_index_gain import prepare_index_gain_weight
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


def owner_for(tp):
    generator = torch.Generator().manual_seed(41)
    gain = torch.randn(32, 16, generator=generator).bfloat16()
    query = torch.randn(32 // tp * 128, 16, generator=generator).bfloat16()
    owner = SimpleNamespace(
        owns_index=True, tensor_parallel_size=tp, index_heads=32 // tp, _index_gain_weight=None,
        native_rope=False,
        weights=SimpleNamespace(indexer=SimpleNamespace(
            weights_proj=SimpleNamespace(weight=gain[:32 // tp].clone()), wq_b=SimpleNamespace(weight=query))),
        gather=lambda value, dim: gain.clone(),
    )
    return owner, gain


@pytest.mark.parametrize("tp", (1, 2, 4))
def test_cold_gain_replica_order_and_invalidation(tp):
    owner, gain = owner_for(tp)
    local = owner.weights.indexer.weights_proj.weight
    prepare_index_gain_weight(owner)
    torch.testing.assert_close(owner._index_gain_weight, gain, rtol=0, atol=0)
    assert owner.weights.indexer.weights_proj.weight is local
    with pytest.raises(ValueError, match="invalidated"):
        prepare_index_gain_weight(owner)
    PagedCSA2Attention.invalidate_index_gain_weight(owner)
    prepare_index_gain_weight(owner)


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("tokens,prefill,request_batch,exchanges", (
    (1, False, False, 1), (1, True, False, 2), (1, False, True, 2),
    (2, False, False, 2), (6, False, False, 2),
))
def test_only_c1_removes_gain_exchange(monkeypatch, tp, tokens, prefill, request_batch, exchanges):
    from vllm_gaudi.ops import deepseek_v41_paged_attention as module

    owner, gain = owner_for(tp)
    prepare_index_gain_weight(owner)
    value = torch.arange(tokens * 16).reshape(tokens, 16).bfloat16() / 64
    calls = []

    def gather(tensor, dim):
        calls.append(tuple(tensor.shape))
        if tensor.ndim == 2:
            return F.linear(value, gain) * (128**-.5 * 32**-.5)
        return torch.cat([tensor] * tp, dim)

    owner.gather = gather
    owner.linear = lambda x, projection: F.linear(x, projection.weight)
    owner._rope = lambda q, p, request_batch: q
    monkeypatch.setattr(module, "fp4_roundtrip", lambda q, group: q)
    monkeypatch.setattr(module.gaudi_envs, "VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP", False)
    q, weights = PagedCSA2Attention._prepare_index_queries(
        owner, value, value, torch.arange(tokens), prefill=prefill, request_batch=request_batch
    )
    assert len(calls) == exchanges
    assert q.shape == (tokens, 32, 128)
    torch.testing.assert_close(weights, F.linear(value, gain) * (128**-.5 * 32**-.5), rtol=0, atol=0)


@pytest.mark.parametrize("bad", ("scale", "bias", "dtype", "heads", "gather"))
def test_rejects_unqualified_projection_contract(bad):
    owner, _ = owner_for(4)
    projection = owner.weights.indexer.weights_proj
    if bad == "scale":
        projection.scale = torch.ones(1)
    elif bad == "bias":
        projection.bias = torch.zeros(8)
    elif bad == "dtype":
        projection.weight = projection.weight.float()
    elif bad == "heads":
        owner.index_heads = 16
    else:
        owner.gather = lambda value, dim: value
    with pytest.raises(ValueError):
        prepare_index_gain_weight(owner)


def test_non_index_layer_does_not_collect():
    owner = SimpleNamespace(owns_index=False)
    prepare_index_gain_weight(owner)


@pytest.mark.parametrize("tokens", (1, 2, 6))
@pytest.mark.parametrize("replicated", (False, True))
def test_native_topology_counts_the_collectives_actually_recorded(monkeypatch, tokens, replicated):
    from vllm_gaudi.models import deepseek_v41_program as program_module
    from vllm_gaudi.ops import deepseek_v41_replay as replay_module

    monkeypatch.setattr(program_module, "CompiledStage", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(replay_module, "stage_state_tensors", lambda program: ())
    layers = [SimpleNamespace(layer=8 + i, attention=SimpleNamespace(
        owns_index=i == 0, search_length=8192, ratio=2,
        _index_gain_weight=torch.empty(32, 16) if replicated and i == 0 else None,
    )) for i in range(4)]
    program = SimpleNamespace(layers=layers, dspark=False, pp_rank=0, length=8192)
    variant = replay_module.StageVariant(
        program, torch.empty(tokens, 4, 5120), torch.empty(tokens, 4),
        torch.arange(tokens), torch.zeros(tokens, dtype=torch.int32), (),
    )
    assert variant.adapter.collectives == (9 if replicated and tokens == 1 else 10)
