# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_grouped_prefill import route_batches


@pytest.mark.parametrize("tokens,skew", [(1, False), (73, False), (8192, False), (8192, True)])
def test_every_route_restored_once_under_workspace_bound(tokens, skew):
    ids = np.random.default_rng(41).integers(0, 384, (tokens, 6), dtype=np.int32)
    if skew:
        ids.fill(17)
    reconstructed = np.full(ids.size, -1, dtype=np.int32)
    visits = np.zeros(ids.size, dtype=np.int32)
    for experts, slots, valid in route_batches(ids, 384):
        assert slots.shape[0] <= 8
        assert slots.size <= 8192
        dest = slots.reshape(-1)[valid]
        source_experts = np.broadcast_to(experts.T, slots.shape).reshape(-1)[valid]
        reconstructed[dest] = source_experts
        visits[dest] += 1
    np.testing.assert_array_equal(reconstructed, ids.reshape(-1))
    assert np.all(visits == 1)


@pytest.mark.parametrize("invalid", [-1, 384])
def test_invalid_routing_is_rejected(invalid):
    ids = np.zeros((1, 6), dtype=np.int32)
    ids[0, 2] = invalid
    with pytest.raises(ValueError, match="out-of-range"):
        list(route_batches(ids, 384))


def test_large_prefill_reaches_experts_as_one_scheduler_transaction(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.models.deepseek_v41_program import PreparedStage

    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED", "1")
    calls = []

    def execute(*args):
        calls.append(args[0].shape[0])
        return "complete"

    stage = SimpleNamespace(_forward_impl=execute)
    # This exercises the actual model dispatch, including the long-context
    # entry that bypasses CompiledStage for ordinary prefill.
    residual = SimpleNamespace(shape=(8192, 4, 5120))
    assert PreparedStage.forward(stage, residual, None, None, None, ()) == "complete"
    assert calls == [8192]


def test_dedicated_entrypoint_selects_validated_grouped_prefill():
    from vllm_gaudi.entrypoints.deepseek_v41 import _C1_FASTPATH_DEFAULTS

    assert _C1_FASTPATH_DEFAULTS["VLLM_HPU_DSV41_PREFILL_GROUPED"] == "1"
    assert "VLLM_HPU_DSV41_PREFILL_MXFP4" not in _C1_FASTPATH_DEFAULTS


def test_prompt_lengths_do_not_exhaust_shared_dynamo_recompile_limit(monkeypatch):
    import torch
    from vllm_gaudi.ops import deepseek_v41_grouped_prefill as grouped

    compile_calls = []
    original_compile = torch.compile

    def compile_cpu(entry, **options):
        compile_calls.append(entry.__code__)
        options["backend"] = "eager"
        return original_compile(entry, **options)

    grouped.compiled_reduce.cache_clear()
    monkeypatch.setattr(torch, "compile", compile_cpu)
    generator = torch.Generator().manual_seed(41)
    try:
        with torch._dynamo.config.patch(recompile_limit=2):
            for tokens in range(7, 19):
                signature = (tokens, 32)
                fn = grouped.compiled_reduce(signature)
                for _ in range(2):
                    value = torch.randn(tokens, 6, 32, generator=generator).bfloat16()
                    assert torch.equal(fn(value), grouped.ordered_reduce(value))
                assert grouped.compiled_reduce(signature) is fn
        assert len(compile_calls) == 12
        assert len({id(code) for code in compile_calls}) == 12
    finally:
        grouped.compiled_reduce.cache_clear()
