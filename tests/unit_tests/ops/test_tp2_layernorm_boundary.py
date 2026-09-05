# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm, HPURMSNorm


@pytest.mark.parametrize("norm_type", [HPURMSNorm, HPUGemmaRMSNorm])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_prompt", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_tp2_norm_boundary_preserves_collective_and_weight(monkeypatch, norm_type, weight_dtype, is_prompt, deferred):
    import vllm.forward_context as forward_context
    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module
    import vllm_gaudi.extension.kernels as kernels

    calls = []

    class FakeRMSNorm:

        @staticmethod
        def apply(x, weight, epsilon):
            calls.append((weight.clone(), epsilon))
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
            return (normalized * weight).to(x.dtype)

    reductions = []
    boundary_options = []
    original_boundary = fused_module.tp2_allreduce_residual_rms_norm

    def boundary(*args, **kwargs):
        boundary_options.append(kwargs)
        return original_boundary(*args, **kwargs)

    def all_reduce(x):
        reductions.append(x.clone())
        return x + peer

    monkeypatch.setattr(kernels, "rms_norm", lambda: FakeRMSNorm)
    monkeypatch.setattr(fused_module, "rms_norm", lambda: FakeRMSNorm)
    monkeypatch.setattr(fused_module, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(fused_module, "tp2_allreduce_residual_rms_norm", boundary)
    monkeypatch.setattr(fused_module, "get_config", lambda: SimpleNamespace(tp2_fused_ar_norm_max_bytes=524288))
    monkeypatch.setattr(forward_context, "get_forward_context",
                        lambda: SimpleNamespace(attn_metadata=SimpleNamespace(is_prompt=is_prompt)))
    # CPU forces the production HCCL fallback, so a missing collective is
    # observable without requiring a two-device process group in unit tests.
    weight = torch.linspace(-0.25, 0.5, 8).to(weight_dtype)
    partial = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 16
    peer = torch.full((1, 2, 8), 0.5, dtype=torch.bfloat16)
    residual = torch.full_like(peer, 0.25)
    original_residual = residual.clone()
    norm = SimpleNamespace(weight=weight, variance_epsilon=1e-6, _hpu_tp2_fused_ar_norm=deferred)

    normalized, residual_out = norm_type.forward_oot(norm, partial, residual)

    expected_residual = original_residual + (partial.reshape(residual.shape) +
                                             peer if deferred else partial.reshape(residual.shape))
    expected_weight = weight + 1.0 if norm_type is HPUGemmaRMSNorm else weight
    expected_normalized = FakeRMSNorm.apply(expected_residual, expected_weight, norm.variance_epsilon)
    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(normalized, expected_normalized.reshape(partial.shape), atol=0, rtol=0)
    torch.testing.assert_close(residual, original_residual, atol=0, rtol=0)
    assert len(reductions) == int(deferred)
    assert boundary_options == ([{"is_prompt": is_prompt, "allow_fused": norm_type is HPURMSNorm}] if deferred else [])
    torch.testing.assert_close(calls[0][0], expected_weight, atol=0, rtol=0)
    assert calls[0][0].dtype == weight_dtype
    assert normalized.dtype == partial.dtype
