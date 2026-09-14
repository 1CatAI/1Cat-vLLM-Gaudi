# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_gaudi.omni import minimax_h3_ops


def _inputs(dtype: torch.dtype = torch.bfloat16):
    generator = torch.Generator().manual_seed(2101)
    x = torch.randn(7, 16, generator=generator, dtype=dtype)
    other = torch.randn(7, 16, generator=generator, dtype=dtype)
    shift = torch.randn(5, 16, generator=generator, dtype=dtype)
    scale = torch.randn(5, 16, generator=generator, dtype=dtype)
    gate = torch.randn(5, 16, generator=generator, dtype=dtype)
    weight = torch.randn(16, generator=generator, dtype=dtype)
    indices = torch.tensor([4, 0, 1, 4, 3, 2, 0], dtype=torch.long)
    return x, other, shift, scale, gate, weight, indices


def test_indexed_modulation_matches_fp32_accumulation():
    x, other, shift, scale, gate, _, indices = _inputs()
    expected_affine = (x.float() * (1.0 + scale.index_select(0, indices).float()) +
                       shift.index_select(0, indices).float()).to(x.dtype)
    disposable = x.clone()

    actual_affine = minimax_h3_ops.indexed_scale_shift_(disposable, shift, scale, indices)
    actual_gate = minimax_h3_ops.indexed_gate(x, gate, other, indices)
    expected_gate = (x.float() + gate.index_select(0, indices).float() * other.float()).to(x.dtype)

    assert actual_affine.data_ptr() == disposable.data_ptr()
    torch.testing.assert_close(actual_affine, expected_affine, rtol=0, atol=0)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)


def test_rms_norm_modulation_matches_reference():
    x, other, shift, scale, gate, weight, indices = _inputs()
    eps = 1e-5
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    normalized = (normalized * weight.float()).to(x.dtype)
    expected = (normalized.float() * (1.0 + scale.index_select(0, indices).float()) +
                shift.index_select(0, indices).float()).to(x.dtype)

    actual = minimax_h3_ops.rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
    residual, modulated = minimax_h3_ops.indexed_gate_rms_norm_scale_shift(x, gate, other, weight, shift, scale,
                                                                           indices, eps)
    expected_residual = (x.float() + gate.index_select(0, indices).float() * other.float()).to(x.dtype)
    expected_normalized = expected_residual.float() * torch.rsqrt(
        expected_residual.float().square().mean(-1, keepdim=True) + eps)
    expected_normalized = (expected_normalized * weight.float()).to(x.dtype)
    expected_modulated = (expected_normalized.float() * (1.0 + scale.index_select(0, indices).float()) +
                          shift.index_select(0, indices).float()).to(x.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(modulated, expected_modulated, rtol=0, atol=0)


def test_install_rebinds_transformer_imports():
    from vllm_gaudi.omni.minimax_h3 import install_minimax_h3_patches

    install_minimax_h3_patches()
    from vllm_omni.diffusion.attention.ops import minimax_h3_modulation
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer

    for name in minimax_h3_ops.__all__:
        implementation = getattr(minimax_h3_ops, name)
        assert getattr(minimax_h3_modulation, name) is implementation
        assert getattr(minimax_h3_transformer, name) is implementation
