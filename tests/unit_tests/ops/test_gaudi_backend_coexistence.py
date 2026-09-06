# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_gaudi.models.qwen3_5 as qwen
from vllm_gaudi.extension import ops


def _attention(store_separately=False):
    indices = torch.tensor([0], dtype=torch.int32)
    state = torch.zeros(1, 1, 1, 1)
    conv = torch.zeros(1, 3, 3)
    metadata = (False, conv, state, indices, indices.clone() if store_separately else indices,
                torch.arange(2, dtype=torch.int32), None, None, 1, 0, 0, 0, None, False)
    return SimpleNamespace(
        _extract_metadata=lambda _: metadata,
        in_proj_qkv=lambda _: (torch.ones(1, 3), None),
        in_proj_ba=lambda _: (torch.ones(1, 2), None),
        in_proj_z=lambda _: (torch.ones(1, 1), None),
        head_v_dim=1,
        head_k_dim=1,
        num_v_heads=1,
        tp_size=1,
        conv1d=SimpleNamespace(weight=torch.zeros(3, 1, 4), bias=None),
        compact_state_group_count=None,
        compact_state_group_offset=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        activation="silu",
        _triton_dt_bias_ready=False,
        rearrange_mixed_qkv=lambda _: (torch.zeros(1), ) * 3,
        norm=lambda output, _: output,
        out_proj=lambda output: (output, None),
    )


@pytest.mark.parametrize("mode", ["off", "hybrid", "strict"])
def test_gdn_strict_precedes_flashinfer_and_hybrid_preserves_it(monkeypatch, mode):
    attention = _attention()
    flashinfer = Mock(return_value=(torch.ones(1, 1, 1, 1), None))
    triton = Mock(return_value=torch.full((1, 1, 1), 2.0))
    convolution = Mock(return_value=torch.ones(1, 3))
    vendor = Mock(side_effect=AssertionError("unexpected vendor recurrence"))
    monkeypatch.setattr(qwen, "_triton_gaudi_mode", mode)
    monkeypatch.setattr(qwen, "maybe_run_gdn_fused_decode_step", flashinfer)
    monkeypatch.setattr(qwen, "_try_triton_gdn_decode_packed", triton)
    monkeypatch.setattr(qwen, "hpu_causal_conv1d_update", convolution)
    monkeypatch.setattr(qwen, "hpu_fused_recurrent_gated_delta_rule", vendor)
    result = qwen.HPUGatedDeltaNetAttention.forward(attention, torch.zeros(1, 1))
    torch.testing.assert_close(result, torch.full((1, 1), 2.0 if mode == "strict" else 1.0))
    assert triton.call_count == (mode == "strict")
    assert convolution.call_count == (mode == "strict")
    assert flashinfer.call_count == (mode != "strict")
    vendor.assert_not_called()


def test_gdn_strict_rejects_separate_state_destinations_before_mutation(monkeypatch):
    attention = _attention(store_separately=True)
    convolution = Mock(side_effect=AssertionError("must not mutate convolution state"))
    monkeypatch.setattr(qwen, "_triton_gaudi_mode", "strict")
    monkeypatch.setattr(qwen, "hpu_causal_conv1d_update", convolution)
    with pytest.raises(RuntimeError, match="identical load/store"):
        qwen.HPUGatedDeltaNetAttention.forward(attention, torch.zeros(1, 1))
    convolution.assert_not_called()


@pytest.mark.parametrize("mode", ["off", "hybrid"])
def test_gdn_vendor_remains_available_without_flashinfer(monkeypatch, mode):
    monkeypatch.setattr(qwen, "_triton_gaudi_mode", mode)
    monkeypatch.setattr(qwen, "maybe_run_gdn_fused_decode_step", lambda **_: None)
    monkeypatch.setattr(qwen, "maybe_run_gdn_decode_packed", lambda **_: None)
    monkeypatch.setattr(qwen, "hpu_causal_conv1d_update", lambda **_: torch.ones(1, 3))
    monkeypatch.setattr(qwen, "hpu_fused_recurrent_gated_delta_rule", lambda **_: (torch.full((1, 1, 1, 1), 3.0), None))
    result = qwen.HPUGatedDeltaNetAttention.forward(_attention(), torch.zeros(1, 1))
    torch.testing.assert_close(result, torch.full((1, 1), 3.0))


@pytest.mark.parametrize("triton_selected", [True, False])
def test_quantization_triton_selection_precedes_cguid(monkeypatch, triton_selected):
    data = torch.ones(2, 8, dtype=torch.bfloat16)
    expected = (data, torch.ones(2, 1))
    monkeypatch.setattr(ops, "_triton_dynamic_quant", lambda _: expected if triton_selected else None)
    monkeypatch.setattr(ops, "_use_cguid_dynamic_quant", lambda *_, **__: True)
    calculate = Mock(return_value=torch.ones(2, 1))
    monkeypatch.setattr(ops.torch.ops.hpu, "calculate_scale_for_cast", calculate)
    monkeypatch.setattr(ops.torch.ops.hpu, "cast_to_fp8_v2", Mock(return_value=(data, )))
    actual = ops.dynamic_quant(data)
    if triton_selected:
        assert actual is expected
    assert calculate.call_count == (not triton_selected)


def test_tp2_collective_norm_boundary_precedes_local_triton(monkeypatch):
    import vllm.forward_context
    from vllm_gaudi.distributed import tp2_fused_ar_norm
    from vllm_gaudi.extension import kernels
    from vllm_gaudi.ops import hpu_layernorm

    triton = Mock(side_effect=AssertionError("must not bypass all-reduce"))
    x, residual, weight = torch.ones(1, 8), torch.ones(1, 8), torch.ones(8)
    expected = (x * 3, residual * 2)
    collective = Mock(return_value=expected)
    monkeypatch.setattr(kernels, "rms_norm", lambda: None)
    monkeypatch.setattr(hpu_layernorm, "_fused_add_rms_norm", triton)
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    monkeypatch.setattr(tp2_fused_ar_norm, "tp2_allreduce_residual_rms_norm", collective)
    norm = SimpleNamespace(_hpu_tp2_fused_ar_norm=True, weight=weight, variance_epsilon=1e-6)
    actual = hpu_layernorm.HPURMSNorm.forward_oot(norm, x, residual)
    torch.testing.assert_close(actual, expected)
    collective.assert_called_once()
    triton.assert_not_called()
