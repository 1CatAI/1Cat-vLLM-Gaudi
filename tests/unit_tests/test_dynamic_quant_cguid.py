from unittest import mock

import torch

from vllm_gaudi.extension import ops


def test_dynamic_quant_cguid_preserves_zero_row_epsilon(monkeypatch):
    data = torch.zeros((2, 8), dtype=torch.bfloat16)
    calculated_scale = torch.zeros((2, 1), dtype=torch.bfloat16)
    calculate = mock.Mock(return_value=calculated_scale)
    cast = mock.Mock(return_value=(data,))
    monkeypatch.setattr(
        ops.gaudi_envs,
        "VLLM_HPU_DYNAMIC_QUANT_CGUID",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        ops.gaudi_envs,
        "VLLM_HPU_DYNAMIC_QUANT_CGUID_MIN_TOKENS",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        ops.torch.ops.hpu,
        "calculate_scale_for_cast",
        calculate,
    )
    monkeypatch.setattr(ops.torch.ops.hpu, "cast_to_fp8_v2", cast)

    quantized, scale = ops.dynamic_quant(data)

    assert quantized is data
    torch.testing.assert_close(
        scale,
        torch.full_like(
            calculated_scale,
            1e-8 / ops.FP8_MAX,
        ).float(),
        rtol=0,
        atol=0,
    )
    calculate.assert_called_once_with(
        data,
        2,
        0,
        -1,
        True,
        float(ops.FP8_MAX),
        1.0,
    )


def test_dynamic_quant_cguid_bypasses_short_inputs(monkeypatch):
    data = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    calculate = mock.Mock()
    cast = mock.Mock(return_value=(data,))
    monkeypatch.setattr(
        ops.gaudi_envs,
        "VLLM_HPU_DYNAMIC_QUANT_CGUID",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        ops.gaudi_envs,
        "VLLM_HPU_DYNAMIC_QUANT_CGUID_MIN_TOKENS",
        2048,
        raising=False,
    )
    monkeypatch.setattr(
        ops.torch.ops.hpu,
        "calculate_scale_for_cast",
        calculate,
    )
    monkeypatch.setattr(ops.torch.ops.hpu, "cast_to_fp8_v2", cast)

    _, scale = ops.dynamic_quant(data)

    calculate.assert_not_called()
    expected = ((data.abs()).max(dim=-1).values + 1e-8) / ops.FP8_MAX
    torch.testing.assert_close(scale, expected.unsqueeze(-1).float())


def test_dynamic_quant_weight_path_bypasses_cguid(monkeypatch):
    data = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    calculate = mock.Mock()
    cast = mock.Mock(return_value=(data,))
    monkeypatch.setattr(
        ops.gaudi_envs,
        "VLLM_HPU_DYNAMIC_QUANT_CGUID",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        ops.torch.ops.hpu,
        "calculate_scale_for_cast",
        calculate,
    )
    monkeypatch.setattr(ops.torch.ops.hpu, "cast_to_fp8_v2", cast)

    _, scale = ops.dynamic_quant(data, use_cguid=False)

    calculate.assert_not_called()
    expected = ((data.abs()).max(dim=-1).values + 1e-8) / ops.FP8_MAX
    torch.testing.assert_close(scale, expected.unsqueeze(-1).float())
