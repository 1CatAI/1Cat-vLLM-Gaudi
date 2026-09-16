# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_gaudi.ops.hpu_modelopt import (
    HPUModelOptFp8Config,
    HPUModelOptFp8LinearMethod,
    HPUModelOptFp8PcPtLinearMethod,
)


def test_modelopt_selects_hpu_per_channel_per_token_method():
    config = HPUModelOptFp8Config(
        quant_method="FP8_PER_CHANNEL_PER_TOKEN",
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )

    assert config.LinearMethodCls is HPUModelOptFp8PcPtLinearMethod
    assert config.get_supported_act_dtypes() == [torch.bfloat16]


def test_modelopt_keeps_static_fp8_method():
    config = HPUModelOptFp8Config(
        quant_method="FP8",
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )

    assert config.LinearMethodCls is HPUModelOptFp8LinearMethod


def test_modelopt_rejects_unimplemented_fp8_algorithms():
    with pytest.raises(ValueError, match="FP8_PB_WO"):
        HPUModelOptFp8Config(
            quant_method="FP8_PB_WO",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        )


def test_pcpt_postload_keeps_channel_scales_and_transposes_weight():
    method = HPUModelOptFp8PcPtLinearMethod(SimpleNamespace())
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32), requires_grad=False)
    expected_weight = layer.weight.detach().t()
    expected_scale = layer.weight_scale.detach().clone()

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (4, 6)
    assert torch.equal(layer.weight.float(), expected_weight.float())
    assert torch.equal(layer.weight_scale, expected_scale)


def test_pcpt_apply_uses_dynamic_native_fp8_contract():
    method = HPUModelOptFp8PcPtLinearMethod(SimpleNamespace())
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.empty(4, 6, dtype=torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.ones(6, dtype=torch.float32), requires_grad=False)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    expected = torch.randn(6, 6, dtype=torch.bfloat16)

    with patch("vllm_gaudi.ops.hpu_modelopt.hpu_ops.apply_fp8_linear_hpu", return_value=expected) as fp8_gemm:
        output = method.apply(layer, x)

    assert output.shape == (2, 3, 6)
    call = fp8_gemm.call_args.kwargs
    assert call["input"].shape == (6, 4)
    assert call["weight"] is layer.weight
    assert call["weight_scale"] is layer.weight_scale
    assert call["input_scale"] is None
    assert call["trans_B"] is False
