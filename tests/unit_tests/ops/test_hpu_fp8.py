# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import habana_frameworks.torch as htorch
import vllm_gaudi.extension.ops as hpu_ops
from utils import get_data_path, create_row_parallel_linear, create_fused_moe
from vllm_gaudi.extension.ops import (
    apply_block_fp8_linear_hpu,
    fp8_block_linear_postprocess_weights,
)
from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod, HPUFp8MoEMethod
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from safetensors import safe_open
from vllm_gaudi.extension.ops import _use_cguid_dynamic_quant, dynamic_quant


def test_cguid_dynamic_quant_decode_shape_gate(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "1")
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")

    assert _use_cguid_dynamic_quant(torch.empty(32, 64))
    assert not _use_cguid_dynamic_quant(torch.empty(33, 64))
    assert not _use_cguid_dynamic_quant(torch.empty(1, 1, 64))


def test_cguid_dynamic_quant_follows_flashinfer_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", raising=False)
    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN", "1")
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")

    assert _use_cguid_dynamic_quant(torch.empty(32, 64))

    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "0")
    assert not _use_cguid_dynamic_quant(torch.empty(32, 64))


@pytest.mark.parametrize("rows", [1, 8, 16, 32])
def test_cguid_dynamic_quant_matches_decode_reference(monkeypatch, rows):
    torch.manual_seed(37 + rows)
    data = torch.randn(rows, 256, dtype=torch.bfloat16, device="hpu")
    if rows > 1:
        data[0].zero_()
        data[1].fill_(1e-8)

    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "0")
    expected_fp8, expected_scale = dynamic_quant(data)
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "1")
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")
    actual_fp8, actual_scale = dynamic_quant(data)

    assert torch.equal(expected_fp8, actual_fp8)
    torch.testing.assert_close(expected_scale, actual_scale, atol=3e-13, rtol=0)


def test_jit_dynamic_quant_guard(monkeypatch):
    value = torch.empty((1, 1024), dtype=torch.bfloat16)
    monkeypatch.setattr(hpu_ops, "is_hpu_gaudi2", True)
    monkeypatch.setenv("VLLM_HPU_FP8_JIT_DYNAMIC_QUANT", "1")

    assert hpu_ops._use_jit_dynamic_quant(value)
    assert not hpu_ops._use_jit_dynamic_quant(value, single_scale=True)
    assert not hpu_ops._use_jit_dynamic_quant(value.flatten())

    monkeypatch.setenv("VLLM_HPU_FP8_JIT_DYNAMIC_QUANT", "0")
    assert not hpu_ops._use_jit_dynamic_quant(value)


def test_fp8_linear_method(default_vllm_config: None, dist_init, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "0")
    config = {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]}
    oot_quant_config = Fp8Config.from_config(config)

    # Prepare linear layer with oot Fp8LinearMethod
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, Fp8LinearMethod)

    # Weight and weight_scale_inv were extracted from first RowParallelLinear layer of Qwen/Qwen3-8B-FP8
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fp8/linear.safetensors"), framework="pt", device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale_inv.copy_(f.get_tensor("weight_scale_inv"))
    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        # Setting fullgraph to False, because currently there is a graph break
        compile_config = HPUCompileConfig(fullgraph=False)
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of Fp8LinearMethod for given input
    # (Fp8LinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fp8/linear.safetensors"), framework="pt", device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


def test_block_fp8_linear_accepts_non_contiguous_tp_input():
    device = torch.device("hpu")
    base = torch.arange(
        16, dtype=torch.bfloat16, device=device
    ).view(2, 8)
    input_tensor = base[:, :4]
    assert not input_tensor.is_contiguous()

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.eye(4, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(
            (1, 1), dtype=torch.bfloat16, device=device
        ),
        requires_grad=False,
    )
    layer.quant_config = SimpleNamespace(weight_block_size=[4, 4])
    fp8_block_linear_postprocess_weights(layer)

    assert layer._hpu_orig_M == 4
    assert layer._hpu_orig_N == 4
    assert isinstance(layer._hpu_orig_M, int)
    assert isinstance(layer._hpu_orig_N, int)

    output = apply_block_fp8_linear_hpu(
        input_tensor,
        layer,
        block_size=[4, 4],
        do_unpad=True,
    )

    torch.testing.assert_close(output, input_tensor)


@pytest.mark.xfail(reason="Failed due upstream MOE refactor - PR's: 30627, 30825, 31036")
def test_fp8_moe_method(default_vllm_config: None, dist_init, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "0")
    config = {
        'activation_scheme': 'dynamic',
        'modules_to_not_convert': [],
        'fmt': 'e4m3',
        'quant_method': 'fp8',
        'weight_block_size': [128, 128]
    }
    oot_quant_config = Fp8Config.from_config(config)

    # Prepare FusedMoE layer with oot HPUFp8MoEMethod
    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.routed_experts.quant_method, HPUFp8MoEMethod)

    # Weights were extracted from first FusedMoE layer of Qwen/Qwen3-30B-A3B-FP8
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fp8/moe.safetensors"), framework="pt", device="hpu") as f:
        w13_weight = f.get_tensor("w13_weight")
        oot_op.routed_experts.w13_weight.copy_(w13_weight.repeat(128, 1, 1))

        w13_weight_scale_inv = f.get_tensor("w13_weight_scale_inv")
        oot_op.routed_experts.w13_weight_scale_inv.copy_(w13_weight_scale_inv.repeat(128, 1, 1))

        w2_weight = f.get_tensor("w2_weight")
        oot_op.routed_experts.w2_weight.copy_(w2_weight.repeat(128, 1, 1))

        w2_weight_scale_inv = f.get_tensor("w2_weight_scale_inv")
        oot_op.routed_experts.w2_weight_scale_inv.copy_(w2_weight_scale_inv.repeat(128, 1, 1))

    oot_op.routed_experts.quant_method.process_weights_after_loading(oot_op.routed_experts)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of Fp8MoEMethod for given input
    # (Fp8MoEMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fp8/moe.safetensors"), framework="pt", device="hpu") as f:
        hidden_states = f.get_tensor("hidden_states")
        router_logits = f.get_tensor("router_logits")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.forward_impl(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)
