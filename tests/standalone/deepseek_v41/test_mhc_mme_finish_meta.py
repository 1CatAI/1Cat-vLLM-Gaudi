# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


@pytest.mark.parametrize('tokens',[1,2,6])
def test_shared_rrms_layout(tokens):
    x=torch.empty(tokens,20480,device='meta',dtype=torch.bfloat16)
    p=torch.empty(tokens,48,device='meta')
    y=torch.ops.custom_op.custom_deepseek_v41_control_mme_finish_gaudi2(x,p,1e-6)
    assert y.shape==(tokens,25) and y.dtype==torch.float32


@pytest.mark.parametrize('width',[24,25,49])
def test_requires_full_hi_lo_projection(width):
    x=torch.empty(1,20480,device='meta',dtype=torch.bfloat16)
    p=torch.empty(1,width,device='meta')
    with pytest.raises(RuntimeError,match='Invalid control'):
        torch.ops.custom_op.custom_deepseek_v41_control_mme_finish_gaudi2(x,p,1e-6)


@pytest.mark.parametrize('tokens',[1,2,6])
def test_parallel_controller_uses_common_control_layout(tokens):
    x=torch.empty(tokens,20480,device='meta',dtype=torch.bfloat16)
    weight=torch.empty(24,20480,device='meta',dtype=torch.float32)
    y=torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2(x,weight,1e-6)
    assert y.shape==(tokens,25) and y.dtype==torch.float32


@pytest.mark.parametrize('tokens',[1,2,6])
def test_swizzled_controller_layout(tokens):
    x=torch.empty(tokens,20480,device='meta',dtype=torch.bfloat16)
    weight=torch.empty(160,24,128,device='meta',dtype=torch.float32)
    y=torch.ops.custom_op.custom_deepseek_v41_control_rrms_swizzled_bf16_gaudi2(x,weight,1e-6)
    assert y.shape==(tokens,25) and y.dtype==torch.float32
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_control_rrms_swizzled_bf16_gaudi2(x,weight.reshape(24,20480),1e-6)


@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_bf16_weight_controller_keeps_fp32_outputs(tokens):
    x = torch.empty(tokens, 20480, device='meta', dtype=torch.bfloat16)
    weight = torch.empty(24, 20480, device='meta', dtype=torch.bfloat16)
    y = torch.ops.custom_op.custom_deepseek_v41_control_rrms_bf16_weight_gaudi2(x, weight, 1e-20)
    assert y.shape == (tokens, 25) and y.dtype == torch.float32
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_control_rrms_bf16_weight_gaudi2(x, weight.float(), 1e-20)


@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_bf16_weight_dispatch_retains_wider_batch_precision(tokens, monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm_gaudi.ops.deepseek_v41_math import hc_control_and_collapse
    monkeypatch.setenv('VLLM_HPU_DSV41_MHC_PARALLEL_CONTROL', '1')
    calls = []

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            calls.append(str(func))
            return func(*args, **(kwargs or {}))

    residual = torch.empty(tokens, 4, 5120, device='meta', dtype=torch.bfloat16)
    previous = torch.empty(tokens, 4, device='meta')
    weight = torch.empty(24, 20480, device='meta')
    collapsed = torch.empty(tokens, 5120, device='meta', dtype=torch.bfloat16)
    with Observe():
        _, output = hc_control_and_collapse(residual, previous, weight, 1e-20, collapsed_input=collapsed,
                                            bf16_fn=weight.bfloat16())
    assert output.shape == (tokens, 25) and output.dtype == torch.float32
    assert any('control_rrms_bf16_weight' in call for call in calls) == (tokens == 1)
