# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(tokens=1):
    shapes=[(tokens,48),(tokens,20480),(tokens,5120),(5120,),(3,),(24,)]
    return [torch.empty(shape,device='meta',dtype=torch.bfloat16 if i in (1,2,3) else torch.float32)
            for i,shape in enumerate(shapes)]


@pytest.mark.parametrize('tokens',[1,2,6])
def test_explicit_fused_outputs(tokens):
    result=torch.ops.custom_op.custom_deepseek_v41_mhc_mme_gates_norm_gaudi2(*operands(tokens),1e-20)
    assert [(tuple(v.shape),v.dtype) for v in result]==[
        ((tokens,24),torch.float32),((tokens,5120),torch.bfloat16),
        ((tokens,5120),torch.float8_e4m3fn),((tokens,1),torch.float32)]


@pytest.mark.parametrize('index',[0,1,2,3,4,5])
def test_dtype_contract(index):
    values=operands();values[index]=values[index].double()
    with pytest.raises(RuntimeError,match='contract'):
        torch.ops.custom_op.custom_deepseek_v41_mhc_mme_gates_norm_gaudi2(*values,1e-20)


def test_c1_mme_high_low_shape():
    x=torch.empty((1,20480),device='meta',dtype=torch.bfloat16)
    weight=torch.empty((48,20480),device='meta',dtype=torch.bfloat16)
    y=torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(x,weight)
    assert y.dtype==torch.float32 and y.shape==(1,48)
