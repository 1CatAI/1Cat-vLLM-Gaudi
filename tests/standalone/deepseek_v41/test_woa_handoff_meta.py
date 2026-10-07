# SPDX-License-Identifier: Apache-2.0
"""C1 WO handoff accepts either TP geometry, while rejecting larger buckets."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(groups):
    return [torch.empty(shape,dtype=dtype,device='meta') for shape,dtype in (
        ((1,groups,4096),torch.bfloat16),
        ((groups,4096,1024),torch.float8_e4m3fn),
        ((groups,1,1024),torch.float32),
        ((5120,groups*1024),torch.float8_e4m3fn),
        ((1,5120),torch.float32))]


@pytest.mark.parametrize('groups',[2,4])
def test_projection_shape_and_functionalization(groups):
    op=torch.ops.custom_op.custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2
    for fn in (op,torch.func.functionalize(op)):
        out=fn(*operands(groups))
        assert out.shape==(1,5120) and out.dtype==torch.bfloat16


@pytest.mark.parametrize('groups',[2,4])
def test_quant_outputs_use_a_global_row_scale(groups):
    op=torch.ops.custom_op.custom_deepseek_v41_woa_scale_dense_quant_gaudi2
    p=torch.empty((groups,1,1024),device='meta')
    a=torch.empty((groups,1,1),device='meta')
    q,s=op(p,p,a)
    assert q.shape==(1,groups*1024) and q.dtype==torch.float8_e4m3fn
    assert s.shape==(1,1) and s.dtype==torch.float32


@pytest.mark.parametrize('index,shape,dtype',[
    (0,(2,2,4096),torch.bfloat16),
    (1,(2,1024,4096),torch.float8_e4m3fn),
    (2,(2,1,1024),torch.bfloat16),
    (3,(5120,4096),torch.float8_e4m3fn),
])
def test_invalid_handoff_contract(index,shape,dtype):
    args=operands(2);args[index]=torch.empty(shape,dtype=dtype,device='meta')
    with pytest.raises(RuntimeError,match='WO handoff'):
        torch.ops.custom_op.custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2(*args)


@pytest.mark.parametrize('groups', [2, 4])
def test_rope_handoff_geometry_and_functionalization(groups):
    args=operands(groups)
    args[0]=torch.empty((1,groups*8,512),device='meta',dtype=torch.bfloat16)
    args.extend([torch.empty(1,device='meta',dtype=torch.int32),
                 torch.empty((524288,64),device='meta',dtype=torch.float32)])
    op=torch.ops.custom_op.custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2
    for fn in (op,torch.func.functionalize(op)):
        out=fn(*args)
        assert out.shape==(1,5120) and out.dtype==torch.bfloat16
    args[-2]=torch.empty(1,device='meta',dtype=torch.int64)
    with pytest.raises(RuntimeError,match='I32 position'):
        op(*args)
