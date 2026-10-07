# SPDX-License-Identifier: Apache-2.0
"""Peer/post/norm fusion retains the C1 residual and expert-input contracts."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(ranks=4):
    return [torch.empty(shape,dtype=dtype,device='meta') for shape,dtype in (
        ((ranks,1,5120),torch.bfloat16),((1,4,5120),torch.bfloat16),
        ((1,4),torch.float32),((1,4,4),torch.float32),((1,4),torch.float32),
        ((5120,),torch.bfloat16))]


@pytest.mark.parametrize('ranks',[1,2,4])
def test_tp_geometry_and_functionalization(ranks):
    op=torch.ops.custom_op.custom_deepseek_v41_peer_post_norm_quant_gaudi2
    for fn in (op,torch.func.functionalize(lambda *args:op(*args,1e-20))):
        outputs=fn(*operands(ranks),1e-20) if fn is op else fn(*operands(ranks))
        assert [tuple(t.shape) for t in outputs]==[(1,4,5120),(1,5120),(1,5120),(1,5120),(1,1)]
        assert [t.dtype for t in outputs]==[torch.bfloat16]*3+[torch.float8_e4m3fn,torch.float32]


@pytest.mark.parametrize('index,shape,dtype',[
    (0,(4,2,5120),torch.bfloat16),
    (1,(1,4,5120),torch.float32),
    (2,(1,4),torch.bfloat16),
    (3,(1,16),torch.float32),
    (4,(2,4),torch.float32),
    (5,(1280,),torch.bfloat16),
])
def test_invalid_geometry_rejected(index,shape,dtype):
    args=operands();args[index]=torch.empty(shape,dtype=dtype,device='meta')
    with pytest.raises(RuntimeError,match='Peer post/norm'):
        torch.ops.custom_op.custom_deepseek_v41_peer_post_norm_quant_gaudi2(*args,1e-20)


@pytest.mark.parametrize('eps',[0.,-1.,float('nan'),float('inf')])
def test_invalid_epsilon(eps):
    with pytest.raises(RuntimeError,match='epsilon'):
        torch.ops.custom_op.custom_deepseek_v41_peer_post_norm_quant_gaudi2(*operands(),eps)
