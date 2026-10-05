# SPDX-License-Identifier: Apache-2.0
"""Fused Q/KV contracts and functionalization preserve scheduler cache ownership."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(width=8192):
    def tensor(shape,dtype):return torch.empty(shape,dtype=dtype,device='meta')
    return [tensor((1,1280),torch.bfloat16),tensor((1280,),torch.bfloat16),
            tensor((1,512),torch.bfloat16),tensor((512,),torch.bfloat16),
            tensor((1,),torch.int32),tensor((524288,64),torch.float32),
            tensor((256,528),torch.uint8),tensor((1024,512),torch.bfloat16),
            tensor((width,1280),torch.float8_e4m3fn),tensor((1,width),torch.float32)]


@pytest.mark.parametrize('width',[8192,16384])
@pytest.mark.parametrize('offset',[-1,0,512])
def test_tp_local_shapes_and_cache_state(width,offset):
    args=operands(width)
    out=torch.ops.custom_op.custom_deepseek_v41_qkv_projection_publish_gaudi2(*args,1e-20,offset)
    assert [tuple(t.shape) for t in out]==[(1,width),(1,512),(16,),(1,1280)]
    assert [t.dtype for t in out]==[torch.bfloat16,torch.bfloat16,torch.int32,torch.bfloat16]


def test_functionalized_meta_uses_ordered_dispatch():
    op=torch.ops.custom_op.custom_deepseek_v41_qkv_projection_publish_gaudi2
    out=torch.func.functionalize(lambda *args:op(*args,1e-20,-1))(*operands())
    assert [tuple(t.shape) for t in out]==[(1,8192),(1,512),(16,),(1,1280)]


@pytest.mark.parametrize('index,shape,dtype,match',[
    (0,(2,1280),torch.bfloat16,'C1'),
    (2,(1,640),torch.bfloat16,'C1'),
    (4,(1,),torch.int64,'I32'),
    (6,(256,512),torch.uint8,'canonical'),
    (8,(4096,1280),torch.float8_e4m3fn,'TP-local'),
])
def test_invalid_runtime_geometry_rejected(index,shape,dtype,match):
    args=operands();args[index]=torch.empty(shape,dtype=dtype,device='meta')
    with pytest.raises(RuntimeError,match=match):
        torch.ops.custom_op.custom_deepseek_v41_qkv_projection_publish_gaudi2(*args,1e-20,-1)


@pytest.mark.parametrize('offset',[1,1024])
def test_invalid_decoded_state_offset_rejected(offset):
    with pytest.raises(RuntimeError,match='aligned'):
        torch.ops.custom_op.custom_deepseek_v41_qkv_projection_publish_gaudi2(*operands(),1e-20,offset)
