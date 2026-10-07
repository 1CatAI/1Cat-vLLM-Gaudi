# SPDX-License-Identifier: Apache-2.0
"""Explicit active K is a shape argument; the packed source contract is unchanged."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def inputs(width=640):
    def t(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device='meta')
    hidden, experts = 5120, 384
    return [t((1,hidden)),t((1,6),torch.int32),t((1,6),torch.float32),
            t((experts,2*width//256,hidden*64),torch.int16),
            t((experts,hidden//256,width*64),torch.int16),
            t((experts,2*width//256,hidden*4+128),torch.int16),
            t((experts,hidden//256,width*4+128),torch.int16),t((128,)),
            t((experts,2*width//256,256)),t((experts,hidden//256,256)),
            t((1,hidden),torch.float8_e4m3fn),t((1,1),torch.float32),t((1,hidden))]


@pytest.mark.parametrize('padded,active', [(640,576),(640,640),(1152,1152)])
def test_shape_parameter_accepts_tp_shards(padded,active):
    out=torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_active_k_sat_shared_fp8_gaudi2(
        *inputs(padded),active,True)
    assert out.shape==(1,5120) and out.dtype==torch.bfloat16


@pytest.mark.parametrize('active', [0,575,512,704])
def test_invalid_active_extent_fails_before_native_launch(active):
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_active_k_sat_shared_fp8_gaudi2(
            *inputs(),active,True)
