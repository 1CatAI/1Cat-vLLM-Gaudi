# SPDX-License-Identifier: Apache-2.0
"""Shared token-wide schemas keep C1 and larger scheduler buckets compatible."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(batch):
    def tensor(shape, dtype): return torch.empty(shape, dtype=dtype, device='meta')
    return [tensor((batch,5120),torch.bfloat16),tensor((batch,6),torch.int32),tensor((batch,6),torch.float32),
            tensor((384,5,327680),torch.int16),tensor((384,20,40960),torch.int16),
            tensor((384,5,20608),torch.int16),tensor((384,20,2688),torch.int16),
            tensor((128,),torch.bfloat16),tensor((384,5,256),torch.bfloat16),
            tensor((384,20,256),torch.bfloat16),tensor((batch,5120),torch.float8_e4m3fn),
            tensor((batch,1),torch.float32)]


@pytest.mark.parametrize('batch',[1,2,6])
def test_sat_and_shared_shapes(batch):
    args=operands(batch)
    result=torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_fp8_gaudi2(*args,True)
    shared=torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_shared_fp8_gaudi2(
        *args,args[0],True)
    assert result.shape==shared.shape==(batch,5120)
    assert result.dtype==shared.dtype==torch.bfloat16


def test_unqualified_scales_rejected():
    with pytest.raises(RuntimeError,match='qualif'):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_fp8_gaudi2(*operands(1),False)


@pytest.mark.parametrize('schema',['paired_decode','silu_decode','two_group_w2_sat'])
@pytest.mark.parametrize('batch',[1,2,6])
def test_paired_decode_bucket_contract(batch,schema):
    args=operands(batch)
    op=getattr(torch.ops.custom_op,'custom_deepseek_v41_expert_n256_moe_'+schema+'_shared_fp8_gaudi2')
    if batch==1:
        result=op(*args,args[0],True)
        assert result.shape==(1,5120) and result.dtype==torch.bfloat16
    else:
        with pytest.raises(RuntimeError,match='single-token|one BF16 shared row'):
            op(*args,args[0],True)


@pytest.mark.parametrize('schema',['paired_decode','silu_decode','two_group_w2_sat'])
def test_paired_decode_rejects_unqualified_scales(schema):
    args=operands(1)
    with pytest.raises(RuntimeError,match='qualif'):
        getattr(torch.ops.custom_op,'custom_deepseek_v41_expert_n256_moe_'+schema+'_shared_fp8_gaudi2')(*args,args[0],False)
