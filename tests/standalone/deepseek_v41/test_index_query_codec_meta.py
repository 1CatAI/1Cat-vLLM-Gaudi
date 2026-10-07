# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
OP = torch.ops.custom_op.custom_deepseek_v41_index_query_rope_fp4_bf16_gaudi2

@pytest.mark.parametrize('batch,heads', [(1,8),(1,16),(1,32),(2,8),(6,16),(64,128)])
def test_decode_contract(batch,heads):
    x=torch.empty((batch,heads,128),dtype=torch.bfloat16,device='meta')
    result=OP(x,torch.empty(batch,dtype=torch.int32,device='meta'),torch.empty((524288,64),device='meta'))
    assert result.shape==x.shape and result.dtype==x.dtype

@pytest.mark.parametrize('batch,heads,width,dtype', [(0,8,128,torch.bfloat16),(65,8,128,torch.bfloat16),
    (1,129,128,torch.bfloat16),(1,8,256,torch.bfloat16),(1,8,128,torch.float32)])
def test_incompatible_contract_is_rejected(batch,heads,width,dtype):
    with pytest.raises(RuntimeError):
        OP(torch.empty((batch,heads,width),dtype=dtype,device='meta'),
           torch.empty(batch,dtype=torch.int32,device='meta'),torch.empty((512,64),device='meta'))

def test_interleaved_phase_contract_is_rejected():
    with pytest.raises(RuntimeError):
        OP(torch.empty((1,8,128),dtype=torch.bfloat16,device='meta'),
           torch.empty(1,dtype=torch.int32,device='meta'),torch.empty((512,32,2),device='meta'))
