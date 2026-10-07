# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
OP=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_positive_gaudi2

@pytest.mark.parametrize("batch",[1,2,6])
def test_supported_decode_buckets(batch):
    result=OP(torch.empty((batch,24),device="meta"),torch.empty((batch,1),device="meta"),
              torch.empty(3,device="meta"),torch.empty(24,device="meta"))
    assert result.shape==(batch,24) and result.dtype==torch.float32

@pytest.mark.parametrize("batch,width,dtype",[(0,24,torch.float32),(16385,24,torch.float32),
    (1,23,torch.float32),(1,24,torch.bfloat16)])
def test_bad_mixes_rejected(batch,width,dtype):
    with pytest.raises(RuntimeError):
        OP(torch.empty((batch,width),dtype=dtype,device="meta"),torch.empty((batch,1),device="meta"),
           torch.empty(3,device="meta"),torch.empty(24,device="meta"))
