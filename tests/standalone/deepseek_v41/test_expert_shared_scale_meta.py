# SPDX-License-Identifier: Apache-2.0
"""Shared-scale finalization retains active-K checkpoint and rounding contracts."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401
from test_expert_active_k_meta import inputs

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(padded=640, active=576):
    return [*inputs(padded),torch.empty(1,1,device='meta'),torch.empty(1,5120,device='meta'),active,True]


@pytest.mark.parametrize('padded,active',[(640,576),(640,640),(1152,1152)])
def test_shared_scale_accepts_same_tp_shards(padded,active):
    out=torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_shared_scale_sat_fp8_gaudi2(
        *operands(padded,active))
    assert out.shape==(1,5120) and out.dtype==torch.bfloat16


@pytest.mark.parametrize('slot,shape',[(12,(2,5120)),(13,(1,2)),(14,(5120,1))])
def test_unqualified_shared_shapes_rejected(slot,shape):
    values=operands();values[slot]=torch.empty(shape,dtype=values[slot].dtype,device='meta')
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_shared_scale_sat_fp8_gaudi2(*values)


def test_checkpoint_qualification_cannot_be_bypassed():
    values=operands();values[-1]=False
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_shared_scale_sat_fp8_gaudi2(*values)
