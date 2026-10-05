# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(tokens=1,ranks=None,width=48):
    shapes=[(tokens,5120) if ranks is None else (ranks,tokens,5120),
            (tokens,4,5120),(tokens,width),(3,),(24,)]
    return [torch.empty(s,device='meta',dtype=torch.bfloat16 if i<2 else torch.float32)
            for i,s in enumerate(shapes)]


@pytest.mark.parametrize('width',[25,48])
@pytest.mark.parametrize('tokens',[1,2,6])
@pytest.mark.parametrize('ranks',[None,2,4,8])
def test_common_decode_geometry(tokens,ranks,width):
    residual,collapsed,gates=torch.ops.custom_op.custom_deepseek_v41_mhc_mme_post_collapse_gaudi2(
        *operands(tokens,ranks,width),1e-20)
    assert residual.shape==(tokens,4,5120) and residual.dtype==torch.bfloat16
    assert collapsed.shape==(tokens,5120) and collapsed.dtype==torch.bfloat16
    assert gates.shape==(tokens,24) and gates.dtype==torch.float32


@pytest.mark.parametrize('index',list(range(5)))
def test_inference_contract(index):
    values=operands();values[index]=values[index].double()
    with pytest.raises(RuntimeError,match='inference'):
        torch.ops.custom_op.custom_deepseek_v41_mhc_mme_post_collapse_gaudi2(*values,1e-20)
