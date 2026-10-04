# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands():
    shapes=[(1,512),(1,128),(128,),(1,),(32768,64),(256,),
            (8256,288),(8256,68),(8192,512),(8192,128),(8192,128)]
    return [torch.empty(shape,device='meta',dtype=torch.int32 if i in (3,5) else
                        torch.float32 if i==4 else torch.uint8 if i in (6,7) else torch.bfloat16)
            for i,shape in enumerate(shapes)]


@pytest.mark.parametrize('ratio',[1,2])
@pytest.mark.parametrize('enabled',[False,True])
def test_publication_completion_metadata(ratio,enabled):
    done=torch.ops.custom_op.custom_deepseek_v41_fp4_norm_rope_publish_gaudi2(
        *operands(),1e-20,ratio,enabled,enabled,enabled)
    assert done.dtype==torch.int32 and done.shape==(36,)


@pytest.mark.parametrize('index',[0,1,2,3,4,5,6,7,8,9,10])
def test_dtype_ownership_contract(index):
    values=operands();values[index]=values[index].double()
    with pytest.raises(RuntimeError,match='contract'):
        torch.ops.custom_op.custom_deepseek_v41_fp4_norm_rope_publish_gaudi2(
            *values,1e-20,2,True,True,True)


def test_compressor_ratio_contract():
    with pytest.raises(RuntimeError,match='ratio'):
        torch.ops.custom_op.custom_deepseek_v41_fp4_norm_rope_publish_gaudi2(
            *operands(),1e-20,4,True,True,True)
