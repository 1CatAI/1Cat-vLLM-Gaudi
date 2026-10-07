# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(heads=16):
    shapes=[(1,heads,512),(1,512),(512,),(256,528),(1,640,512),(1,640),
            (1,),(524288,64),(heads,),(1,),(1,)]
    return [torch.empty(shape,device='meta',dtype=torch.uint8 if i==3 else
                        torch.int32 if i in (6,10) else torch.float32 if i in (5,7,8,9) else torch.bfloat16)
            for i,shape in enumerate(shapes)]


@pytest.mark.parametrize('heads',[8,16,32,64])
def test_tp_parameterized_output(heads):
    out=torch.ops.custom_op.custom_deepseek_v41_kv_norm_reuse_mla_gaudi2(*operands(heads),1e-20)
    assert out.dtype==torch.bfloat16 and out.shape==(1,heads,512)


@pytest.mark.parametrize('index',list(range(11)))
def test_inference_operand_contract(index):
    values=operands();values[index]=values[index].double()
    with pytest.raises(RuntimeError,match='inference'):
        torch.ops.custom_op.custom_deepseek_v41_kv_norm_reuse_mla_gaudi2(*values,1e-20)
