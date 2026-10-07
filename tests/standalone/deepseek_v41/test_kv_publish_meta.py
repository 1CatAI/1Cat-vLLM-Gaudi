# SPDX-License-Identifier: Apache-2.0
"""KV publication state ownership and optional-mirror metadata contracts."""
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands():
    def t(shape,dtype):return torch.empty(shape,device='meta',dtype=dtype)
    return [t((1,512),torch.bfloat16),t((512,),torch.bfloat16),t((1,),torch.int32),
            t((524288,64),torch.float32),t((256,528),torch.uint8),t((20480,512),torch.bfloat16)]


@pytest.mark.parametrize('offset',[-1,0,512])
def test_live_cache_and_mirror_metadata(offset):
    result,completion=torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_publish_gaudi2(
        *operands(),1e-20,offset)
    assert result.shape==(1,512) and result.dtype==torch.bfloat16
    assert completion.shape==(16,) and completion.dtype==torch.int32


def test_mirror_offset_alignment():
    with pytest.raises(RuntimeError,match='scheduler-owned'):
        torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_publish_gaudi2(*operands(),1e-20,1)
