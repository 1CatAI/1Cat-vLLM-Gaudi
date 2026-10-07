# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


@pytest.mark.parametrize('heads', [8, 16, 32])
@pytest.mark.parametrize('codec', ['', 'vector_', 'vector_mask_', 'native_codec_'])
def test_optional_register_softmax_keeps_publisher_and_reader_abi(heads, codec):
    t = lambda shape, dtype: torch.empty(shape, dtype=dtype, device='meta')
    q = t((1, heads, 512), torch.bfloat16)
    swa = t((256, 528), torch.uint8)
    main = t((262144, 288), torch.uint8)
    selected = t((1, 512), torch.int32)
    pos = t((1, ), torch.int32)
    pages = t((8192, ), torch.int32)
    sink = t((heads, ), torch.float32)
    scale = t((1, ), torch.float32)
    lens = t((1, ), torch.int32)
    publish = getattr(torch.ops.custom_op, f'custom_deepseek_v41_main_publish_{codec}mla_gaudi2')
    reuse = getattr(torch.ops.custom_op, f'custom_deepseek_v41_main_reuse_{codec}mla_gaudi2')
    reference = publish(q, swa, main, selected, pos, pages, sink, scale, lens, 2)
    candidate = publish(q, swa, main, selected, pos, pages, sink, scale, lens, 2, True)
    assert [(v.shape, v.dtype) for v in candidate] == [(v.shape, v.dtype) for v in reference]
    out = reuse(q, swa, candidate[1], candidate[2], pos, sink, scale, lens, True)
    assert out.shape == q.shape and out.dtype == q.dtype
