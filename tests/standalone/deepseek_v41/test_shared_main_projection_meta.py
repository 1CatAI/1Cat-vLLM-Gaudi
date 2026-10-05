# SPDX-License-Identifier: Apache-2.0
"""Shared-main projection keeps publish outputs and TP-dependent WO dimensions."""
import os

import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(heads):
    def make(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device='meta')
    q = make((1, heads, 512), torch.bfloat16)
    swa = make((256, 528), torch.uint8)
    main = make((1, 640, 512), torch.bfloat16)
    mask = make((1, 640))
    pos = make((1,), torch.int32)
    sink, scale, lengths = make((heads,)), make((1,)), make((1,), torch.int32)
    groups = heads // 8
    tail = (make((groups, 4096, 1024), torch.float8_e4m3fn), make((groups, 1, 1024)),
            make((524288, 64)), make((5120, groups * 1024), torch.float8_e4m3fn), make((1, 5120)))
    return (q, swa, main, mask, pos, sink, scale, lengths), tail


@pytest.mark.parametrize('heads', (16, 32))
def test_reuse_projection(heads):
    args, tail = operands(heads)
    out = torch.ops.custom_op.custom_deepseek_v41_main_reuse_projection_gaudi2(*args, *tail)
    assert out.shape == (1, 5120) and out.dtype == torch.bfloat16


@pytest.mark.parametrize('heads,ratio', ((16, 1), (16, 2), (32, 1), (32, 2)))
def test_publish_projection(heads, ratio):
    args, tail = operands(heads)
    q, swa, _, _, pos, sink, scale, lengths = args
    main = torch.empty((524288, 288), dtype=torch.uint8, device='meta')
    selected = torch.empty((1, 512), dtype=torch.int32, device='meta')
    pages = torch.empty((4096,), dtype=torch.int32, device='meta')
    out = torch.ops.custom_op.custom_deepseek_v41_main_publish_projection_gaudi2(
        q, swa, main, selected, pos, pages, sink, scale, lengths, ratio, *tail)
    assert [v.shape for v in out] == [(1, 5120), (1, 640, 512), (1, 640)]
    assert [v.dtype for v in out] == [torch.bfloat16, torch.bfloat16, torch.float32]


def test_wrong_channel_scale_shape_rejected():
    args, tail = operands(16)
    with pytest.raises(RuntimeError, match='prepared FP8 WO weights'):
        torch.ops.custom_op.custom_deepseek_v41_main_reuse_projection_gaudi2(*args, *tail[:-1], tail[-1].flatten())
