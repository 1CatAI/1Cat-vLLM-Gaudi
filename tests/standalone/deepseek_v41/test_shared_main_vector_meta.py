# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


@pytest.mark.parametrize('heads', [16, 32])
def test_vector_body_retains_shared_row_and_mask_contract(heads):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device='meta')
    q = tensor((1, heads, 512), torch.bfloat16)
    swa = tensor((256, 528), torch.uint8)
    main = tensor((8192, 288), torch.uint8)
    selection = tensor((1, 512), torch.int32)
    position = tensor((1,), torch.int32)
    pages = tensor((8192,), torch.int32)
    sink = tensor((heads,), torch.float32)
    scale = tensor((1,), torch.float32)
    lengths = tensor((1,), torch.int32)
    out, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_publish_vector_mla_gaudi2(
        q, swa, main, selection, position, pages, sink, scale, lengths, 1)
    reused = torch.ops.custom_op.custom_deepseek_v41_main_reuse_vector_mla_gaudi2(
        q, swa, rows, mask, position, sink, scale, lengths)
    assert out.shape == reused.shape == q.shape
    assert rows.shape == (1, 640, 512) and rows.dtype == torch.bfloat16
    assert mask.shape == (1, 640) and mask.dtype == torch.float32


def test_linear_control_has_unchanged_output_contract():
    for batch in (1, 2, 6):
        x = torch.empty((batch, 20480), dtype=torch.bfloat16, device='meta')
        weight = torch.empty((24, 20480), dtype=torch.float32, device='meta')
        out = torch.ops.custom_op.custom_deepseek_v41_control_rrms_unpack_bf16_gaudi2(x, weight, 1e-20)
        assert out.shape == (batch, 25) and out.dtype == torch.float32
