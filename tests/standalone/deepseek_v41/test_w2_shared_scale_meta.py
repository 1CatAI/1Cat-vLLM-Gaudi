# SPDX-License-Identifier: Apache-2.0
"""Additive bounded producer and deferred shared scale ownership contracts."""
import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def registration():
    path = os.environ.get('DSV41_SHARED_SCALE_META_LIBRARY')
    if not path:
        pytest.skip('Requires the additive shared-scale extension')
    torch.ops.load_library(path)


def operands(rows):
    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device='meta')

    p13 = empty((384, 5, 20608), torch.int16)
    p2 = empty((384, 20, 2688), torch.int16)
    return [empty((rows, 5120), torch.bfloat16), empty((rows, 6), torch.int32),
            empty((rows, 6), torch.float32), empty((384, 5, 327680), torch.int16),
            empty((384, 20, 40960), torch.int16), p13, p2, empty((128,), torch.bfloat16),
            empty((384, 5, 256), torch.bfloat16), empty((384, 20, 256), torch.bfloat16),
            empty((rows, 5120), torch.float8_e4m3fn), empty((rows, 1), torch.float32),
            p13[..., :20480], p13[..., 20480:], p2[..., :2560], p2[..., 2560:],
            empty((rows, 5120), torch.bfloat16), empty((rows, 1), torch.float32),
            empty((1, 5120), torch.float32), True]


@pytest.mark.parametrize('rows', [2, 5, 6])
def test_bounded_rows(rows):
    result = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_w2_shared_scale_n256_fp8_gaudi2(
        *operands(rows))
    assert result.shape == (rows, 5120)
    assert result.dtype == torch.bfloat16


def test_c1_excluded():
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_w2_shared_scale_n256_fp8_gaudi2(*operands(1))


def test_raw_product_keeps_bf16_boundary():
    values = operands(6)
    values[16] = values[16].float()
    with pytest.raises(RuntimeError, match='Shared scale chain'):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_w2_shared_scale_n256_fp8_gaudi2(*values)


def test_row_scale_owner():
    values = operands(6)
    values[17] = values[17][:1]
    with pytest.raises(RuntimeError, match='Shared scale chain'):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_w2_shared_scale_n256_fp8_gaudi2(*values)
