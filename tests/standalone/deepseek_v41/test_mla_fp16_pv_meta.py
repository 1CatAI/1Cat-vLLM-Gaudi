# SPDX-License-Identifier: Apache-2.0
"""Additive FP16 PV schema contracts, with no device allocation."""
import os

import pytest
import torch


@pytest.fixture(scope='module')
def operators():
    library = os.getenv('DSV41_FP16_PV_OPERATOR_LIBRARY')
    if not library:
        pytest.skip('Requires the separately built experimental FP16 PV registration')
    torch.ops.load_library(library)
    return torch.ops.custom_op


def operands(rows, heads=16):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device='meta')

    return (tensor((rows, heads, 512), torch.bfloat16), tensor((256, 528), torch.uint8),
            tensor((32768, 288), torch.uint8), tensor((rows, 512), torch.int32),
            tensor((rows,), torch.int32), tensor((256,), torch.int32), tensor((heads,), torch.float32),
            tensor((1,), torch.float32), tensor((rows,), torch.int32), 1)


@pytest.mark.parametrize('rows', [2, 6])
@pytest.mark.parametrize('heads', [8, 16])
def test_direct_pv_public_planes(operators, rows, heads):
    args = operands(rows, heads)
    output, bank, mask, values = operators.custom_deepseek_v41_main_fp16_direct_publish_mla_gaudi2(*args)
    assert output.shape == args[0].shape and output.dtype == torch.bfloat16
    assert bank.shape == (rows, 640, 512) and bank.dtype == torch.bfloat16
    assert mask.shape == (rows, 640) and mask.dtype == torch.float32
    assert values.shape == bank.shape and values.dtype == torch.float16
    result = operators.custom_deepseek_v41_main_fp16_direct_reuse_mla_gaudi2(
        args[0], args[1], bank, mask, args[4], args[6], args[7], args[8], values)
    assert result.shape == output.shape and result.dtype == output.dtype
    with pytest.raises(RuntimeError, match='matching contiguous'):
        operators.custom_deepseek_v41_main_fp16_direct_reuse_mla_gaudi2(
            args[0], args[1], bank, mask, args[4], args[6], args[7], args[8], values.float())


def test_direct_c1_excluded(operators):
    with pytest.raises(RuntimeError, match='confined to C2-C6'):
        operators.custom_deepseek_v41_main_fp16_direct_publish_mla_gaudi2(*operands(1))
