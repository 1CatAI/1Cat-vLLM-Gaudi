# SPDX-License-Identifier: Apache-2.0
"""Shared draft FP8 compound validates checkpoint expert geometry."""
import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def register():
    path = os.environ.get('DSV41_MTP_SAT_LIBRARY')
    if not path:
        pytest.skip('Set the separately built MTP SAT registration')
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows, intermediate, *, experts=128, routes=3, qualified=True):
    def x(shape, dtype):
        return torch.empty(shape, dtype=dtype, device='meta')

    return (x((rows, 5120), torch.bfloat16), x((rows, routes), torch.int32),
            x((rows, routes), torch.float32),
            x((experts, intermediate * 2 // 256, 5120 * 64), torch.int16),
            x((experts, 20, intermediate * 64), torch.int16),
            x((experts, intermediate * 2 // 256, 5120 * 4 + 128), torch.int16),
            x((experts, 20, intermediate * 4 + 128), torch.int16),
            x((128,), torch.bfloat16),
            x((experts, intermediate * 2 // 256, 256), torch.bfloat16),
            x((experts, 20, 256), torch.bfloat16), qualified)


@pytest.mark.parametrize('rows,intermediate', ((1, 640), (5, 640), (6, 1280)))
def test_native_draft_shape(rows, intermediate):
    result = torch.ops.custom_op.custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2(*operands(rows, intermediate))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


@pytest.mark.parametrize('case', ('target', 'qualification'))
def test_reject_target_router_or_unqualified_exponents(case):
    with pytest.raises(RuntimeError, match='Draft SAT'):
        torch.ops.custom_op.custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2(
            *operands(5, 640, experts=384 if case == 'target' else 128,
                      routes=6 if case == 'target' else 3, qualified=case != 'qualification'))
