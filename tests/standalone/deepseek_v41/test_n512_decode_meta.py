# SPDX-License-Identifier: Apache-2.0
"""Coarser decode workpoints retain the production routed-expert matrix shapes."""
import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def register():
    path = os.environ.get('DSV41_N512_DECODE_LIBRARY')
    if not path:
        pytest.skip('Set the independent decode registration')
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def inputs(rows, intermediate, compact):
    def x(shape, dtype):
        return torch.empty(shape, dtype=dtype, device='meta')

    def scales(k):
        return k * 4 + 128 if compact else k * 8

    return (x((rows, 5120), torch.bfloat16), x((rows, 6), torch.int32), x((rows, 6), torch.float32),
            x((384, intermediate * 2 // 256, 5120 * 64), torch.int16),
            x((384, 20, intermediate * 64), torch.int16),
            x((384, intermediate * 2 // 256, scales(5120)), torch.int16),
            x((384, 20, scales(intermediate)), torch.int16), x((128,), torch.bfloat16),
            x((384, intermediate * 2 // 256, 256), torch.bfloat16), x((384, 20, 256), torch.bfloat16),
            x((rows, 5120), torch.float8_e4m3fn), x((rows, 1), torch.float32), True)


@pytest.mark.parametrize('rows,intermediate,compact', ((2, 1152, False), (6, 1152, False), (6, 1280, True)))
def test_checkpoint_shape(rows, intermediate, compact):
    result = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_n512_decode_fp8_gaudi2(
        *inputs(rows, intermediate, compact))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


def test_c1_default_is_outside_experiment():
    with pytest.raises(RuntimeError):
        fn = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_n512_decode_fp8_gaudi2
        fn(*inputs(1, 1280, True))
