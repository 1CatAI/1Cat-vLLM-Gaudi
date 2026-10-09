# SPDX-License-Identifier: Apache-2.0
"""Production shape and unchanged RoPE boundary contract."""
import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def register():
    path = os.environ.get('DSV41_ROPE_COHERENT_LIBRARY')
    if not path:
        pytest.skip('Set the independent coherent access-map registration')
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize('rows,heads,inverse', ((2, 8, False), (6, 16, False), (5, 16, True), (6, 8, True)))
def test_native_shape(rows, heads, inverse):
    x = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device='meta')
    positions = torch.empty(rows, dtype=torch.int32, device='meta')
    table = torch.empty(32768, 64, dtype=torch.float32, device='meta')
    op = (torch.ops.custom_op.custom_deepseek_v41_rope_inverse_coherent_bf16_gaudi2 if inverse else
          torch.ops.custom_op.custom_deepseek_v41_rope_coherent_bf16_gaudi2)
    result = op(x, positions, table)
    assert result.shape == x.shape and result.dtype == x.dtype
