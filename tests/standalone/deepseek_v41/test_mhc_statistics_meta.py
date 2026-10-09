# SPDX-License-Identifier: Apache-2.0
"""Feature-parallel statistics preserve the production mHC operand contract."""
import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def register():
    path = os.environ.get('DSV41_MHC_STATISTICS_LIBRARY')
    if not path:
        pytest.skip('Set the independently built statistics registration')
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize('rows,width', ((2, 24), (5, 48), (6, 48)))
def test_mme_projection_contract(rows, width):
    p = torch.empty((rows, width), dtype=torch.float32, device='meta')
    x = torch.empty((rows, 20480), dtype=torch.bfloat16, device='meta')
    scale = torch.empty((3,), dtype=torch.float32, device='meta')
    base = torch.empty((24,), dtype=torch.float32, device='meta')
    result = torch.ops.custom_op.custom_deepseek_v41_mhc_statistics_epilogue_gaudi2(p, x, scale, base)
    assert result.shape == (rows, 24) and result.dtype == torch.float32


@pytest.mark.parametrize('kind', ('rows', 'residual'))
def test_reject_wrong_checkpoint_geometry(kind):
    rows = 1 if kind == 'rows' else 6
    width = 5120 if kind == 'residual' else 20480
    p = torch.empty((rows, 48), dtype=torch.float32, device='meta')
    x = torch.empty((rows, width), dtype=torch.bfloat16, device='meta')
    scale = torch.empty((3,), dtype=torch.float32, device='meta')
    base = torch.empty((24,), dtype=torch.float32, device='meta')
    with pytest.raises(RuntimeError, match='mHC MME epilogue'):
        torch.ops.custom_op.custom_deepseek_v41_mhc_statistics_epilogue_gaudi2(p, x, scale, base)
