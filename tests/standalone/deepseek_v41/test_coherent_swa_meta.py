# SPDX-License-Identifier: Apache-2.0
"""Shared SWA consumers retain TP-parametric heads and a bounded C2-C6 contract."""
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def load_addon():
    path = os.getenv('DSV41_COHERENT_SWA_LIBRARY')
    if not path:
        pytest.skip('Set the isolated coherent SWA addon library')
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(str(Path(path).resolve()))


@pytest.mark.parametrize('rows,heads', [(2, 8), (5, 16), (6, 8)])
def test_meta_preserves_query_shape(rows, heads):
    args = (torch.empty(rows, heads, 512, dtype=torch.bfloat16, device='meta'),
            torch.empty(256, 528, dtype=torch.uint8, device='meta'),
            torch.empty(rows, 640, 512, dtype=torch.bfloat16, device='meta'),
            torch.empty(rows, 640, dtype=torch.float32, device='meta'),
            torch.empty(rows, dtype=torch.int32, device='meta'),
            torch.empty(heads, dtype=torch.float32, device='meta'),
            torch.empty(1, dtype=torch.float32, device='meta'),
            torch.empty(rows, dtype=torch.int32, device='meta'),
            torch.empty(rows, 640, 512, dtype=torch.float32, device='meta'))
    result = torch.ops.custom_op.custom_deepseek_v41_coherent_swa_mla_gaudi2(*args)
    assert result.shape == args[0].shape and result.dtype == torch.bfloat16


@pytest.mark.parametrize('rows', [1, 7])
def test_rejects_non_speculative_rows(rows):
    args = (torch.empty(rows, 8, 512, dtype=torch.bfloat16, device='meta'),
            torch.empty(256, 528, dtype=torch.uint8, device='meta'),
            torch.empty(rows, 640, 512, dtype=torch.bfloat16, device='meta'),
            torch.empty(rows, 640, device='meta'), torch.empty(rows, dtype=torch.int32, device='meta'),
            torch.empty(8, device='meta'), torch.empty(1, device='meta'),
            torch.empty(rows, dtype=torch.int32, device='meta'), torch.empty(rows, 640, 512, device='meta'))
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_coherent_swa_mla_gaudi2(*args)
