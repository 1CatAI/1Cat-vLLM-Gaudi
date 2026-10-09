# SPDX-License-Identifier: Apache-2.0
"""Model-width norm preserves C1-C6 geometry and rejects incompatible inputs."""
import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def register():
    path = os.environ.get('DSV41_INPUT_NORM_LIBRARY')
    if not path:
        pytest.skip('Set the separately built input norm registration')
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize('rows', (1, 2, 6))
def test_checkpoint_rows(rows):
    value = torch.empty((rows, 5120), dtype=torch.bfloat16, device='meta')
    weight = torch.empty((5120,), dtype=torch.bfloat16, device='meta')
    result = torch.ops.custom_op.custom_deepseek_v41_input_norm_bf16_gaudi2(value, weight, 1e-20)
    assert result.shape == value.shape and result.dtype == value.dtype


@pytest.mark.parametrize('kind', ('width', 'epsilon'))
def test_reject_incompatible_model_width_and_epsilon(kind):
    width = 1280 if kind == 'width' else 5120
    value = torch.empty((6, width), dtype=torch.bfloat16, device='meta')
    weight = torch.empty((width,), dtype=torch.bfloat16, device='meta')
    with pytest.raises(RuntimeError, match='Decode input norm'):
        torch.ops.custom_op.custom_deepseek_v41_input_norm_bf16_gaudi2(
            value, weight, 0 if kind == 'epsilon' else 1e-20)
