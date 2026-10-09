# SPDX-License-Identifier: Apache-2.0
"""Compound outputs are separate BF16 buffers with the checkpoint widths."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_NORM_ROUNDTRIP_LIBRARY")
    if not path:
        pytest.skip("Set the private norm/roundtrip registration library")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize("width", (1280, 5120))
def test_c6_outputs_have_independent_storage(width):
    value = torch.empty((6, width), device="meta", dtype=torch.bfloat16)
    gamma = torch.empty(width, device="meta", dtype=torch.bfloat16)
    normalized, rounded = torch.ops.custom_op.custom_deepseek_v41_norm_roundtrip_bf16_gaudi2(value, gamma, 1e-20)
    assert normalized.shape == rounded.shape == value.shape
    assert normalized.dtype == rounded.dtype == torch.bfloat16
    assert normalized is not rounded and normalized is not value


def test_reject_wrong_checkpoint_weight_contract():
    value = torch.empty((6, 5120), device="meta", dtype=torch.bfloat16)
    gamma = torch.empty(1280, device="meta", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="checkpoint rows"):
        torch.ops.custom_op.custom_deepseek_v41_norm_roundtrip_bf16_gaudi2(value, gamma, 1e-20)
