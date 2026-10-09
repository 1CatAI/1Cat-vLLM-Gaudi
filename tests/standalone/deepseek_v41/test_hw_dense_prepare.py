# SPDX-License-Identifier: Apache-2.0
"""Bounded preparation and hardware scalar ABI; not a numerical acceptance gate."""
import os

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_hw_dense import encode_weight, scale_covering


def test_scalar_bound_and_decoded_weight():
    assert scale_covering(0.) == 1 / 256
    assert scale_covering(15.) == 1 / 16
    assert scale_covering(25.2) == 1.
    with pytest.raises(ValueError):
        scale_covering(float("nan"))
    with pytest.raises(ValueError):
        encode_weight(torch.zeros(4, 1280, dtype=torch.uint8))
    original = torch.tensor([[.125, .25, -.5, 0.]], dtype=torch.bfloat16)
    encoded, scale = encode_weight(original)
    assert scale == 1 / 256
    assert torch.equal((encoded.float() * scale).bfloat16(), original)


@pytest.mark.parametrize("rows,k,n", ((2, 5120, 1792), (6, 1280, 8192), (5, 1280, 16384)))
def test_meta_abi(rows, k, n):
    path = os.environ.get("DSV41_HW_DENSE_LIBRARY")
    if not path:
        pytest.skip("Set the isolated static-scale registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)
    x = torch.empty(rows, k, dtype=torch.bfloat16, device="meta")
    w = torch.empty(n, k, dtype=torch.float8_e4m3fn, device="meta")
    op = torch.ops.custom_op.custom_deepseek_v41_hw_dense_fp8_gaudi2
    result = op(x, w, 1., 1 / 256)
    assert result.shape == (rows, n) and result.dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="hardware-aligned"):
        op(x, w, .5, 1 / 256)
