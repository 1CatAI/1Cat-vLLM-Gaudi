# SPDX-License-Identifier: Apache-2.0
"""The fused producer retains both TP query widths and excludes C1/prefill."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_HW_DENSE_FUSED_LIBRARY")
    if not path:
        pytest.skip("Set the independent fixed-quant registration")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(path)


@pytest.mark.parametrize("rows,k,n", ((2, 5120, 1792), (6, 5120, 1792),
                                      (5, 1280, 8192), (6, 1280, 16384)))
def test_tp_shapes(rows, k, n):
    x = torch.empty(rows, k, dtype=torch.bfloat16, device="meta")
    w = torch.empty(n, k, dtype=torch.float8_e4m3fn, device="meta")
    result = torch.ops.custom_op.custom_deepseek_v41_hw_dense_roundtrip_fp8_gaudi2(x, w, 1., 1 / 256)
    assert result.shape == (rows, n) and result.dtype == torch.bfloat16


def test_c1_excluded():
    x = torch.empty(1, 5120, dtype=torch.bfloat16, device="meta")
    w = torch.empty(1792, 5120, dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(RuntimeError, match="Static FP8"):
        torch.ops.custom_op.custom_deepseek_v41_hw_dense_roundtrip_fp8_gaudi2(x, w, 1., 1 / 256)
