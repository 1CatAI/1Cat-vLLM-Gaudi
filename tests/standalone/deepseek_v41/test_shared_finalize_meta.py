# SPDX-License-Identifier: Apache-2.0
"""Ordered routed/shared finalize shape qualification."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_SHARED_FINALIZE_LIBRARY")
    if not path:
        pytest.skip("Set the independently built shared finalize registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows=6):
    def t(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device="meta", dtype=dtype)
    return (t((rows, 5120)), t((rows, 6), torch.int32), t((rows, 6), torch.float32),
            t((384, 5, 327680), torch.int16), t((384, 20, 40960), torch.int16),
            t((384, 5, 20608), torch.int16), t((384, 20, 2688), torch.int16), t((128,)),
            t((384, 5, 256)), t((384, 20, 256)), t((rows, 5120), torch.float8_e4m3fn),
            t((rows, 1), torch.float32), t((rows, 5120)), True)


@pytest.mark.parametrize("rows", (1, 2, 5, 6))
def test_shared_row_tracks_each_token(rows):
    result = torch.ops.custom_op.custom_deepseek_v41_moe_shared_finalize_gaudi2(*operands(rows))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


@pytest.mark.parametrize("kind", ("rows", "shared", "qualification"))
def test_reject_mismatched_shared_state(kind):
    args = list(operands(7 if kind == "rows" else 6))
    if kind == "shared":
        args[12] = torch.empty((1, 5120), device="meta", dtype=torch.bfloat16)
    if kind == "qualification":
        args[-1] = False
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_moe_shared_finalize_gaudi2(*args)
