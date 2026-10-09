# SPDX-License-Identifier: Apache-2.0
"""Bounded decoded-row cache preserves logical MLA tensor geometry."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_ROW_CACHE_LIBRARY")
    if not path:
        pytest.skip("Set the independently built row-cache registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows=6):
    def t(shape, dtype=torch.float32):
        return torch.empty(shape, device="meta", dtype=dtype)
    return (t((rows, 16, 512), torch.bfloat16), t((256, 528), torch.uint8), t((32768, 288), torch.uint8),
            t((rows, 512), torch.int32), t((rows,), torch.int32), t((256,), torch.int32),
            t((16,)), t((1,)), t((rows,), torch.int32), 1, True)


@pytest.mark.parametrize("rows", (2, 5, 6))
def test_original_per_query_kv_contract(rows):
    result = torch.ops.custom_op.custom_deepseek_v41_logical_row_cache_gaudi2(*operands(rows))
    assert result.shape == (rows, 16, 512) and result.dtype == torch.bfloat16


@pytest.mark.parametrize("kind", ("rows", "selection", "main"))
def test_reject_incompatible_cache_inputs(kind):
    args = list(operands(7 if kind == "rows" else 6))
    if kind == "selection":
        args[3] = torch.empty((6, 256), device="meta", dtype=torch.int32)
    if kind == "main":
        args[2] = torch.empty((32768, 288), device="meta", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_logical_row_cache_gaudi2(*args)
