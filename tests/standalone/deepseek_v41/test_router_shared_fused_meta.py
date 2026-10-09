# SPDX-License-Identifier: Apache-2.0
"""Joint producer strides and output contracts across both TP widths."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_ROUTER_SHARED_FUSED_LIBRARY")
    if not path:
        pytest.skip("Set the independent fused Router/shared registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize("rows,columns", ((2, 1792), (6, 1792), (5, 3072), (6, 3072)))
def test_full_product_contract(rows, columns):
    p = torch.empty(rows, columns, dtype=torch.float32, device="meta")
    bias = torch.empty(384, dtype=torch.float32, device="meta")
    mask = torch.empty(rows, dtype=torch.bool, device="meta")
    channel = torch.empty(1, 384, dtype=torch.float32, device="meta")
    sx = torch.empty(rows, 1, dtype=torch.float32, device="meta")
    ids, weights = torch.ops.custom_op.custom_deepseek_v41_router_shared_scaled_gaudi2(
        p, bias, bias, mask, channel, sx)
    assert ids.shape == weights.shape == (rows, 6)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    selected = torch.empty(1, rows, dtype=torch.int32, device="meta")
    routing = torch.empty(1, rows, dtype=torch.float32, device="meta")
    shared_channel = torch.empty(1, (columns - 512) // 256, 256, dtype=torch.bfloat16, device="meta")
    value, scale = torch.ops.custom_op.custom_deepseek_v41_shared_silu_full_product_gaudi2(
        p.reshape(rows, 1, columns), selected, sx, shared_channel, routing)
    assert value.shape == (rows, 1, (columns - 512) // 2) and value.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, 1, 1) and scale.dtype == torch.float32
