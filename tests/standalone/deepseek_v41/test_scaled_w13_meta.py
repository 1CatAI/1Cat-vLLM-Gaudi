# SPDX-License-Identifier: Apache-2.0
"""Scaled W13 retains the prepared layout and rejects MTP/C1."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_SCALED_W13_LIBRARY")
    if not path:
        pytest.skip("Set the additive SiLU/decode registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def inputs(rows, intermediate, compact):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def scales(k):
        return k * 4 + 128 if compact else k * 8

    return [tensor((rows, 5120), torch.bfloat16), tensor((rows, 6), torch.int32),
            tensor((rows, 6), torch.float32), tensor((384, intermediate * 2 // 256, 5120 * 64), torch.int16),
            tensor((384, 20, intermediate * 64), torch.int16),
            tensor((384, intermediate * 2 // 256, scales(5120)), torch.int16),
            tensor((384, 20, scales(intermediate)), torch.int16), tensor((128,), torch.bfloat16),
            tensor((384, intermediate * 2 // 256, 256), torch.bfloat16),
            tensor((384, 20, 256), torch.bfloat16), tensor((rows, 5120), torch.float8_e4m3fn),
            tensor((rows, 1), torch.float32), True]


@pytest.mark.parametrize("rows,intermediate,compact", ((2, 640, True), (6, 640, True), (6, 1152, False)))
def test_checkpoint_layout(rows, intermediate, compact):
    result = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_scaled_w13_fp8_gaudi2(
        *inputs(rows, intermediate, compact))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


def test_reject_draft_routing():
    args = inputs(5, 640, True)
    args[1] = torch.empty((5, 3), dtype=torch.int32, device="meta")
    args[2] = torch.empty((5, 3), dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="top6"):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_scaled_w13_fp8_gaudi2(*args)


def test_c1_uses_existing_dispatch():
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_scaled_w13_fp8_gaudi2(
            *inputs(1, 640, True))
