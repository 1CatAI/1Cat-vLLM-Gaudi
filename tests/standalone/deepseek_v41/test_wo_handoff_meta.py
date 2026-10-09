# SPDX-License-Identifier: Apache-2.0
"""Shared C1-C6 output handoff shape/ownership boundary checks."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_WO_HANDOFF_LIBRARY")
    if not path:
        pytest.skip("Set the independently built handoff registration library")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize("rows", (1, 2, 6))
@pytest.mark.parametrize("groups", (2, 4))
def test_batched_projection_and_scale(rows, groups):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    f8 = torch.float8_e4m3fn
    product = tensor((groups, rows, 1024), torch.float32)
    weight_scale = tensor((groups, 1, 1024), torch.float32)
    activation_scale = tensor((groups, rows, 1), torch.float32)
    quantized, scale = torch.ops.custom_op.custom_deepseek_v41_woa_scale_dense_quant_gaudi2(
        product, weight_scale, activation_scale)
    assert quantized.shape == (rows, groups * 1024) and quantized.dtype == f8
    assert scale.shape == (rows, 1) and scale.dtype == torch.float32
    output = torch.ops.custom_op.custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2(
        tensor((rows, groups * 8, 512), torch.bfloat16), tensor((groups, 4096, 1024), f8), weight_scale,
        tensor((5120, groups * 1024), f8), tensor((1, 5120), torch.float32),
        tensor((rows,), torch.int32), tensor((16384, 64), torch.float32))
    assert output.shape == (rows, 5120) and output.dtype == torch.bfloat16


def test_reject_position_row_mismatch():
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    f8 = torch.float8_e4m3fn
    with pytest.raises(RuntimeError, match="position"):
        torch.ops.custom_op.custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2(
            tensor((6, 16, 512), torch.bfloat16), tensor((2, 4096, 1024), f8),
            tensor((2, 1, 1024), torch.float32), tensor((5120, 2048), f8),
            tensor((1, 5120), torch.float32), tensor((1,), torch.int32), tensor((16384, 64), torch.float32))
