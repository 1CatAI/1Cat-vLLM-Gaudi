# SPDX-License-Identifier: Apache-2.0
"""Validate scale-plane ownership without allocating or copying expert weights."""
import os

import pytest
import torch

from test_silu_decode_affine_meta import inputs


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_SPLIT_SCALE_PLANES_LIBRARY")
    if not path:
        pytest.skip("Set the additive split-scale registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows=6, intermediate=640, compact=True):
    args = inputs(rows, intermediate, compact)
    qualified = args.pop()
    for source in args[5:7]:
        group = source[..., :-128] if compact else source
        channel = (source[..., -128:] if compact else
                   torch.empty((*source.shape[:2], 128), dtype=source.dtype, device="meta"))
        args.extend((group, channel))
    return [*args, qualified]


@pytest.mark.parametrize("rows,intermediate,compact", ((2, 640, True), (6, 640, True), (6, 1152, False)))
def test_checkpoint_layout(rows, intermediate, compact):
    result = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_split_scale_planes_fp8_gaudi2(
        *operands(rows, intermediate, compact))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


def test_reject_group_copy():
    args = operands()
    args[12] = args[12].clone()
    with pytest.raises(RuntimeError, match="original storage/strides"):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_split_scale_planes_fp8_gaudi2(*args)


def test_reject_channel_offset():
    args = operands()
    args[13] = args[5][..., :128]
    with pytest.raises(RuntimeError, match="storage-sharing tail"):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_split_scale_planes_fp8_gaudi2(*args)


def test_reject_draft_routing():
    args = operands()
    args[1] = torch.empty((6, 3), dtype=torch.int32, device="meta")
    args[2] = torch.empty((6, 3), dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="top6"):
        torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_split_scale_planes_fp8_gaudi2(*args)
