# SPDX-License-Identifier: Apache-2.0
"""Checkpoint top3 N128 draft expert shape qualification."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_MTP_FP8_LIBRARY")
    if not path:
        pytest.skip("Set the independently built MTP FP8 registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows=5, experts=128, topk=3):
    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")
    return (tensor((rows, 5120)), tensor((rows, topk), torch.int32), tensor((rows, topk), torch.float32),
            tensor((experts, 10, 163840), torch.int16), tensor((experts, 40, 20480), torch.int16),
            tensor((experts, 10, 20480)), tensor((experts, 40, 2560)), tensor((128,)),
            tensor((experts, 10, 128)), tensor((experts, 40, 128)), True)


@pytest.mark.parametrize("rows", (1, 5, 6))
def test_real_n128_draft_geometry(rows):
    result = torch.ops.custom_op.custom_deepseek_v41_mtp_moe_fp8_gaudi2(*operands(rows))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


@pytest.mark.parametrize("kind", ("rows", "experts", "routing", "qualification"))
def test_reject_target_or_unqualified_operands(kind):
    args = list(operands(7 if kind == "rows" else 5,
                         384 if kind == "experts" else 128,
                         6 if kind == "routing" else 3))
    if kind == "qualification":
        args[-1] = False
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_mtp_moe_fp8_gaudi2(*args)
