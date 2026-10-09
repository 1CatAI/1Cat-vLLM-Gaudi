# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.getenv("DSV41_CONTROL_FP8_PAIR_LIBRARY")
    if not path:
        pytest.skip("Set the paired control registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows):
    return (torch.empty(rows, 20480, dtype=torch.bfloat16, device="meta"),
            torch.empty(48, 20480, dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty(1, 48, dtype=torch.float32, device="meta"), 1e-20)


@pytest.mark.parametrize("rows", [2, 5, 6])
def test_two_planes_retain_one_control_and_rrms_per_row(rows):
    projection, rrms = torch.ops.custom_op.custom_deepseek_v41_control_fp8_pair_gaudi2(*operands(rows))
    assert projection.shape == (rows, 24) and rrms.shape == (rows, 1)
    assert projection.dtype == rrms.dtype == torch.float32


@pytest.mark.parametrize("rows", [1, 7])
def test_c1_and_larger_batches_keep_their_controller(rows):
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_control_fp8_pair_gaudi2(*operands(rows))
