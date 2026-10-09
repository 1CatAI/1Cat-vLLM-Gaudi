# SPDX-License-Identifier: Apache-2.0
"""The controller preserves C1 dispatch and validates prepared FP8 operands."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    library = os.getenv("DSV41_CONTROL_FP8_LIBRARY")
    if not library:
        pytest.skip("Set the additive control registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(library)


def inputs(rows):
    return [torch.empty((rows, 20480), dtype=torch.bfloat16, device="meta"),
            torch.empty((24, 20480), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((1, 24), dtype=torch.float32, device="meta"), 1e-20]


@pytest.mark.parametrize("rows", (2, 6))
def test_prepared_contract(rows):
    projection, rrms = torch.ops.custom_op.custom_deepseek_v41_control_fp8_rrms_gaudi2(*inputs(rows))
    assert projection.shape == (rows, 24) and rrms.shape == (rows, 1)
    assert projection.dtype == rrms.dtype == torch.float32


def test_c1_keeps_production_controller():
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_control_fp8_rrms_gaudi2(*inputs(1))


def test_reject_wrong_weights():
    values = inputs(6)
    values[1] = torch.empty((24, 20480), dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_control_fp8_rrms_gaudi2(*values)
