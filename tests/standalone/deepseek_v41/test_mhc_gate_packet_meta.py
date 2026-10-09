# SPDX-License-Identifier: Apache-2.0
"""The common C1 gate consumer accepts native C6 and parameterized peer ranks."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_MHC_GATE_PACKET_LIBRARY")
    if not path:
        pytest.skip("Set the common gate-packet registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def operands(rows, ranks):
    shape = (rows, 5120) if ranks == 1 else (ranks, rows, 5120)
    return (torch.empty(shape, dtype=torch.bfloat16, device="meta"),
            torch.empty((rows, 4, 5120), dtype=torch.bfloat16, device="meta"),
            torch.empty((rows, 24), dtype=torch.float32, device="meta"))


@pytest.mark.parametrize("rows,ranks", ((1, 1), (6, 1), (6, 4)))
def test_shared_shape_contract(rows, ranks):
    residual, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_post_gaudi2(*operands(rows, ranks))
    assert residual.shape == (rows, 4, 5120) and collapsed.shape == (rows, 5120)
    assert residual.dtype == collapsed.dtype == torch.bfloat16


def test_reject_incomplete_packet():
    args = list(operands(6, 1))
    args[2] = args[2][:, :20]
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_mhc_gates_post_gaudi2(*args)
