# SPDX-License-Identifier: Apache-2.0
"""The partial-GEMM path retains the existing ownership and shape contract."""
import os

import pytest
import torch

from test_split_scale_planes_meta import operands


def operation():
    name = "custom_deepseek_v41_expert_n256_moe_w13_k_pipeline"
    if os.getenv("DSV41_EXPERT_K_TILE", "128") == "512":
        name += "_k512"
    elif os.getenv("DSV41_W13_K_PIPELINE_STAGES", "2") == "4":
        name += "_four"
    return getattr(torch.ops.custom_op, name + "_fp8_gaudi2")


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_W13_K_PIPELINE_LIBRARY")
    if not path:
        pytest.skip("Set the additive partial-GEMM registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize("rows,intermediate,compact", ((2, 640, True), (6, 640, True), (6, 1152, False)))
def test_checkpoint_shape(rows, intermediate, compact):
    result = operation()(*operands(rows, intermediate, compact))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


def test_reject_independent_group_storage():
    args = operands()
    args[12] = args[12].clone()
    with pytest.raises(RuntimeError, match="original storage/strides"):
        operation()(*args)


def test_c1_retains_original_dispatch():
    with pytest.raises(RuntimeError):
        operation()(*operands(1))
