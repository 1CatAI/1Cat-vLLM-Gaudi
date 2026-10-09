# SPDX-License-Identifier: Apache-2.0
"""Weighted selection contracts for full production vocabulary and native capture."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_WEIGHTED_NUCLEUS_LIBRARY")
    if not path:
        pytest.skip("Set the independently built weighted nucleus registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def inputs(rows=5, columns=129280):
    return (torch.empty((rows, columns), device="meta", dtype=torch.float32),
            torch.empty((rows, columns), device="meta", dtype=torch.float32),
            torch.empty((rows, 1), device="meta", dtype=torch.int32))


@pytest.mark.parametrize("rows", (1, 5, 6))
def test_partition_tail_and_refinement_geometry(rows):
    scores, probability, prefix = inputs(rows)
    ops = torch.ops.custom_op
    partial = ops.custom_deepseek_v41_weighted_mass_bins_gaudi2(scores, probability, prefix, 28)
    assert partial.shape == (rows, 64, 16) and partial.dtype == torch.float32
    target = torch.empty((rows, 1), device="meta", dtype=torch.float32)
    next_prefix, remaining = ops.custom_deepseek_v41_weighted_mass_advance_gaudi2(
        partial, prefix, target, 28)
    assert next_prefix.shape == remaining.shape == (rows, 1)
    assert next_prefix.dtype == torch.int32 and remaining.dtype == torch.float32
    kept, counts, ids = ops.custom_deepseek_v41_weighted_mass_finish_gaudi2(scores, probability, next_prefix)
    assert kept.shape == scores.shape and counts.shape == ids.shape == (rows, 64)
    assert counts.dtype == ids.dtype == torch.int32


@pytest.mark.parametrize("kind", ("shift", "tokens", "prefix", "probability"))
def test_reject_incompatible_coordinates(kind):
    values = list(inputs(7 if kind == "tokens" else 5))
    if kind == "prefix":
        values[2] = torch.empty((5, 1), device="meta", dtype=torch.int64)
    if kind == "probability":
        values[1] = torch.empty((5, 128), device="meta", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Weighted"):
        torch.ops.custom_op.custom_deepseek_v41_weighted_mass_bins_gaudi2(*values, 29 if kind == "shift" else 28)


def test_fake_functionalization_preserves_pure_outputs():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        values = [torch.empty(v.shape, dtype=v.dtype) for v in inputs()]
        fn = torch.func.functionalize(
            lambda *args: torch.ops.custom_op.custom_deepseek_v41_weighted_mass_finish_gaudi2(*args))
        retained, counts, ids = fn(*values)
        assert retained.shape == (5, 129280) and counts.shape == ids.shape == (5, 64)
