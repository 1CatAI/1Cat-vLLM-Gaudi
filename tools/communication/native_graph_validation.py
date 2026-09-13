# SPDX-License-Identifier: Apache-2.0
"""Exact, per-step validation shared by native graph qualification tools."""

from collections.abc import Iterable

import torch


def assert_exact(actual, expected, *, step):
    if len(actual) != len(expected):
        raise RuntimeError(f"Step {step}: output count changed")
    errors = []
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeError(f"Step {step}, output {index}: layout/dtype mismatch")
        mismatches = int(torch.count_nonzero(value != reference))
        maximum = float((value.float() - reference.float()).abs().max())
        errors.append({"mismatches": mismatches, "max_abs": maximum})
        if not torch.equal(value, reference):
            raise RuntimeError(f"Step {step}, output {index}: exact mismatch: {errors[-1]}")
    return errors


def validate_snapshots(pending: Iterable, references):
    """Check every queued snapshot, including earlier outputs in a drained batch."""
    count = 0
    for step, snapshot in pending:
        actual = tuple(value.cpu() for value in snapshot)
        assert_exact(actual, references[step], step=step)
        count += 1
    return count
