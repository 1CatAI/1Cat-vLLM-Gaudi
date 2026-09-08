# SPDX-License-Identifier: Apache-2.0
"""CPU-side correctness gates shared by TP2 diagnostics."""

import math

import torch


def validate_outputs(actual, expected, *, atol, rtol, launches=None, expected_launches=None):
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("Validation tolerances must be finite and nonnegative")
    if len(actual) != 2 or len(expected) != 2:
        return False, [None, None]
    passed = expected_launches is None or launches == expected_launches
    errors = []
    for output, reference in zip(actual, expected):
        valid = (output.shape == reference.shape and output.dtype == reference.dtype and output.numel() > 0)
        finite = valid and bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
        error = (output.float() - reference.float()).abs().max().item() if finite else None
        errors.append(error)
        passed &= finite and torch.allclose(output.float(), reference.float(), atol=atol, rtol=rtol)
    return bool(passed), errors
