# SPDX-License-Identifier: Apache-2.0
"""Shared fail-closed policy for complete operations, including their prologues."""

from __future__ import annotations

from flashinfer_gaudi._config import get_backend_policy

STRICT_BACKENDS = frozenset(("native", "public", "bridge"))


class BackendUnavailableError(RuntimeError):
    """The selected backend cannot execute the complete requested operation."""


def require_reference_allowed(operation: str) -> None:
    """Reject before allocations, decompositions, or state mutations occur.

    A native kernel surrounded by torch computation is not a native operation.
    Keep this check outside numerical reference functions so tests can replace
    those functions with sentinels and still verify the public boundary.
    """
    policy = get_backend_policy()
    if policy in STRICT_BACKENDS:
        raise BackendUnavailableError(f"{operation}: backend={policy} cannot execute this complete operation natively; "
                                      "PyTorch decomposition is forbidden. No reference implementation was executed.")
