# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration for the Gaudi FlashInfer compatibility layer."""

from __future__ import annotations

import os
from typing import Literal, cast

BackendPolicy = Literal["auto", "native", "public", "bridge", "pytorch"]
StatePrecision = Literal["fp32", "bf16"]

_BACKEND_ENV = "FLASHINFER_GAUDI_BACKEND"
_STATE_DTYPE_ENV = "FLASHINFER_GAUDI_STATE_DTYPE"
_VALID_BACKENDS = frozenset(("auto", "native", "public", "bridge", "pytorch"))
_VALID_STATE_DTYPES = frozenset(("fp32", "bf16"))

_backend_override: BackendPolicy | None = None


def _read_choice(name: str, default: str, choices: frozenset[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"Invalid {name}={value!r}; expected one of: {expected}.")
    return value


def get_backend_policy() -> BackendPolicy:
    """Return the process-wide backend policy.

    ``auto`` prefers a validated public HPU implementation. The ABI-private
    bridge is only considered automatically when explicitly enabled.
    """
    if _backend_override is not None:
        return _backend_override
    return cast(BackendPolicy, _read_choice(_BACKEND_ENV, "auto", _VALID_BACKENDS))


def set_backend_policy(policy: BackendPolicy) -> BackendPolicy:
    """Override the backend policy and return the previous effective value."""
    global _backend_override
    if policy not in _VALID_BACKENDS:
        expected = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Invalid backend policy {policy!r}; expected one of: {expected}.")
    previous = get_backend_policy()
    _backend_override = policy
    return previous


def clear_backend_policy_override() -> None:
    """Return backend selection to the environment-controlled policy."""
    global _backend_override
    _backend_override = None


def get_state_precision() -> StatePrecision:
    return cast(StatePrecision, _read_choice(_STATE_DTYPE_ENV, "fp32", _VALID_STATE_DTYPES))


def bridge_auto_enabled() -> bool:
    # The bridge prototype has no qualified whole-operation native tactic.
    # A process environment flag is not evidence of qualification.
    return False


def mtp_prepared_enabled() -> bool:
    """Opt into qualification of the graph-native B1/T8 MTP core."""
    return os.environ.get("FLASHINFER_GAUDI_ENABLE_MTP_PREPARED", "0").strip().lower() in ("1", "true", "yes", "on")
