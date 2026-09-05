# SPDX-License-Identifier: Apache-2.0
"""Offline tactic selection for deterministic serving startup."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def _manifest() -> dict[str, object]:
    manifest = resources.files("flashinfer_gaudi").joinpath("tactics/gaudi2_qwen38_gdn.json")
    with manifest.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def public_gdn_auto_promoted() -> bool:
    # Environment flags must not bypass offline correctness/performance gates.
    return bool(_manifest().get("public_native_promoted", False) and _manifest().get("whole_operation_native", False))


def gdn_prefill_tactic() -> dict[str, object]:
    """Return the offline-promoted Gaudi2 GDN prefill tactic."""
    value = _manifest().get("gdn_prefill")
    return dict(value) if isinstance(value, dict) else {}


def gdn_fused_decode_tactic() -> dict[str, object]:
    """Return the offline-promoted Gaudi2 fused GDN decode tactic."""
    value = _manifest().get("gdn_fused_decode")
    return dict(value) if isinstance(value, dict) else {}


def tactic_manifest() -> dict[str, object]:
    return dict(_manifest())
