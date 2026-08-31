# SPDX-License-Identifier: Apache-2.0
"""Offline tactic selection for deterministic serving startup."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def _manifest() -> dict[str, object]:
    manifest = resources.files("flashinfer_gaudi").joinpath("tactics/gaudi2_qwen38_gdn.json")
    with manifest.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def public_gdn_auto_promoted() -> bool:
    override = os.environ.get("FLASHINFER_GAUDI_ENABLE_PUBLIC_AUTO")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    return bool(_manifest().get("public_native_promoted", False))


def tactic_manifest() -> dict[str, object]:
    return dict(_manifest())
