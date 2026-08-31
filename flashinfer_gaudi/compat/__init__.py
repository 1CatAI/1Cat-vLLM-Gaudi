# SPDX-License-Identifier: Apache-2.0
"""Opt-in import shim for applications hard-coded to FlashInfer's namespace."""

from __future__ import annotations

import importlib.util
import sys
import types


def install_flashinfer_shim() -> None:
    """Expose supported modules under ``flashinfer`` when it is absent.

    The shim intentionally refuses to shadow an installed official package.
    vLLM-Gaudi does not depend on this compatibility mechanism.
    """
    existing = importlib.util.find_spec("flashinfer")
    if existing is not None and "flashinfer" not in sys.modules:
        raise RuntimeError("Official flashinfer is installed; refusing to shadow it with the Gaudi shim.")

    from flashinfer_gaudi import gdn_decode

    root = sys.modules.get("flashinfer")
    if root is None:
        root = types.ModuleType("flashinfer")
        root.__path__ = []  # type: ignore[attr-defined]
        root.__package__ = "flashinfer"
        sys.modules["flashinfer"] = root
    root.gdn_decode = gdn_decode
    sys.modules["flashinfer.gdn_decode"] = gdn_decode


__all__ = ["install_flashinfer_shim"]

