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

    from flashinfer_gaudi import (
        chunk_gated_delta_rule,
        gdn_decode,
        gdn_fused_decode,
        gdn_fused_decode_step,
        gdn_fused_decode_step_supported,
        gdn_prefill,
    )

    root = sys.modules.get("flashinfer")
    if root is None:
        root = types.ModuleType("flashinfer")
        root.__path__ = []  # type: ignore[attr-defined]
        root.__package__ = "flashinfer"
        sys.modules["flashinfer"] = root
    root.gdn_decode = gdn_decode
    root.gdn_fused_decode = gdn_fused_decode
    root.gdn_prefill = gdn_prefill
    root.chunk_gated_delta_rule = chunk_gated_delta_rule
    root.gdn_fused_decode_step = gdn_fused_decode_step
    root.gdn_fused_decode_step_supported = gdn_fused_decode_step_supported
    sys.modules["flashinfer.gdn_decode"] = gdn_decode
    sys.modules["flashinfer.gdn_fused_decode"] = gdn_fused_decode
    sys.modules["flashinfer.gdn_prefill"] = gdn_prefill


__all__ = ["install_flashinfer_shim"]
