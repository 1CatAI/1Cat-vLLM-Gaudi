# SPDX-License-Identifier: Apache-2.0
"""Gaudi-native, FlashInfer-compatible inference primitives."""

from __future__ import annotations

import os

from flashinfer_gaudi import gdn_decode
from flashinfer_gaudi._capabilities import get_capabilities
from flashinfer_gaudi._config import clear_backend_policy_override, set_backend_policy
from flashinfer_gaudi._native import load_native_extensions

__version__ = "0.1.0"
FLASHINFER_API_COMPAT = "0.6.18"

if os.environ.get("FLASHINFER_GAUDI_ENABLE_COMPAT_SHIM", "0").strip().lower() in ("1", "true", "yes", "on"):
    from flashinfer_gaudi.compat import install_flashinfer_shim

    install_flashinfer_shim()

__all__ = [
    "FLASHINFER_API_COMPAT",
    "clear_backend_policy_override",
    "gdn_decode",
    "get_capabilities",
    "load_native_extensions",
    "set_backend_policy",
]
