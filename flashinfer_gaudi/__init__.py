# SPDX-License-Identifier: Apache-2.0
"""Gaudi-native, FlashInfer-compatible inference primitives."""

from __future__ import annotations

import os

from flashinfer_gaudi import dflash2, gdn_decode, gdn_fused_decode, gdn_prefill
from flashinfer_gaudi._capabilities import get_capabilities
from flashinfer_gaudi._config import clear_backend_policy_override, set_backend_policy
from flashinfer_gaudi._native import load_native_extensions
from flashinfer_gaudi.activation import silu_and_mul
from flashinfer_gaudi.quantization import silu_and_mul_quant
from flashinfer_gaudi.block_scaled import block_fp8_dequant, block_fp8_linear
from flashinfer_gaudi.gdn_prefill import chunk_gated_delta_rule
from flashinfer_gaudi.gdn_fused_decode import gdn_fused_decode_step, gdn_fused_decode_step_supported
from flashinfer_gaudi.gdn_decode import gated_delta_rule_mtp_packed, gated_delta_rule_mtp_rollback
from flashinfer_gaudi.dflash2 import top_k

__version__ = "0.1.0"
FLASHINFER_API_COMPAT = "0.6.18"

if os.environ.get("FLASHINFER_GAUDI_ENABLE_COMPAT_SHIM", "0").strip().lower() in ("1", "true", "yes", "on"):
    from flashinfer_gaudi.compat import install_flashinfer_shim

    install_flashinfer_shim()

__all__ = [
    "FLASHINFER_API_COMPAT",
    "clear_backend_policy_override",
    "chunk_gated_delta_rule",
    "dflash2",
    "gdn_decode",
    "gdn_fused_decode",
    "gdn_fused_decode_step",
    "gdn_fused_decode_step_supported",
    "gdn_prefill",
    "gated_delta_rule_mtp_rollback",
    "gated_delta_rule_mtp_packed",
    "get_capabilities",
    "load_native_extensions",
    "set_backend_policy",
    "silu_and_mul",
    "silu_and_mul_quant",
    "block_fp8_dequant",
    "block_fp8_linear",
    "top_k",
]
