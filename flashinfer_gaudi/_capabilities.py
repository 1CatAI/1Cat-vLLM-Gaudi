# SPDX-License-Identifier: Apache-2.0
"""Capability reporting for framework adapters and diagnostics."""

from __future__ import annotations

import torch

from flashinfer_gaudi._config import get_backend_policy, get_state_precision
from flashinfer_gaudi._native import native_diagnostics
from flashinfer_gaudi._tactics import tactic_manifest


def get_capabilities() -> dict[str, object]:
    native = native_diagnostics()
    hpu_available = bool(hasattr(torch, "hpu") and torch.hpu.is_available())
    return {
        "api_compat": "flashinfer-0.6.18",
        "device": "gaudi2",
        "hpu_available": hpu_available,
        "backend_policy": get_backend_policy(),
        "state_precision": get_state_precision(),
        "gdn_decode": {
            "reference": True,
            "public_native": native["public_packed_gdn"],
            "bridge_native": native["bridge_packed_gdn"],
            "state_layouts": ("VK", "KV"),
            "input_dtypes": ("bfloat16", "float32"),
            "state_dtypes": ("float32", "bfloat16"),
            "reference_decode_tokens": "any",
            "public_native_decode_tokens": (1, ),
            "separate_load_store_indices": True,
            "intermediate_mtp_states": True,
        },
        "native": native,
        "tactics": tactic_manifest(),
        "nvidia_only": (
            "cuda_ipc",
            "nvshmem",
            "cutlass",
            "cute_dsl",
            "deep_gemm",
        ),
    }
