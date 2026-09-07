# SPDX-License-Identifier: Apache-2.0
"""Capability reporting for framework adapters and diagnostics."""

from __future__ import annotations

import torch
import json
from importlib import resources

from flashinfer_gaudi._config import get_backend_policy, get_state_precision, mtp_prepared_enabled
from flashinfer_gaudi._native import native_diagnostics
from flashinfer_gaudi._tactics import (
    dflash2_select_path_auto_promoted,
    dflash2_score_select_auto_promoted,
    dflash2_top_k_auto_promoted,
    public_mtp_gdn_auto_promoted,
    tactic_manifest,
)


def get_capabilities() -> dict[str, object]:
    native = native_diagnostics()
    hpu_available = bool(hasattr(torch, "hpu") and torch.hpu.is_available())
    return {
        "api_compat":
        "flashinfer-0.6.18",
        "device":
        "gaudi2",
        "hpu_available":
        hpu_available,
        "backend_policy":
        get_backend_policy(),
        "state_precision":
        get_state_precision(),
        "gdn_decode": {
            "reference": True,
            "public_native": False,
            "bridge_native": False,
            "public_tpc_prototype_loaded": native["public_packed_gdn"],
            "bridge_tpc_prototype_loaded": native["bridge_packed_gdn"],
            "state_layouts": ("VK", "KV"),
            "input_dtypes": ("bfloat16", "float32"),
            "state_dtypes": ("float32", "bfloat16"),
            "reference_decode_tokens": "any",
            "public_native_decode_tokens": (),
            "prototype_decode_tokens": (1, ),
            "separate_load_store_indices": True,
            "intermediate_mtp_states": True,
            "public_native_mtp_tokens": (8, ) if native["public_mtp_gdn"] else (),
            "public_native_mtp_auto_promoted": public_mtp_gdn_auto_promoted(),
            "experimental_prepared_mtp_available": native.get("public_mtp_prepared", False),
            "experimental_prepared_mtp_requested": mtp_prepared_enabled(),
        },
        "gdn_fused_decode": {
            "reference": True,
            "specialized": False,
            "registered_batches": (),
            "vllm_direct_recipe": True,
            "vllm_direct_recipe_batches": (1, 2, 4, 8, 16, 32),
            "conv_state_layouts": ("SD", "DS"),
        },
        "gdn_prefill": {
            "reference": True,
            "flashqla_graph": True,
            "input_layout": "NHD",
            "state_layout": "VK",
            "input_dtypes": ("bfloat16", "float32"),
            "state_dtypes": ("float32", "bfloat16"),
            "public_chunk_size": 64,
            "promoted_model_chunk_size": 128,
            "context_parallel": False,
            "state_checkpoints": False,
            "indexed_state_pool": False,
        },
        "dflash2": {
            "reference": True,
            "greedy_path": True,
            "grouped_conv_native": native["dflash2_grouped_conv"],
            "selector_path_native": native["dflash2_select_path"],
            "selector_path_auto_promoted": dflash2_select_path_auto_promoted(),
            "score_select_native": native["dflash2_score_select"],
            "score_select_auto_promoted": dflash2_score_select_auto_promoted(),
            "top_k_native": native["dflash2_top_k"],
            "top_k_vendor_cguid": True,
            "top_k_auto_promoted": dflash2_top_k_auto_promoted(),
            "query_tokens_per_request": 8,
            "selector_top_k": 16,
        },
        "native":
        native,
        "tactics":
        tactic_manifest(),
        "native_coverage":
        json.loads(resources.files("flashinfer_gaudi").joinpath("tactics/native_coverage.json").read_text()),
        "strict_backends": ("native", "public", "bridge"),
        "native_availability_is_not_qualification":
        True,
        "nvidia_only": (
            "cuda_ipc",
            "nvshmem",
            "cutlass",
            "cute_dsl",
            "deep_gemm",
        ),
    }
