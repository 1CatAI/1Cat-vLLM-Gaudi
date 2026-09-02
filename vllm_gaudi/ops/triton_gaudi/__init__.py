# SPDX-License-Identifier: Apache-2.0

"""Gaudi2-native Triton fast paths used by vLLM hot operators."""

from vllm_gaudi.ops.triton_gaudi.runtime import (FastPathMode, diagnostics,
                                                 dynamic_quant,
                                                 fused_add_rms_norm,
                                                 gdn_decode_conv_packed,
                                                 gdn_decode_conv_split_packed,
                                                 gdn_decode_packed,
                                                 prepare_if_enabled,
                                                 silu_and_mul, vector_add)

__all__ = [
    "FastPathMode",
    "diagnostics",
    "dynamic_quant",
    "fused_add_rms_norm",
    "gdn_decode_conv_packed",
    "gdn_decode_conv_split_packed",
    "gdn_decode_packed",
    "prepare_if_enabled",
    "silu_and_mul",
    "vector_add",
]
