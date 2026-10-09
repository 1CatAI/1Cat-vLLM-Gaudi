# SPDX-License-Identifier: Apache-2.0
"""Experimental static exponent biases for bounded DSpark dense projections."""
import math

import numpy as np
import torch

from vllm_gaudi import envs
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import encode_gaudi2

SCALES = (1 / 256, 1 / 16, 1., 16.)
ROUNDING_MARGIN = 1.08  # BF16 norm + block32 E4M3 roundtrip + BF16 materialization.


def scale_covering(peak):
    if not math.isfinite(peak) or peak < 0:
        raise ValueError("Hardware FP8 requires a finite nonnegative maximum")
    for scale in SCALES:
        if peak <= 240 * scale:
            return scale
    raise ValueError("Projection exceeds Gaudi2 hardware exponent-bias range")


def encode_weight(weight):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Static dense preparation must read the decoded BF16 production weight")
    value = weight.cpu().float().numpy()
    scale = scale_covering(float(np.max(np.abs(value))))
    encoded = torch.from_numpy(encode_gaudi2(value / scale)).view(torch.float8_e4m3fn)
    return encoded.to(weight.device), scale


def prepare(attention, input_norm_weight):
    if not (envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE or envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT):
        return
    if attention._fused_qkv_weight is None or "fused_qkv_channel" in attention._buffers:
        raise ValueError("Static dense candidate requires the joined BF16 QKV production reference")
    query = attention.weights.wq_b
    if getattr(query, "dense_fp8", False) or getattr(query, "dense_fp8_direct_input", False):
        raise ValueError("Static query preparation requires decoded BF16 production weights")
    qkv, qkv_scale = encode_weight(attention._fused_qkv_weight)
    q, q_scale = encode_weight(query.weight)
    # RMS norm bounds every component by sqrt(K) * max(abs(gamma)).
    # This avoids data-dependent amax or a replay-time clipping decision.
    # Allow margin for the established BF16 norm/activation rounding.
    norm_peak = float(input_norm_weight.cpu().float().abs().max())
    query_peak = float(attention.weights.q_norm.weight.cpu().float().abs().max())
    attention.register_buffer("dspark_hw_qkv", qkv, False)
    attention.dspark_hw_qkv_weight_scale = qkv_scale
    attention.dspark_hw_qkv_input_scale = scale_covering(math.sqrt(qkv.shape[1]) * norm_peak * ROUNDING_MARGIN)
    query.register_buffer("dspark_hw_weight", q, False)
    query.dspark_hw_weight_scale = q_scale
    query.dspark_hw_input_scale = scale_covering(math.sqrt(q.shape[1]) * query_peak * ROUNDING_MARGIN)


def project(value, weight, input_scale, weight_scale):
    operation = (
        torch.ops.custom_op.custom_deepseek_v41_hw_dense_roundtrip_fp8_gaudi2
        if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT
        else torch.ops.custom_op.custom_deepseek_v41_hw_dense_fp8_gaudi2
    )
    return operation(value.contiguous(), weight, input_scale, weight_scale)
