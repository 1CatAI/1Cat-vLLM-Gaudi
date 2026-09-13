# SPDX-License-Identifier: Apache-2.0
"""Functional residual/RMSNorm/row-FP8 fusion for the Gaudi MME input contract."""

from __future__ import annotations

import math

import torch

from flashinfer_gaudi._config import get_backend_policy
from flashinfer_gaudi._dispatch import BackendUnavailableError, require_reference_allowed
from flashinfer_gaudi._native import add_rmsnorm_quant_op


def _reference(input, residual, weight, eps, scale_mode="bf16_reciprocal"):
    summed = input + residual
    fp32 = summed.float()
    # Preserve the BF16 weight-product boundary of the current HPU vendor op.
    weighted = (summed * weight).float()
    normalized = (weighted * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)).to(input.dtype)
    inverse_range = 1.0 / 240.0 if scale_mode == "fp32_divide" else 0.004180908203125
    maximum = normalized.abs().amax(-1, keepdim=True) + 1e-8
    # Keep the BF16-mode multiply in BF16. A cast-to-FP32 / multiply /
    # cast-to-BF16 / cast-to-FP32 chain can lose the intermediate rounding in
    # current HPU graph optimization, changing both scale and FP8 bins.
    scale = ((maximum.float() * inverse_range).to(input.dtype) if scale_mode == "fp32_divide" else maximum *
             inverse_range)
    inverse = scale.reciprocal()
    if input.device.type == "hpu":
        quantized = torch.ops.hpu.cast_to_fp8_v2(normalized, inverse, False, False, torch.float8_e4m3fn)[0]
    else:
        quantized = (normalized * inverse).float().clamp(-240, 240).to(torch.float8_e4m3fn)
    return quantized, scale.float(), normalized, summed


def fused_add_rmsnorm_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    scale_mode: str = "bf16_reciprocal",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (FP8 values, FP32 row scales, BF16 normalized, BF16 residual).

    This functional Gaudi extension is not upstream's in-place or block-scaled
    API. All four outputs are new tensors; inputs are read-only. Scale and FP8
    conversion preserve the existing HPU BF16 weight-product/row-quant boundaries and
    the Gaudi2 +/-240 range. The public native path is one TPC kernel with no
    numerical torch prologue. Load the native library before torch.compile.
    Auto/pytorch remain reference-only until full-operation qualification.
    Scale modes make the two observed HPU compiler contracts explicit:
    bf16_reciprocal rounds 1/240 to BF16 before multiplying the row maximum;
    fp32_divide uses the FP32 reciprocal instead. Observed vendor/CGUID graphs
    do not always share this boundary, even with the same GEMM consumer.
    Neither mode is production-qualified; select by the consumer's validated
    numerical contract, never by guessing from shape or the presence of GEMM.
    """
    if (input.ndim != 2 or min(input.shape) <= 0 or residual.shape != input.shape
            or weight.shape != (input.shape[1], )):
        raise ValueError("Norm-quant requires nonempty [B,D], matching residual, and weight [D]")
    if not math.isfinite(eps) or not 1.1754943508222875e-38 <= eps <= 3.4028234663852886e38:
        raise ValueError("Norm-quant epsilon must be a positive normal FP32 value")
    if scale_mode not in ("bf16_reciprocal", "fp32_divide"):
        raise ValueError("Norm-quant scale_mode must be bf16_reciprocal or fp32_divide")
    if any(value.dtype != torch.bfloat16 or value.requires_grad for value in (input, residual, weight)):
        raise BackendUnavailableError("Norm-quant requires inference BF16 tensors")
    if any(value.device != input.device for value in (residual, weight)):
        raise ValueError("Norm-quant inputs must use the same device")
    if get_backend_policy() in ("native", "public"):
        if (input.device.type != "hpu" or input.shape[0] > 2**31 - 1 or not 256 <= input.shape[1] <= 17408
                or input.shape[1] % 128 or any(not value.is_contiguous() for value in (input, residual, weight))):
            raise BackendUnavailableError("Norm-quant native device/shape/layout unsupported; no fallback executed")
        op = add_rmsnorm_quant_op()
        if op is None:
            raise BackendUnavailableError("Norm-quant native library is unavailable; no fallback executed")
        return op(input, residual, weight, eps, scale_mode == "fp32_divide")
    require_reference_allowed("fused_add_rmsnorm_quant")
    return _reference(input, residual, weight, eps, scale_mode)
