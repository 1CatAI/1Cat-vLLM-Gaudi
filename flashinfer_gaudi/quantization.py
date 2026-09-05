# SPDX-License-Identifier: Apache-2.0
"""Gaudi-specific gated activation and row-wise FP8 quantization candidate."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flashinfer_gaudi._config import get_backend_policy
from flashinfer_gaudi._dispatch import BackendUnavailableError, require_reference_allowed
from flashinfer_gaudi._native import silu_mul_quant_op


def _silu_and_mul_quant_reference(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = input.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    scale = (activated.abs().amax(dim=-1, keepdim=True) + 1e-8) / 240.0
    inverse = scale.reciprocal()
    if input.device.type == "hpu":
        result = torch.ops.hpu.cast_to_fp8_v2(activated, inverse, False, False, torch.float8_e4m3fn)[0]
    else:
        result = (activated.float() * inverse.float()).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return result, scale.float()


def silu_and_mul_quant(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 values [B,D] and FP32 dequantization scales [B,1].

    This is a Gaudi extension, not upstream FlashInfer's block/NVFP4 API.
    SiLU, multiplication, epsilon addition, scale division and reciprocal
    retain their HPU BF16 boundaries. SiLU uses a native Synapse node;
    the remaining chain is one custom TPC kernel. The FP8 range is +/-240.
    Load the ABI-locked Bridge adapter before compilation. Native v1 accepts
    contiguous inference BF16 [B,2D], B>0, D divisible by 128 in [256,17408].
    No hidden cast, contiguous, output copy, or numerical torch prologue is
    performed in native/bridge mode. Auto/pytorch retain the reference;
    public rejects this private-ABI implementation.
    """
    if input.ndim != 2 or input.shape[0] <= 0 or input.shape[1] <= 0 or input.shape[1] % 2:
        raise ValueError("SiLU-quant requires nonempty [B,2D] input")
    if input.dtype != torch.bfloat16 or input.requires_grad:
        raise BackendUnavailableError("SiLU-quant v1 requires inference BF16 input")
    if get_backend_policy() in ("native", "bridge"):
        if (input.device.type != "hpu" or not input.is_contiguous() or input.shape[1] < 512 or input.shape[1] > 34816
                or input.shape[1] % 256 or input.shape[0] > 2**31 - 1):
            raise BackendUnavailableError("SiLU-quant native shape/layout/device is unsupported; no fallback executed")
        op = silu_mul_quant_op()
        if op is None:
            raise BackendUnavailableError("SiLU-quant native library is unavailable; no fallback executed")
        return op(input)
    require_reference_allowed("silu_and_mul_quant")
    return _silu_and_mul_quant_reference(input)
