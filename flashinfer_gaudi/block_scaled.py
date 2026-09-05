# SPDX-License-Identifier: Apache-2.0
"""Experimental exact-contract block-weight dequantization and BF16 linear."""

from __future__ import annotations

import torch

from flashinfer_gaudi._config import get_backend_policy
from flashinfer_gaudi._dispatch import BackendUnavailableError, require_reference_allowed
from flashinfer_gaudi._native import block_fp8_dequant_op, block_fp8_linear_op


def _validate_weights(weight, scale):
    if (weight.ndim != 2 or scale.ndim != 2 or weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32
            or not weight.is_contiguous() or not scale.is_contiguous() or weight.requires_grad or scale.requires_grad
            or weight.device != scale.device or any(not 0 < dim <= 2**31 - 1 or dim % 128 for dim in weight.shape)
            or tuple(scale.shape) != (weight.shape[0] // 128, weight.shape[1] // 128)):
        raise BackendUnavailableError("block-FP8 v1 requires contiguous inference E4M3 [N,K] and FP32 [N/128,K/128]")


def _dequant_reference(weight, scale):
    n, k = weight.shape
    values = weight.reshape(n // 128, 128, k // 128, 128).to(torch.bfloat16)
    # Rounding the scale after multiplication would change the model contract.
    scales = scale.reshape(n // 128, 1, k // 128, 1).to(torch.bfloat16)
    return (values * scales).reshape(n, k)


def _linear_reference(x, weight, scale):
    return torch.nn.functional.linear(x, _dequant_reference(weight, scale))


def block_fp8_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return BF16 [N,K], preserving separate weight/scale BF16 casts and BF16 multiplication.

    Only aligned 128x128 blocks are supported. No padding, scale conversion,
    copies, data-dependent host reads or autograd. Load the private ABI-locked
    adapter before compilation. Auto/pytorch use the established reference;
    native/bridge fail closed and public rejects the private implementation.
    """
    _validate_weights(weight, scale)
    if get_backend_policy() in ("native", "bridge"):
        if weight.device.type != "hpu":
            raise BackendUnavailableError("Native block-FP8 dequant requires HPU; no fallback")
        op = block_fp8_dequant_op()
        if op is None:
            raise BackendUnavailableError("Native block-FP8 dequant unavailable for this device/build; no fallback")
        return op(weight, scale)
    require_reference_allowed("block_fp8_dequant")
    return _dequant_reference(weight, scale)


def block_fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """BF16 [M,K] times block-FP8 [N,K] weights, returning BF16 [M,N].

    The native recipe uses TPC dequantization followed by BF16 MME GEMM. It is
    NOT FP8 MME compute and does not quantize activations or change block
    scales. Weights are dequantized per call into recipe-owned scratch, not
    cached persistently. No bias, padding/unpadding, arbitrary strides or out
    mutation in v1. This primitive is experimental and never auto-promoted.
    """
    _validate_weights(weight, scale)
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.requires_grad
            or not 0 < x.shape[0] <= 2**31 - 1 or x.shape[1] != weight.shape[1] or x.device != weight.device):
        raise BackendUnavailableError("block-FP8 linear requires contiguous inference BF16 [M,K] on the weight device")
    if get_backend_policy() in ("native", "bridge"):
        if x.device.type != "hpu":
            raise BackendUnavailableError("Native block-FP8 linear requires HPU; no fallback")
        op = block_fp8_linear_op()
        if op is None:
            raise BackendUnavailableError("Native block-FP8 linear unavailable for this device/build; no fallback")
        return op(x, weight, scale)
    require_reference_allowed("block_fp8_linear")
    return _linear_reference(x, weight, scale)
