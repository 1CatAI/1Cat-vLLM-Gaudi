# SPDX-License-Identifier: Apache-2.0
"""Native activation kernels; unsupported native contracts fail before execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flashinfer_gaudi._config import get_backend_policy
from flashinfer_gaudi._dispatch import BackendUnavailableError, require_reference_allowed
from flashinfer_gaudi._native import silu_and_mul_op


def silu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None, enable_pdl: bool | None = None) -> torch.Tensor:
    """Compute SiLU(input[..., :D]) * input[..., D:].

    Native v1 supports nonempty contiguous rank-2 BF16 HPU input, D divisible
    by 128, and an allocated result. Preallocated outputs and other layouts
    remain reference-only until their mutation/capture contract is qualified.
    ``auto`` intentionally retains the reference until offline promotion.
    """
    if enable_pdl:
        raise BackendUnavailableError("silu_and_mul: CUDA programmatic dependent launch is unavailable on Gaudi2.")
    if input.ndim < 1 or input.shape[-1] == 0 or input.shape[-1] % 2:
        raise ValueError("silu_and_mul requires a nonzero even final dimension.")
    if out is not None and (out.shape != (*input.shape[:-1], input.shape[-1] // 2) or out.dtype != input.dtype
                            or out.device != input.device):
        raise ValueError("silu_and_mul output shape, dtype, and device must match input.")
    policy = get_backend_policy()
    if policy in ("native", "public"):
        if (input.device.type not in ("hpu", "privateuseone") or input.dtype != torch.bfloat16 or input.ndim != 2
                or not input.is_contiguous() or input.shape[0] == 0 or input.shape[1] % 256 or out is not None
                or input.requires_grad or max(input.shape) > 2**31 - 1):
            raise BackendUnavailableError("silu_and_mul: native v1 requires contiguous [B,2D] BF16 HPU input, "
                                          "B>0, D divisible by 128, and out=None.")
        op = silu_and_mul_op()
        if op is None:
            raise BackendUnavailableError("silu_and_mul: native library is unavailable; no fallback was executed.")
        return op(input)
    require_reference_allowed("silu_and_mul")
    gate, value = input.chunk(2, dim=-1)
    result = F.silu(gate) * value
    if out is not None:
        out.copy_(result)
        return out
    return result
