# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU-native MiniMax H3 modulation operators.

The pinned Omni implementation uses CUDA Triton for accelerator tensors.  An
HPU process can import that module through vLLM's Triton compatibility stub,
but the stub cannot launch the CUDA kernels.  These implementations preserve
the H3 FP32 accumulation boundaries while keeping every model tensor on HPU.
"""

from __future__ import annotations

import torch


def _indexed(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(values, 0, indices)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type == "hpu":
        from vllm_gaudi.extension.kernels import rms_norm

        fused_rms_norm = rms_norm()
        if fused_rms_norm is None:
            raise RuntimeError("MiniMax H3 requires the Habana FusedRMSNorm kernel on HPU")
        # Habana FusedRMSNorm accepts the model's three-dimensional hidden
        # layout.  H3 packs requests into a two-dimensional token matrix.
        return fused_rms_norm.apply(x.unsqueeze(0), weight, eps).squeeze(0)

    normalized = x.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    return (normalized * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def indexed_scale_shift_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply indexed AdaLN affine parameters to a disposable input in-place."""

    if x.numel() == 0:
        return x
    selected_shift = _indexed(shift, indices).float()
    selected_scale = _indexed(scale, indices).float().add_(1.0)
    x.copy_(torch.addcmul(selected_shift, x.float(), selected_scale).to(x.dtype))
    return x


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return the H3 gated residual with FP32 multiply/add accumulation."""

    if x.numel() == 0:
        return torch.empty_like(x)
    selected_gate = _indexed(gate, indices).float()
    return torch.addcmul(x.float(), selected_gate, other.float()).to(x.dtype)


def rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run H3 RMSNorm followed by indexed AdaLN without a host fallback."""

    if x.numel() == 0:
        return torch.empty_like(x)
    normalized = _rms_norm(x, weight, eps)
    selected_shift = _indexed(shift, indices).float()
    selected_scale = _indexed(scale, indices).float().add_(1.0)
    return torch.addcmul(selected_shift, normalized.float(), selected_scale).to(x.dtype)


def indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the H3 gated residual, RMSNorm, and indexed AdaLN on HPU."""

    if residual.numel() == 0:
        return torch.empty_like(residual), torch.empty_like(residual)
    selected_gate = _indexed(gate, indices).float()
    residual_out = torch.addcmul(residual.float(), selected_gate, branch.float()).to(residual.dtype)
    normalized = _rms_norm(residual_out, weight, eps)
    selected_shift = _indexed(shift, indices).float()
    selected_scale = _indexed(scale, indices).float().add_(1.0)
    modulated = torch.addcmul(selected_shift, normalized.float(), selected_scale).to(residual.dtype)
    return residual_out, modulated


__all__ = [
    "indexed_gate",
    "indexed_gate_rms_norm_scale_shift",
    "indexed_scale_shift_",
    "rms_norm_indexed_scale_shift",
]
