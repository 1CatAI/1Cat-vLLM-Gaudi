# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-compatible fused GDN decode step for Intel Gaudi.

The public operation mirrors FlashInfer's fused Qwen decode contract.  The
portable implementation is also the executable specification used to qualify
Gaudi-specific tactics before they are promoted into serving.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_QWEN38_BATCHES = frozenset((1, 2, 4, 8, 16, 32))
_QWEN38_GEOMETRY = (5120, 96, 10240, 16, 48, 128, 4, 3)


def _device_type(device: torch.device | str | None) -> str | None:
    if device is not None:
        return torch.device(device).type
    hpu = getattr(torch, "hpu", None)
    if hpu is not None and hpu.is_available():
        return "hpu"
    return None


def gdn_fused_decode_step_supported(
    batch_size: int,
    hidden_size: int = 5120,
    n_ba: int = 96,
    qkv_dim: int = 10240,
    num_qk_heads: int = 16,
    num_v_heads: int = 48,
    head_dim: int = 128,
    conv_width: int = 4,
    conv_state_len: int = 3,
    device: torch.device | str | None = None,
    conv_state_layout: str = "SD",
) -> bool:
    """Return whether an offline-promoted Gaudi fused tactic serves a shape.

    ``False`` does not make :func:`gdn_fused_decode_step` unavailable: the
    operation remains correct through its composable implementation.  It tells
    framework adapters to retain their already-optimized composition until a
    faster Gaudi tactic has passed qualification.
    """
    geometry = (
        hidden_size,
        n_ba,
        qkv_dim,
        num_qk_heads,
        num_v_heads,
        head_dim,
        conv_width,
        conv_state_len,
    )
    if geometry != _QWEN38_GEOMETRY or batch_size not in _QWEN38_BATCHES:
        return False
    if conv_state_layout not in ("SD", "DS"):
        return False
    if _device_type(device) not in ("hpu", "privateuseone"):
        return False

    # No fused Gaudi tactic is promoted yet.  Keep this explicit so adding the
    # public API cannot silently route serving onto its correctness fallback.
    return False


def _scatter_live_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    padding: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Update live rows while making duplicate padding writes deterministic."""
    owns_zero = ~padding & (slots == 0)
    donor = torch.argmax(owns_zero.to(torch.int64)).reshape(1)
    slot_zero = torch.where(owns_zero.any(), rows.index_select(0, donor), pool[0:1])
    mask = padding.reshape((-1, ) + (1, ) * (rows.ndim - 1))
    pool.index_copy_(0, slots.clamp_min(0), torch.where(mask, slot_zero, rows))


def _validate_inputs(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if hidden_states.ndim != 2 or mixed_qkv.ndim != 2:
        raise ValueError("hidden_states and mixed_qkv must both be rank-2 tensors.")
    batch, hidden_size = hidden_states.shape
    if mixed_qkv.shape[0] != batch:
        raise ValueError("hidden_states and mixed_qkv must have the same batch size.")
    value_heads = A_log.numel()
    if value_heads == 0 or dt_bias.shape != A_log.shape:
        raise ValueError("A_log and dt_bias must have the same non-empty [HV] shape.")
    if w_ba.shape != (hidden_size, 2 * value_heads):
        raise ValueError(f"w_ba has shape {tuple(w_ba.shape)}, expected {(hidden_size, 2 * value_heads)}.")
    if ssm_state.ndim != 4 or ssm_state.shape[1] != value_heads:
        raise ValueError("ssm_state must have [P,HV,V,K] shape matching A_log.")
    value_dim, key_dim = ssm_state.shape[-2:]
    qkv_dim = mixed_qkv.shape[1]
    qk_width = qkv_dim - value_heads * value_dim
    if qk_width <= 0 or qk_width % (2 * key_dim):
        raise ValueError("mixed_qkv width is incompatible with the recurrent-state geometry.")
    qk_heads = qk_width // (2 * key_dim)
    if value_heads % qk_heads:
        raise ValueError("The number of value heads must be divisible by the number of Q/K heads.")
    if conv_state.ndim != 3 or conv_state.shape[0] != ssm_state.shape[0] or conv_state.shape[1] != qkv_dim:
        raise ValueError("conv_state must be a logical [P,qkv_dim,width-1] view.")
    conv_width = conv_state.shape[2] + 1
    if conv_weight.shape != (qkv_dim, conv_width) or conv_bias.shape != (qkv_dim, ):
        raise ValueError("conv_weight or conv_bias does not match mixed_qkv and conv_state.")
    if state_indices.shape != (batch, ):
        raise ValueError(f"state_indices has shape {tuple(state_indices.shape)}, expected {(batch,)}.")
    expected_out = (batch, 1, value_heads, value_dim)
    if out is not None and out.shape != expected_out:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {expected_out}.")
    return value_heads, qk_heads, key_dim, value_dim


def gdn_fused_decode_step(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool = True,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one fused-contract GDN decode step and update both state pools.

    The tensor contract matches ``flashinfer.gdn_fused_decode_step``.  The
    current Gaudi implementation is composable and intentionally is not
    advertised by :func:`gdn_fused_decode_step_supported` until a specialized
    tactic beats the existing vLLM-Gaudi path.
    """
    value_heads, qk_heads, key_dim, value_dim = _validate_inputs(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        out,
    )
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(key_dim)

    indices = state_indices.to(device=ssm_state.device, dtype=torch.long)
    padding = indices < 0
    safe_indices = indices.clamp_min(0)

    ba = torch.matmul(hidden_states.float(), w_ba.float()).to(torch.bfloat16)
    b_gate, a_gate = ba.split(value_heads, dim=-1)

    state_rows = conv_state.index_select(0, safe_indices)
    raw_qkv = mixed_qkv.to(conv_state.dtype)
    window = torch.cat((state_rows, raw_qkv.unsqueeze(-1)), dim=-1)
    convolved = torch.sum(window.float() * conv_weight.float().unsqueeze(0), dim=-1)
    convolved = convolved + conv_bias.float()
    convolved = (convolved * torch.sigmoid(convolved)).to(torch.bfloat16)
    _scatter_live_rows(conv_state, indices, padding, window[..., 1:])

    q_width = qk_heads * key_dim
    q, k, v = convolved.split((q_width, q_width, value_heads * value_dim), dim=-1)
    batch = hidden_states.shape[0]
    q = q.reshape(batch, qk_heads, key_dim).float()
    k = k.reshape_as(q).float()
    v = v.reshape(batch, value_heads, value_dim).float()

    if use_qk_l2norm:
        q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)

    decay = torch.exp(-torch.exp(A_log.float()) * F.softplus(a_gate.float() + dt_bias.float()))
    beta = torch.sigmoid(b_gate.float())
    selected_state = ssm_state.index_select(0, safe_indices)
    head_repeat = value_heads // qk_heads
    state = selected_state.reshape(batch, qk_heads, head_repeat, value_dim, key_dim)
    q = q.unsqueeze(2)
    k = k.unsqueeze(2)
    v = v.reshape(batch, qk_heads, head_repeat, value_dim)
    decay = decay.reshape(batch, qk_heads, head_repeat, 1, 1)
    beta = beta.reshape(batch, qk_heads, head_repeat, 1)

    state = state * decay
    old_value = torch.matmul(state, k.unsqueeze(-1)).squeeze(-1)
    delta = beta * (v - old_value)
    state = torch.addcmul(state, delta.unsqueeze(-1), k.unsqueeze(-2))
    result = torch.matmul(state, (q * scale).unsqueeze(-1)).squeeze(-1)
    selected_state.copy_(state.reshape_as(selected_state))
    _scatter_live_rows(ssm_state, indices, padding, selected_state)
    result = result.reshape(batch, 1, value_heads, value_dim).to(torch.bfloat16)
    result = result.masked_fill(padding.reshape(-1, 1, 1, 1), 0)

    if out is not None:
        out.copy_(result)
        result = out
    return result, conv_state, ssm_state


__all__ = ["gdn_fused_decode_step", "gdn_fused_decode_step_supported"]
