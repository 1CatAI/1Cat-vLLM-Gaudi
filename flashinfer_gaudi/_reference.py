# SPDX-License-Identifier: Apache-2.0
"""Numerically conservative PyTorch references for Gaudi inference ops."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _as_bt_heads(value: torch.Tensor, batch: int, tokens: int, heads: int, name: str) -> torch.Tensor:
    if value.numel() != batch * tokens * heads:
        raise ValueError(
            f"{name} must contain B*T*HV={batch * tokens * heads} values, got shape {tuple(value.shape)}."
        )
    return value.reshape(batch, tokens, heads)


def _canonical_indices(
    indices: torch.Tensor | None,
    batch: int,
    slots: int,
    device: torch.device,
) -> torch.Tensor:
    if indices is None:
        if slots < batch:
            raise ValueError(f"State pool has {slots} slots but decode batch is {batch}.")
        return torch.arange(batch, dtype=torch.long, device=device)
    if indices.numel() != batch:
        raise ValueError(f"State indices must contain {batch} entries, got shape {tuple(indices.shape)}.")
    return indices.reshape(batch).to(device=device, dtype=torch.long)


def _validate_cpu_store_indices(indices: torch.Tensor, slots: int) -> None:
    if indices.device.type != "cpu":
        return
    valid = indices[indices >= 0]
    if valid.numel() and int(valid.max()) >= slots:
        raise IndexError(f"State index {int(valid.max())} is outside a pool with {slots} slots.")
    if valid.numel() != torch.unique(valid).numel():
        raise ValueError("Non-negative output state indices must be unique within a decode call.")


def _safe_indices(indices: torch.Tensor, slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (indices >= 0) & (indices < slots)
    return torch.where(valid, indices, torch.zeros_like(indices)), valid


def _write_state_rows(
    state_pool: torch.Tensor,
    store_indices: torch.Tensor,
    updated_state: torch.Tensor,
    valid_store: torch.Tensor,
) -> None:
    # Keep shapes static for torch.compile. Invalid rows target slot zero but
    # write back the value read immediately before the update, so padding is a
    # true no-op even when slot zero contains live data.
    for batch_idx in range(updated_state.shape[0]):
        destination = store_indices.narrow(0, batch_idx, 1)
        valid = valid_store.narrow(0, batch_idx, 1)
        safe_destination = torch.where(valid, destination, torch.zeros_like(destination))
        previous = state_pool.index_select(0, safe_destination)
        candidate = updated_state.narrow(0, batch_idx, 1).to(state_pool.dtype)
        row = torch.where(valid.reshape(1, 1, 1, 1), candidate, previous)
        state_pool.index_copy_(0, safe_destination, row)


def recurrent_decode_from_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    load_state_indices: torch.Tensor | None = None,
    store_state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute GDN decode while preserving the recurrent update order."""
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(f"q and k must have matching [B,T,H,K] shapes, got {q.shape} and {k.shape}.")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"v must have [B,T,HV,V] with matching B,T, got {v.shape}.")
    if state_pool.ndim != 4:
        raise ValueError(f"state must have [slots,HV,V,K] shape, got {state_pool.shape}.")

    batch, tokens, q_heads, key_dim = q.shape
    _, _, value_heads, value_dim = v.shape
    slots, state_heads, state_value_dim, state_key_dim = state_pool.shape
    if state_heads != value_heads or state_value_dim != value_dim or state_key_dim != key_dim:
        raise ValueError(
            "State shape does not match Q/K/V: "
            f"state={tuple(state_pool.shape)}, q={tuple(q.shape)}, v={tuple(v.shape)}."
        )
    if value_heads % q_heads:
        raise ValueError(f"Value heads ({value_heads}) must be divisible by Q/K heads ({q_heads}).")

    load_indices = _canonical_indices(load_state_indices, batch, slots, state_pool.device)
    store_indices = _canonical_indices(store_state_indices, batch, slots, state_pool.device)
    _validate_cpu_store_indices(store_indices, slots)
    safe_load, valid_load = _safe_indices(load_indices, slots)
    _, valid_store = _safe_indices(store_indices, slots)

    state = state_pool.index_select(0, safe_load).to(torch.float32)
    state = torch.where(valid_load.reshape(batch, 1, 1, 1), state, torch.zeros_like(state))

    repeat = value_heads // q_heads
    state = state.reshape(batch, q_heads, repeat, value_dim, key_dim)
    q_work = q.to(torch.float32)
    k_work = k.to(torch.float32)
    if use_qk_l2norm:
        q_work = F.normalize(q_work, p=2.0, dim=-1, eps=1e-6)
        k_work = F.normalize(k_work, p=2.0, dim=-1, eps=1e-6)
    if scale is None:
        scale = key_dim**-0.5

    decay_work = _as_bt_heads(log_decay, batch, tokens, value_heads, "log_decay").to(torch.float32)
    beta_work = _as_bt_heads(beta, batch, tokens, value_heads, "beta").to(torch.float32)
    value_work = v.to(torch.float32).reshape(batch, tokens, q_heads, repeat, value_dim)
    outputs: list[torch.Tensor] = []

    for token_idx in range(tokens):
        q_t = q_work[:, token_idx].unsqueeze(2)
        k_t = k_work[:, token_idx].unsqueeze(2)
        v_t = value_work[:, token_idx]
        decay_t = torch.exp(decay_work[:, token_idx]).reshape(batch, q_heads, repeat, 1, 1)
        beta_t = beta_work[:, token_idx].reshape(batch, q_heads, repeat, 1)

        decayed_state = state * decay_t
        projection = torch.matmul(decayed_state, k_t.unsqueeze(-1)).squeeze(-1)
        delta = (v_t - projection) * beta_t
        state = decayed_state + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
        output_t = torch.matmul(state, (q_t * scale).unsqueeze(-1)).squeeze(-1)
        outputs.append(output_t.reshape(batch, value_heads, value_dim))

    updated_state = state.reshape(batch, value_heads, value_dim, key_dim)
    _write_state_rows(state_pool, store_indices, updated_state, valid_store & valid_load)
    output = torch.stack(outputs, dim=1).to(v.dtype)
    output = torch.where(valid_load.reshape(batch, 1, 1, 1), output, torch.zeros_like(output))
    return output, state_pool


def split_packed_qkv(
    packed_qkv: torch.Tensor,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if packed_qkv.ndim != 2:
        raise ValueError(f"packed_qkv must have [B,width] shape, got {packed_qkv.shape}.")
    value_width = value_heads * value_dim
    qk_width = packed_qkv.shape[1] - value_width
    if qk_width <= 0 or qk_width % (2 * key_dim):
        raise ValueError(
            f"Cannot infer Q/K heads from packed width {packed_qkv.shape[1]}, HV={value_heads}, K={key_dim}, V={value_dim}."
        )
    key_heads = qk_width // (2 * key_dim)
    if value_heads % key_heads:
        raise ValueError(f"Value heads ({value_heads}) must be divisible by inferred Q/K heads ({key_heads}).")
    q_width = key_heads * key_dim
    q, k, v = packed_qkv.split((q_width, q_width, value_width), dim=-1)
    batch = packed_qkv.shape[0]
    return (
        q.reshape(batch, 1, key_heads, key_dim),
        k.reshape(batch, 1, key_heads, key_dim),
        v.reshape(batch, 1, value_heads, value_dim),
        key_heads,
    )


def packed_recurrent_decode(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    load_state_indices: torch.Tensor | None,
    store_state_indices: torch.Tensor | None,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, value_heads, value_dim, key_dim = state_pool.shape
    q, k, v, _ = split_packed_qkv(packed_qkv, value_heads, key_dim, value_dim)
    output, updated_pool = recurrent_decode_from_qkv(
        q,
        k,
        v,
        log_decay,
        beta,
        state_pool,
        load_state_indices,
        store_state_indices,
        scale,
        use_qk_l2norm,
    )
    return output[:, 0], updated_pool


def gating_parameters(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    log_decay = -torch.exp(A_log.to(torch.float32)) * F.softplus(x)
    beta = torch.sigmoid(b.to(torch.float32)).to(b.dtype)
    return log_decay, beta


def default_scale(key_dim: int) -> float:
    return 1.0 / math.sqrt(key_dim)
