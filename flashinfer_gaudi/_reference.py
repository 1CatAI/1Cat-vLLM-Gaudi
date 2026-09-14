# SPDX-License-Identifier: Apache-2.0
"""Numerically conservative PyTorch references for Gaudi inference ops."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vllm_gaudi.ops.gdn_recurrent_compound import gdn_recurrent_compound

_QWEN38_KEY_HEADS = 16
_QWEN38_VALUE_HEADS = 48
_QWEN38_DIM = 128
_QWEN38_HEAD_REPEAT = _QWEN38_VALUE_HEADS // _QWEN38_KEY_HEADS
_QWEN38_QK_WIDTH = 2 * _QWEN38_KEY_HEADS * _QWEN38_DIM
_QWEN38_PACKED_WIDTH = (_QWEN38_VALUE_HEADS + 2 * _QWEN38_KEY_HEADS) * _QWEN38_DIM
_QWEN38_TP2_KEY_HEADS = _QWEN38_KEY_HEADS // 2
_QWEN38_TP2_VALUE_HEADS = _QWEN38_VALUE_HEADS // 2
_QWEN38_TP2_HEAD_REPEAT = _QWEN38_TP2_VALUE_HEADS // _QWEN38_TP2_KEY_HEADS
_QWEN38_TP2_QK_WIDTH = 2 * _QWEN38_TP2_KEY_HEADS * _QWEN38_DIM
_QWEN38_TP2_PACKED_WIDTH = (_QWEN38_TP2_VALUE_HEADS + 2 * _QWEN38_TP2_KEY_HEADS) * _QWEN38_DIM


def _as_bt_heads(value: torch.Tensor, batch: int, tokens: int, heads: int, name: str) -> torch.Tensor:
    if value.numel() != batch * tokens * heads:
        raise ValueError(f"{name} must contain B*T*HV={batch * tokens * heads} values, got shape {tuple(value.shape)}.")
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
    reserved_padding_slot: int | None = None,
) -> None:
    candidate = updated_state.to(state_pool.dtype)
    if reserved_padding_slot is not None:
        # DFlash compact state reserves slot zero. All padding rows can then be
        # coalesced into a single batched write without any live-slot alias.
        padding = torch.full_like(store_indices, reserved_padding_slot)
        safe_destination = torch.where(valid_store, store_indices, padding)
        padding_value = state_pool.narrow(0, reserved_padding_slot, 1)
        rows = torch.where(valid_store.reshape(-1, 1, 1, 1), candidate, padding_value)
        state_pool.index_copy_(0, safe_destination, rows)
        return

    # Generic indexed APIs permit slot zero to be live. Preserve exact
    # sequential semantics for invalid rows rather than aliasing a live slot.
    for batch_idx in range(updated_state.shape[0]):
        destination = store_indices.narrow(0, batch_idx, 1)
        valid = valid_store.narrow(0, batch_idx, 1)
        safe_destination = torch.where(valid, destination, torch.zeros_like(destination))
        previous = state_pool.index_select(0, safe_destination)
        row = torch.where(valid.reshape(1, 1, 1, 1), candidate.narrow(0, batch_idx, 1), previous)
        state_pool.index_copy_(0, safe_destination, row)


def _l2_normalize_rsqrt(value: torch.Tensor) -> torch.Tensor:
    """Use the additive epsilon of the GDN kernel contract.

    This is not ``F.normalize``: clamping the norm changes small Q/K
    vectors and can make decode disagree with prefill and target-only.
    Callers select the compute dtype before normalization.
    """
    norm_squared = torch.sum(value * value, dim=-1, keepdim=True)
    return value * torch.rsqrt(norm_squared + 1e-6)


def _direct_single_token_decode_core(
    q_work: torch.Tensor,
    k_work: torch.Tensor,
    value_work: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update contiguous state after Q/K conversion and normalization."""
    batch, q_heads, key_dim = q_work.shape
    value_heads, value_dim = value_work.shape[1:]
    repeat = value_heads // q_heads
    output_dtype = value_work.dtype
    k_grouped = k_work.unsqueeze(2)

    state = state_pool.to(torch.float32).reshape(batch, q_heads, repeat, value_dim, key_dim)
    decay = torch.exp(_as_bt_heads(log_decay, batch, 1, value_heads, "log_decay")[:, 0].to(torch.float32))
    decay = decay.reshape(batch, q_heads, repeat, 1, 1)
    beta_work = _as_bt_heads(beta, batch, 1, value_heads, "beta")[:, 0].to(torch.float32)
    beta_work = beta_work.reshape(batch, q_heads, repeat, 1)
    value_work = value_work.to(torch.float32).reshape(batch, q_heads, repeat, value_dim)

    # Two width-one projections are faster than a single width-two MME on
    # Gaudi2 for this state shape. Keep the recurrent update order explicit so
    # the graph compiler can fuse the surrounding decay and rank-one update.
    decayed_state = state * decay
    projection = torch.matmul(decayed_state, k_grouped.unsqueeze(-1)).squeeze(-1)
    delta = (value_work - projection) * beta_work
    updated_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k_grouped.unsqueeze(-2))
    output = torch.matmul(updated_state, (q_work.unsqueeze(2) * scale).unsqueeze(-1)).squeeze(-1)

    state_pool.copy_(updated_state.reshape_as(state_pool).to(state_pool.dtype))
    output = output.reshape(batch, 1, value_heads, value_dim).to(output_dtype)
    return output, state_pool


def _direct_single_token_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Specialize the contiguous decode path without gather/scatter scaffolding."""
    q_heads = q.shape[2]
    qk_work = torch.cat((q[:, 0], k[:, 0]), dim=1).to(torch.float32)
    if use_qk_l2norm:
        qk_work = _l2_normalize_rsqrt(qk_work)
    q_work, k_work = qk_work.split(q_heads, dim=1)
    return _direct_single_token_decode_core(
        q_work,
        k_work,
        v[:, 0],
        log_decay,
        beta,
        state_pool,
        scale,
    )


def _direct_single_token_packed_decode(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Specialize packed decode so Q and K share one normalization launch."""
    if packed_qkv.ndim != 2:
        raise ValueError(f"packed_qkv must have [B,width] shape, got {packed_qkv.shape}.")
    if state_pool.ndim != 4:
        raise ValueError(f"state must have [B,HV,V,K] shape, got {state_pool.shape}.")
    batch = packed_qkv.shape[0]
    slots, value_heads, value_dim, key_dim = state_pool.shape
    if slots != batch:
        raise ValueError(f"Direct state requires one contiguous row per request; got slots={slots}, B={batch}.")
    value_width = value_heads * value_dim
    qk_width = packed_qkv.shape[1] - value_width
    if qk_width <= 0 or qk_width % (2 * key_dim):
        raise ValueError(f"Cannot infer Q/K heads from packed width {packed_qkv.shape[1]}, "
                         f"HV={value_heads}, K={key_dim}, V={value_dim}.")
    q_heads = qk_width // (2 * key_dim)
    if value_heads % q_heads:
        raise ValueError(f"Value heads ({value_heads}) must be divisible by inferred Q/K heads ({q_heads}).")

    qk_work = packed_qkv[:, :qk_width].reshape(batch, 2 * q_heads, key_dim).to(torch.float32)
    if use_qk_l2norm:
        qk_work = _l2_normalize_rsqrt(qk_work)
    q_work, k_work = qk_work.split(q_heads, dim=1)
    value_work = packed_qkv[:, qk_width:].reshape(batch, value_heads, value_dim)
    return _direct_single_token_decode_core(
        q_work,
        k_work,
        value_work,
        log_decay,
        beta,
        state_pool,
        key_dim**-0.5 if scale is None else scale,
    )


def _direct_qwen38_single_token_packed_decode(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    inplace_state: bool = True,
    recurrent_compound: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the production Qwen shape static for the Gaudi graph compiler."""
    batch = packed_qkv.shape[0]
    qk_work = packed_qkv[:, :_QWEN38_QK_WIDTH].reshape(
        batch,
        2 * _QWEN38_KEY_HEADS,
        _QWEN38_DIM,
    ).to(torch.float32)
    if use_qk_l2norm:
        qk_work = _l2_normalize_rsqrt(qk_work)
    q_work, k_work = qk_work.split(_QWEN38_KEY_HEADS, dim=1)
    k_grouped = k_work.unsqueeze(2)
    value_work = packed_qkv[:, _QWEN38_QK_WIDTH:].reshape(
        batch,
        _QWEN38_KEY_HEADS,
        _QWEN38_HEAD_REPEAT,
        _QWEN38_DIM,
    ).to(torch.float32)
    state = state_pool.reshape(
        batch,
        _QWEN38_KEY_HEADS,
        _QWEN38_HEAD_REPEAT,
        _QWEN38_DIM,
        _QWEN38_DIM,
    )
    decay = torch.exp(log_decay.reshape(
        batch,
        _QWEN38_KEY_HEADS,
        _QWEN38_HEAD_REPEAT,
    ).to(torch.float32)).reshape(
        batch,
        _QWEN38_KEY_HEADS,
        _QWEN38_HEAD_REPEAT,
        1,
        1,
    )
    beta_work = beta.reshape(
        batch,
        _QWEN38_KEY_HEADS,
        _QWEN38_HEAD_REPEAT,
    ).to(torch.float32).unsqueeze(-1)

    if recurrent_compound:
        updated_state, output = gdn_recurrent_compound(
            state,
            decay,
            k_grouped,
            q_work.unsqueeze(2) * scale,
            value_work,
            beta_work,
        )
    else:
        decayed_state = state * decay
        projection = torch.matmul(decayed_state, k_grouped.unsqueeze(-1)).squeeze(-1)
        delta = (value_work - projection) * beta_work
        updated_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k_grouped.unsqueeze(-2))
        output = torch.matmul(updated_state, (q_work.unsqueeze(2) * scale).unsqueeze(-1)).squeeze(-1)
    next_state = updated_state.reshape_as(state_pool)
    if inplace_state:
        state_pool.copy_(next_state)
        next_state = state_pool

    output = output.reshape(batch, _QWEN38_VALUE_HEADS, _QWEN38_DIM).to(packed_qkv.dtype)
    return output, next_state


def _direct_qwen38_tp2_single_token_packed_decode(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    inplace_state: bool = True,
    direct_state_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static per-rank Qwen3.8 TP2 geometry for the Gaudi compiler."""
    batch = packed_qkv.shape[0]
    qk_work = packed_qkv[:, :_QWEN38_TP2_QK_WIDTH].reshape(
        batch,
        2 * _QWEN38_TP2_KEY_HEADS,
        _QWEN38_DIM,
    ).to(torch.float32)
    if use_qk_l2norm:
        qk_work = _l2_normalize_rsqrt(qk_work)
    q_work, k_work = qk_work.split(_QWEN38_TP2_KEY_HEADS, dim=1)
    k_grouped = k_work.unsqueeze(2)
    value_work = packed_qkv[:, _QWEN38_TP2_QK_WIDTH:].reshape(
        batch,
        _QWEN38_TP2_KEY_HEADS,
        _QWEN38_TP2_HEAD_REPEAT,
        _QWEN38_DIM,
    ).to(torch.float32)
    state = state_pool.reshape(
        batch,
        _QWEN38_TP2_KEY_HEADS,
        _QWEN38_TP2_HEAD_REPEAT,
        _QWEN38_DIM,
        _QWEN38_DIM,
    )
    decay = torch.exp(log_decay.reshape(
        batch,
        _QWEN38_TP2_KEY_HEADS,
        _QWEN38_TP2_HEAD_REPEAT,
    ).to(torch.float32)).reshape(
        batch,
        _QWEN38_TP2_KEY_HEADS,
        _QWEN38_TP2_HEAD_REPEAT,
        1,
        1,
    )
    beta_work = beta.reshape(
        batch,
        _QWEN38_TP2_KEY_HEADS,
        _QWEN38_TP2_HEAD_REPEAT,
    ).to(torch.float32).unsqueeze(-1)

    decayed_state = state * decay
    projection = torch.matmul(decayed_state, k_grouped.unsqueeze(-1)).squeeze(-1)
    delta = (value_work - projection) * beta_work
    if direct_state_update:
        if not inplace_state or batch != 1:
            raise ValueError("Direct TP2 state updates require one active row and in-place state semantics")
        next_state = torch.ops.custom_op.gdn_state_update(decayed_state, delta.unsqueeze(-1), k_grouped.unsqueeze(-2),
                                                          state_pool)
        updated_state = next_state.reshape_as(state)
    else:
        updated_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k_grouped.unsqueeze(-2))
    output = torch.matmul(updated_state, (q_work.unsqueeze(2) * scale).unsqueeze(-1)).squeeze(-1)
    next_state = updated_state.reshape_as(state_pool)
    if inplace_state:
        state_pool.copy_(next_state)
        next_state = state_pool

    output = output.reshape(batch, _QWEN38_TP2_VALUE_HEADS, _QWEN38_DIM).to(packed_qkv.dtype)
    return output, next_state


def qwen38_fused_decode_step_direct(
    packed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    ssm_state: torch.Tensor,
    scale: float,
    inplace_state: bool = True,
    direct_state_update: bool = False,
    recurrent_compound: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the static Qwen3.8 direct-state decode composition.

    MME continues to execute the two recurrent projections.  This function
    removes generic convolution/indexing scaffolding while retaining the
    proven recurrent-state update order used by grouped regional graphs.
    """
    gate_input = a.to(torch.float32) + dt_bias.to(torch.float32)
    softplus = torch.where(gate_input <= 20.0, torch.log1p(torch.exp(gate_input)), gate_input)
    log_decay = -torch.exp(A_log.to(torch.float32)) * softplus
    beta = torch.sigmoid(b.to(torch.float32)).to(b.dtype)

    weights = conv_weight.to(torch.float32)
    convolved = conv_state[:, 0, :] * weights[:, 0]
    convolved = convolved + conv_state[:, 1, :] * weights[:, 1]
    convolved = convolved + conv_state[:, 2, :] * weights[:, 2]
    convolved = convolved + packed_qkv * weights[:, 3]
    if conv_bias is not None:
        convolved = convolved + conv_bias.to(torch.float32)
    convolved = convolved.to(torch.bfloat16)
    convolved = F.silu(convolved)
    conv_state.copy_(torch.cat((conv_state[:, 1:, :], packed_qkv.unsqueeze(1)), dim=1))

    if (packed_qkv.shape[1] == _QWEN38_TP2_PACKED_WIDTH and ssm_state.shape[1] == _QWEN38_TP2_VALUE_HEADS):
        output, next_state = _direct_qwen38_tp2_single_token_packed_decode(
            convolved,
            log_decay,
            beta,
            ssm_state,
            scale,
            True,
            inplace_state,
            direct_state_update,
        )
    else:
        output, next_state = _direct_qwen38_single_token_packed_decode(
            convolved,
            log_decay,
            beta,
            ssm_state,
            scale,
            True,
            inplace_state,
            recurrent_compound,
        )
    return output, conv_state, next_state


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
    direct_state: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    update_final_state: bool = True,
    token_mask: torch.Tensor | None = None,
    reserved_padding_slot: int | None = None,
    assume_valid_indices: bool = False,
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
        raise ValueError("State shape does not match Q/K/V: "
                         f"state={tuple(state_pool.shape)}, q={tuple(q.shape)}, v={tuple(v.shape)}.")
    if value_heads % q_heads:
        raise ValueError(f"Value heads ({value_heads}) must be divisible by Q/K heads ({q_heads}).")

    if direct_state and slots != batch:
        raise ValueError(f"Direct state requires one contiguous row per request; got slots={slots}, B={batch}.")

    if token_mask is not None:
        if tuple(token_mask.shape) != (batch, tokens):
            raise ValueError(f"token_mask must have shape [{batch},{tokens}], got {token_mask.shape}.")
        token_mask = token_mask.to(device=state_pool.device, dtype=torch.bool)

    if direct_state:
        if (tokens == 1 and intermediate_states_buffer is None and ssm_state_indices is None and update_final_state
                and token_mask is None):
            if scale is None:
                scale = key_dim**-0.5
            return _direct_single_token_decode(q, k, v, log_decay, beta, state_pool, scale, use_qk_l2norm)
        state = state_pool.to(torch.float32)
        load_indices = torch.arange(batch, dtype=torch.long, device=state_pool.device)
        store_indices = load_indices
        valid_load = torch.ones(batch, dtype=torch.bool, device=state_pool.device)
        valid_store = valid_load
    else:
        load_indices = _canonical_indices(load_state_indices, batch, slots, state_pool.device)
        store_indices = _canonical_indices(store_state_indices, batch, slots, state_pool.device)
        if assume_valid_indices:
            valid_load = torch.ones(batch, dtype=torch.bool, device=state_pool.device)
            valid_store = valid_load
            state = state_pool.index_select(0, load_indices).to(torch.float32)
        else:
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
        # Keep verification numerically aligned with the promoted one-token
        # Qwen3.8 decode path.  A different normalization spelling here is
        # enough to change a near-tied target argmax after several accepted
        # blocks, which breaks greedy speculative decoding's lossless
        # contract even when checkpoint rollback itself is correct.
        qk_work = torch.cat((q_work, k_work), dim=2)
        qk_work = _l2_normalize_rsqrt(qk_work)
        q_work, k_work = qk_work.split(q_heads, dim=2)
    if scale is None:
        scale = key_dim**-0.5

    decay_work = _as_bt_heads(log_decay, batch, tokens, value_heads, "log_decay").to(torch.float32)
    beta_work = _as_bt_heads(beta, batch, tokens, value_heads, "beta").to(torch.float32)
    value_work = v.to(torch.float32).reshape(batch, tokens, q_heads, repeat, value_dim)
    if intermediate_states_buffer is not None and (
            intermediate_states_buffer.ndim != 5 or intermediate_states_buffer.shape[0] < batch
            or intermediate_states_buffer.shape[1] < tokens
            or tuple(intermediate_states_buffer.shape[2:]) != (value_heads, value_dim, key_dim)):
        raise ValueError("intermediate_states_buffer must have at least "
                         f"[{batch},{tokens},{value_heads},{value_dim},{key_dim}], got "
                         f"{tuple(intermediate_states_buffer.shape)}.")
    if ssm_state_indices is not None and tuple(ssm_state_indices.shape) != (batch, tokens):
        raise ValueError(f"ssm_state_indices must have shape [{batch},{tokens}], got {ssm_state_indices.shape}.")
    # Prepare independent token work together for the single-request MTP
    # bucket. Keep the state recurrence and checkpoint update order intact.
    prepare_tokens = assume_valid_indices and token_mask is None and batch == 1 and tokens == 8
    if prepare_tokens:
        decay_work = torch.exp(decay_work)
        q_work = q_work * scale
    outputs: list[torch.Tensor] = []

    for token_idx in range(tokens):
        q_t = q_work[:, token_idx].unsqueeze(2)
        k_t = k_work[:, token_idx].unsqueeze(2)
        v_t = value_work[:, token_idx]
        decay_t = decay_work[:, token_idx] if prepare_tokens else torch.exp(decay_work[:, token_idx])
        decay_t = decay_t.reshape(batch, q_heads, repeat, 1, 1)
        beta_t = beta_work[:, token_idx].reshape(batch, q_heads, repeat, 1)

        decayed_state = state * decay_t
        projection = torch.matmul(decayed_state, k_t.unsqueeze(-1)).squeeze(-1)
        delta = (v_t - projection) * beta_t
        # Match _direct_single_token_decode_core exactly.  In particular,
        # addcmul has a distinct HPU lowering from a separately materialized
        # multiply followed by add.
        candidate_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k_t.unsqueeze(-2))
        active_t = valid_load
        if token_mask is not None:
            active_t = active_t & token_mask[:, token_idx]
        if assume_valid_indices and token_mask is None:
            state = candidate_state
        else:
            state = torch.where(active_t.reshape(batch, 1, 1, 1, 1), candidate_state, state)
        output_q = q_t if prepare_tokens else q_t * scale
        output_t = torch.matmul(state, output_q.unsqueeze(-1)).squeeze(-1)
        output_t = output_t.reshape(batch, value_heads, value_dim)
        if assume_valid_indices and token_mask is None:
            outputs.append(output_t)
        else:
            outputs.append(torch.where(active_t.reshape(batch, 1, 1), output_t, torch.zeros_like(output_t)))

        state_t = state.reshape(batch, value_heads, value_dim, key_dim)
        if intermediate_states_buffer is not None:
            target = intermediate_states_buffer[:batch, token_idx]
            target.copy_(torch.where(
                active_t.reshape(batch, 1, 1, 1),
                state_t.to(target.dtype),
                target,
            ))
        if ssm_state_indices is not None:
            token_store_indices = ssm_state_indices[:, token_idx].to(device=state_pool.device, dtype=torch.long)
            if assume_valid_indices:
                state_pool.index_copy_(0, token_store_indices, state_t.to(state_pool.dtype))
            else:
                _validate_cpu_store_indices(token_store_indices, slots)
                _, valid_token_store = _safe_indices(token_store_indices, slots)
                _write_state_rows(
                    state_pool,
                    token_store_indices,
                    state_t,
                    active_t & valid_token_store,
                    reserved_padding_slot,
                )

    updated_state = state.reshape(batch, value_heads, value_dim, key_dim)
    if not update_final_state or ssm_state_indices is not None:
        pass
    elif direct_state:
        state_pool.copy_(updated_state.to(state_pool.dtype))
    else:
        active_rows = valid_load
        if token_mask is not None:
            active_rows = active_rows & torch.any(token_mask, dim=1)
        _write_state_rows(
            state_pool,
            store_indices,
            updated_state,
            valid_store & active_rows,
            reserved_padding_slot,
        )
    output = torch.stack(outputs, dim=1).to(v.dtype)
    if not assume_valid_indices:
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
        raise ValueError(f"Cannot infer Q/K heads from packed width {packed_qkv.shape[1]}, "
                         f"HV={value_heads}, K={key_dim}, V={value_dim}.")
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
    direct_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if direct_state:
        if (packed_qkv.ndim == 2 and state_pool.ndim == 4 and state_pool.dtype == torch.float32
                and packed_qkv.shape[1] == _QWEN38_PACKED_WIDTH
                and tuple(state_pool.shape[1:]) == (_QWEN38_VALUE_HEADS, _QWEN38_DIM, _QWEN38_DIM)):
            if state_pool.shape[0] != packed_qkv.shape[0]:
                raise ValueError("Direct state requires one contiguous row per request; "
                                 f"got slots={state_pool.shape[0]}, B={packed_qkv.shape[0]}.")
            return _direct_qwen38_single_token_packed_decode(
                packed_qkv,
                log_decay,
                beta,
                state_pool,
                _QWEN38_DIM**-0.5 if scale is None else scale,
                use_qk_l2norm,
            )
        output, updated_pool = _direct_single_token_packed_decode(packed_qkv, log_decay, beta, state_pool, scale,
                                                                  use_qk_l2norm)
        return output[:, 0], updated_pool
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
        direct_state,
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
