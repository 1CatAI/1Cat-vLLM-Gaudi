# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-compatible Gated Delta Rule decode APIs for Intel Gaudi."""

from __future__ import annotations

import warnings

import torch

from flashinfer_gaudi._dispatch import BackendUnavailableError as BackendUnavailableError
from flashinfer_gaudi._dispatch import require_reference_allowed
from flashinfer_gaudi._config import get_backend_policy, mtp_prepared_enabled
from flashinfer_gaudi._native import (
    public_mtp_gdn_op,
    public_mtp_prepared_op,
)
from flashinfer_gaudi._reference import (
    _l2_normalize_rsqrt,
    gating_parameters,
    packed_recurrent_decode,
    recurrent_decode_from_qkv,
    split_packed_qkv,
)
from flashinfer_gaudi._tactics import public_mtp_gdn_auto_promoted

_QWEN38_MTP_TOKENS = 8
_QWEN38_MTP_KEY_HEADS = 16
_QWEN38_MTP_VALUE_HEADS = 48
_QWEN38_MTP_DIM = 128
_QWEN38_MTP_MIN_AUTO_BATCH = 2
_QWEN38_MTP_PACKED_WIDTH = (2 * _QWEN38_MTP_KEY_HEADS + _QWEN38_MTP_VALUE_HEADS) * _QWEN38_MTP_DIM
_QWEN38_MTP_MAX_BATCH = 16


def _device_is_hpu(tensor: torch.Tensor) -> bool:
    return tensor.device.type in ("hpu", "privateuseone")


def _call_native_packed(
    op,
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    require_reference_allowed("gdn_decode_packed legacy torch prologue")
    qualified_name = getattr(op, "_qualified_op_name", "")
    legacy_f32_contract = qualified_name in (
        "custom_op::custom_gdn_packed_decode_f32_gaudi2",
        "flashinfer_gaudi_bridge::gdn_decode_packed",
    )
    decay = torch.exp(log_decay.reshape(packed_qkv.shape[0], -1)).contiguous()
    if legacy_f32_contract:
        result = op(
            state_pool,
            packed_qkv.to(torch.float32).contiguous(),
            decay,
            beta.reshape(packed_qkv.shape[0], -1).to(torch.float32).contiguous(),
            state_indices.reshape(-1).to(torch.int32),
        )
    else:
        # The stable public GUID consumes packed BF16 Q/K/V and beta directly
        # while retaining FP32 recurrent state.
        result = op(
            state_pool,
            packed_qkv.contiguous(),
            decay,
            beta.reshape(packed_qkv.shape[0], -1).contiguous(),
            state_indices.reshape(-1).to(torch.int32),
        )
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        # Compatibility with the version-locked bridge and the first public
        # prototype, which returned an unused state dependency tensor.
        return result[1].to(packed_qkv.dtype)
    raise RuntimeError("Gaudi packed GDN op must return output or (state_dependency, output).")


def gated_delta_rule_decode_packed(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    load_state_indices: torch.Tensor | None = None,
    store_state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the vLLM packed-QKV GDN fast path.

    Backend selection is completed before an implementation is called. A
    failed state-mutating native call is deliberately not retried.
    """
    require_reference_allowed("gated_delta_rule_decode_packed")
    return packed_recurrent_decode(
        packed_qkv,
        log_decay,
        beta,
        state_pool,
        load_state_indices,
        store_state_indices,
        scale,
        use_qk_l2norm,
    )


def gated_delta_rule_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GDN decode using the FlashInfer VK/K-last state convention."""
    require_reference_allowed("gated_delta_rule_decode_pretranspose")
    if state is None:
        if initial_state is None:
            state = torch.zeros(
                q.shape[0],
                v.shape[2],
                v.shape[3],
                q.shape[3],
                dtype=torch.float32,
                device=q.device,
            )
        else:
            state = initial_state
    elif initial_state is not None and initial_state is not state:
        raise ValueError("Pass either state or initial_state, not two different state tensors.")

    log_decay, beta = gating_parameters(A_log, a, dt_bias, b)
    result, updated_state = recurrent_decode_from_qkv(
        q,
        k,
        v,
        log_decay,
        beta,
        state,
        initial_state_indices,
        output_state_indices,
        scale,
        use_qk_l2norm,
    )
    if output is not None:
        if output.shape != result.shape:
            raise ValueError(f"output has shape {output.shape}, expected {result.shape}.")
        output.copy_(result)
        result = output
    return result, updated_state


def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FlashInfer K-major decode API on a ``[B,HV,K,V]`` state."""
    require_reference_allowed("gated_delta_rule_decode")
    if state.ndim != 4:
        raise ValueError(f"state must have [B,HV,K,V] shape, got {state.shape}.")
    batch, _, _, key_dim = q.shape
    _, _, value_heads, value_dim = v.shape
    expected = (batch, value_heads, key_dim, value_dim)
    if tuple(state.shape) != expected:
        raise ValueError(f"K-major state has shape {tuple(state.shape)}, expected {expected}.")

    # The portable recurrence is implemented in VK/K-last layout. Keep this
    # compatibility path explicit so the packed vLLM fast path never pays a
    # transpose, while callers of FlashInfer's K-major API receive its exact
    # state contract.
    state_vk = state.transpose(-1, -2).contiguous()
    log_decay, beta = gating_parameters(A_log, a, dt_bias, b)
    result, _ = recurrent_decode_from_qkv(
        q,
        k,
        v,
        log_decay,
        beta,
        state_vk,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )
    state.copy_(state_vk.transpose(-1, -2))
    if output is not None:
        if output.shape != result.shape:
            raise ValueError(f"output has shape {output.shape}, expected {result.shape}.")
        output.copy_(result)
        result = output
    return result, state


def gated_delta_rule_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    intermediate_states_buffer: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    disable_state_update: bool | None = None,
    use_qk_l2norm: bool = True,
    output_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable implementation of FlashInfer's pooled multi-token GDN API."""
    require_reference_allowed("gated_delta_rule_mtp")
    if disable_state_update is None:
        warnings.warn(
            "disable_state_update currently defaults to True for FlashInfer 0.6.18 compatibility; "
            "pass it explicitly because the upstream default changes in 0.7.0.",
            FutureWarning,
            stacklevel=2,
        )
        disable_state_update = True
    if intermediate_states_buffer is not None and ssm_state_indices is not None:
        raise ValueError("ssm_state_indices and intermediate_states_buffer are mutually exclusive.")
    if ssm_state_indices is not None:
        if q.shape[1] < 2:
            raise ValueError("ssm_state_indices requires T >= 2.")
        if disable_state_update:
            raise ValueError("ssm_state_indices requires disable_state_update=False.")
    if output_state_indices is None:
        output_state_indices = initial_state_indices

    log_decay, beta = gating_parameters(A_log, a, dt_bias, b)
    result, updated_state = recurrent_decode_from_qkv(
        q,
        k,
        v,
        log_decay,
        beta,
        initial_state,
        initial_state_indices,
        output_state_indices,
        scale,
        use_qk_l2norm,
        False,
        intermediate_states_buffer,
        ssm_state_indices,
        not disable_state_update,
    )
    if output is not None:
        if output.shape != result.shape:
            raise ValueError(f"output has shape {output.shape}, expected {result.shape}.")
        output.copy_(result)
        result = output
    return result, updated_state


def gated_delta_rule_mtp_rollback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    assume_full_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a DFlash2 GDN lattice from its accepted checkpoint.

    This is a Gaudi extension to FlashInfer's public MTP contract. It consumes
    precomputed gating tensors, selects checkpoint ``accepted - 1`` for each
    request, and writes every live verification-token state back to the fixed
    checkpoint lattice. ``token_mask`` keeps bucket padding from decaying or
    overwriting recurrent state.
    """
    if q.ndim != 4:
        raise ValueError(f"q must have [B,T,H,K] shape, got {q.shape}.")
    batch, tokens = q.shape[:2]
    if tuple(state_indices.shape) != (batch, tokens):
        raise ValueError(f"state_indices must have shape [{batch},{tokens}], got {state_indices.shape}.")
    if num_accepted_tokens.numel() != batch:
        raise ValueError(f"num_accepted_tokens must contain {batch} entries, got {num_accepted_tokens.shape}.")

    accepted_offsets = num_accepted_tokens.to(device=state_indices.device, dtype=torch.long).clamp(
        min=1,
        max=tokens,
    ) - 1
    initial_state_indices = state_indices.gather(1, accepted_offsets.unsqueeze(1)).squeeze(1)
    return recurrent_decode_from_qkv(
        q,
        k,
        v,
        log_decay,
        beta,
        state_pool,
        load_state_indices=initial_state_indices,
        store_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        ssm_state_indices=state_indices,
        update_final_state=False,
        token_mask=token_mask,
        reserved_padding_slot=0,
        assume_valid_indices=assume_full_query,
    )


def _gated_delta_rule_mtp_packed_reference(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_lengths: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    assume_full_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the exact portable packed MTP recurrence without dispatch."""
    batch, tokens, _ = packed_qkv.shape
    _, value_heads, value_dim, key_dim = state_pool.shape
    q, k, v, _ = split_packed_qkv(
        packed_qkv.reshape(batch * tokens, -1),
        value_heads,
        key_dim,
        value_dim,
    )
    q = q.reshape(batch, tokens, q.shape[2], key_dim)
    k = k.reshape_as(q)
    v = v.reshape(batch, tokens, value_heads, value_dim)
    token_mask = None if assume_full_query else (torch.arange(tokens, device=packed_qkv.device).unsqueeze(0)
                                                 < query_lengths.to(device=packed_qkv.device, ).unsqueeze(1))
    return gated_delta_rule_mtp_rollback(
        q,
        k,
        v,
        state_pool,
        state_indices,
        num_accepted_tokens,
        log_decay,
        beta,
        token_mask,
        scale,
        use_qk_l2norm,
        assume_full_query,
    )


def _gated_delta_rule_mtp_prepared(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    assume_distinct_checkpoints: bool = False,
    checkpoint_start: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-native, full B1/T8 verification; caller proves shape and validity.

    The core is functional: it returns all checkpoints instead of mutating
    input storage. Keep slot selection and ownership in the compiled graph.
    This is opt-in until full-model numerical and latency qualification.
    """
    native = public_mtp_prepared_op()
    if native is None:
        raise BackendUnavailableError("The prepared MTP kernel is not loaded; rebuild the native extension.")
    batch, tokens, _ = packed_qkv.shape
    if (checkpoint_start is not None and (not assume_distinct_checkpoints or checkpoint_start < 1
                                          or checkpoint_start + batch * tokens > state_pool.shape[0])):
        raise ValueError("Direct checkpoints require a valid, distinct, caller-proven contiguous destination.")
    offsets = num_accepted_tokens.long().clamp(1, tokens) - 1
    loads = state_indices.gather(1, offsets.unsqueeze(1)).squeeze(1).long()
    initial_state = state_pool.index_select(0, loads)
    qk = packed_qkv[..., :4096].float().reshape(batch, tokens, 32, 128)
    qk = _l2_normalize_rsqrt(qk)
    q = qk[..., :16, :] * (128**-0.5)
    k = qk[..., 16:, :]
    prepared = torch.cat((q.flatten(2), k.flatten(2), packed_qkv[..., 4096:].float()), dim=-1)
    output, checkpoints = native(initial_state, prepared, torch.exp(log_decay), beta.float())
    if checkpoint_start is not None:
        # Keep the narrow inside the compiled graph: a separately passed
        # aliased view need not lower to the same destination-write recipe.
        state_pool.narrow(0, checkpoint_start, batch * tokens).copy_(checkpoints.reshape(-1, 48, 128, 128))
    elif assume_distinct_checkpoints:
        state_pool.index_copy_(0, state_indices.reshape(-1).long(), checkpoints.reshape(-1, 48, 128, 128))
    else:
        # Repeated token slots require ordered, last-token-wins writes.
        # A parallel flattened index_copy has no such duplicate-index contract.
        for token in range(tokens):
            state_pool.index_copy_(0, state_indices[:, token].long(), checkpoints[:, token])
    return output.to(packed_qkv.dtype), state_pool


def gated_delta_rule_mtp_packed(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_lengths: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    assume_full_query: bool = False,
    assume_distinct_checkpoints: bool = False,
    checkpoint_start: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed DFlash2 GDN verification with rollback checkpoints.

    The public Gaudi2 kernel specializes the Qwen3.8 ``B<=16, T=8, H=16,
    HV=48, K=V=128`` contract. CPU and unsupported auto-policy shapes use the
    exact portable recurrence. The MTP kernel has an independent offline
    promotion gate because it mutates the full rollback lattice.

    ``assume_distinct_checkpoints`` permits one coalesced checkpoint write
    in the experimental prepared route. Only set it when the caller proves
    token slots are distinct; otherwise checkpoint writes remain ordered.
    """
    if packed_qkv.ndim != 3:
        raise ValueError(f"packed_qkv must have [B,T,width] shape, got {packed_qkv.shape}.")
    if state_pool.ndim != 4:
        raise ValueError(f"state_pool must have [slots,HV,V,K] shape, got {state_pool.shape}.")
    packed_qkv = packed_qkv.contiguous()
    log_decay = log_decay.contiguous()
    beta = beta.contiguous()
    state_indices = state_indices.contiguous()
    num_accepted_tokens = num_accepted_tokens.contiguous()
    query_lengths = query_lengths.contiguous()
    batch, tokens, _ = packed_qkv.shape
    if tuple(state_indices.shape) != (batch, tokens):
        raise ValueError(f"state_indices must have shape [{batch},{tokens}], got {state_indices.shape}.")
    if num_accepted_tokens.numel() != batch or query_lengths.numel() != batch:
        raise ValueError("num_accepted_tokens and query_lengths must each contain one entry per request.")

    policy = get_backend_policy()
    is_hpu = _device_is_hpu(packed_qkv)
    expected_scale = state_pool.shape[-1]**-0.5
    native_supported = (is_hpu and packed_qkv.dtype == torch.bfloat16 and 0 < batch <= _QWEN38_MTP_MAX_BATCH
                        and tokens == _QWEN38_MTP_TOKENS and packed_qkv.shape[2] == _QWEN38_MTP_PACKED_WIDTH
                        and state_pool.dtype == torch.float32 and state_pool.shape[0] > 0
                        and tuple(state_pool.shape[1:]) == (_QWEN38_MTP_VALUE_HEADS, _QWEN38_MTP_DIM, _QWEN38_MTP_DIM)
                        and log_decay.dtype == torch.float32
                        and tuple(log_decay.shape) == (batch, tokens, state_pool.shape[1])
                        and beta.dtype == torch.bfloat16 and beta.shape == log_decay.shape
                        and state_indices.dtype == torch.int32 and num_accepted_tokens.dtype == torch.int32
                        and query_lengths.dtype == torch.int32 and use_qk_l2norm
                        and (scale is None or abs(scale - expected_scale) <= 1e-12))
    if policy in ("bridge", "native"):
        raise BackendUnavailableError(f"The {policy} backend does not provide complete DFlash2 MTP GDN verification.")
    if (policy != "pytorch" and mtp_prepared_enabled() and torch.compiler.is_compiling() and native_supported
            and batch == 1 and assume_full_query):
        return _gated_delta_rule_mtp_prepared(
            packed_qkv,
            log_decay,
            beta,
            state_pool,
            state_indices,
            num_accepted_tokens,
            assume_distinct_checkpoints,
            checkpoint_start,
        )
    # The TPC kernel only amortizes its launch/setup cost once two or more
    # eager requests share the verification step. Inside a compiled target
    # model, its mutable custom-op boundary prevents the graph backend from
    # fusing the recurrence with surrounding work and loses to the reference
    # graph. Explicit ``public`` remains available for kernel qualification.
    use_public = policy == "public" or (policy == "auto" and not torch.compiler.is_compiling()
                                        and batch >= _QWEN38_MTP_MIN_AUTO_BATCH and public_mtp_gdn_auto_promoted())
    if use_public and native_supported:
        native = public_mtp_gdn_op()
        if native is not None:
            result = native(
                state_pool,
                packed_qkv,
                torch.exp(log_decay).contiguous(),
                beta,
                state_indices,
                num_accepted_tokens,
                query_lengths,
            )
            return result, state_pool
    if policy == "public":
        raise BackendUnavailableError("The public native MTP GDN backend is unavailable for this runtime or shape.")

    require_reference_allowed("gated_delta_rule_mtp_packed")
    return _gated_delta_rule_mtp_packed_reference(
        packed_qkv,
        log_decay,
        beta,
        state_pool,
        state_indices,
        num_accepted_tokens,
        query_lengths,
        scale,
        use_qk_l2norm,
        assume_full_query,
    )
