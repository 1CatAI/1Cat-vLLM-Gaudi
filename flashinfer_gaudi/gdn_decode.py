# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-compatible Gated Delta Rule decode APIs for Intel Gaudi."""

from __future__ import annotations

import warnings

import torch

from flashinfer_gaudi._config import bridge_auto_enabled, get_backend_policy
from flashinfer_gaudi._native import bridge_packed_gdn_op, public_packed_gdn_op
from flashinfer_gaudi._reference import gating_parameters, packed_recurrent_decode, recurrent_decode_from_qkv
from flashinfer_gaudi._tactics import public_gdn_auto_promoted


class BackendUnavailableError(RuntimeError):
    """Raised when a forced backend cannot execute the requested shape."""


def _device_is_hpu(tensor: torch.Tensor) -> bool:
    return tensor.device.type in ("hpu", "privateuseone")


def _native_packed_supported(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    load_state_indices: torch.Tensor | None,
    store_state_indices: torch.Tensor | None,
    scale: float | None,
    use_qk_l2norm: bool,
) -> bool:
    if not _device_is_hpu(packed_qkv):
        return False
    if state_pool.dtype != torch.float32 or state_pool.ndim != 4:
        return False
    if packed_qkv.dtype != torch.bfloat16 or log_decay.dtype != torch.float32 or beta.dtype != torch.bfloat16:
        return False
    if state_pool.shape[-2:] != (128, 128):
        return False
    if not use_qk_l2norm:
        return False
    expected_scale = 128**-0.5
    if scale is not None and abs(scale - expected_scale) > 1e-12:
        return False
    if load_state_indices is None or store_state_indices is None:
        return False
    if load_state_indices.shape != store_state_indices.shape:
        return False
    # Legacy public GUIDs only support same-slot read/write. New native ops may
    # relax this, but equality remains the safe common contract.
    if load_state_indices is not store_state_indices:
        return False
    return packed_qkv.shape[0] == load_state_indices.numel()


def _call_native_packed(
    op,
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
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
    policy = get_backend_policy()
    if not _device_is_hpu(packed_qkv) and policy in ("public", "bridge"):
        raise BackendUnavailableError(f"The forced {policy} backend requires an HPU tensor.")
    if policy == "pytorch" or not _device_is_hpu(packed_qkv):
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

    supported = _native_packed_supported(
        packed_qkv,
        log_decay,
        beta,
        state_pool,
        load_state_indices,
        store_state_indices,
        scale,
        use_qk_l2norm,
    )
    assert load_state_indices is not None

    if policy == "bridge" or (policy == "auto" and bridge_auto_enabled()):
        bridge_op = bridge_packed_gdn_op()
        if supported and bridge_op is not None:
            return _call_native_packed(bridge_op, packed_qkv, log_decay, beta, state_pool,
                                       load_state_indices), state_pool
        if policy == "bridge":
            raise BackendUnavailableError("The bridge GDN backend is unavailable for this runtime or shape.")

    public_op = public_packed_gdn_op()
    use_public = policy == "public" or (policy == "auto" and public_gdn_auto_promoted())
    if use_public and supported and public_op is not None:
        return _call_native_packed(public_op, packed_qkv, log_decay, beta, state_pool, load_state_indices), state_pool
    if policy == "public":
        raise BackendUnavailableError("The public native GDN backend is unavailable for this runtime or shape.")

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
