# SPDX-License-Identifier: Apache-2.0
"""vLLM-facing adapter for the in-tree Gaudi FlashInfer implementation."""

from __future__ import annotations

import torch

from flashinfer_gaudi._config import bridge_auto_enabled, get_backend_policy
from flashinfer_gaudi._dispatch import require_reference_allowed
from flashinfer_gaudi._reference import packed_recurrent_decode, qwen38_fused_decode_step_direct
from flashinfer_gaudi._tactics import gdn_fused_decode_tactic, gdn_prefill_tactic, public_gdn_auto_promoted
from flashinfer_gaudi.gdn_decode import gated_delta_rule_decode_packed
from flashinfer_gaudi.gdn_prefill import _chunk_gated_delta_rule_log_gate
from vllm_gaudi import envs

_BACKEND_POLICY = get_backend_policy()
_PUBLIC_AUTO_PROMOTED = public_gdn_auto_promoted()
_BRIDGE_AUTO_ENABLED = bridge_auto_enabled()
_GDN_PREFILL_TACTIC = gdn_prefill_tactic()
_GDN_FUSED_DECODE_TACTIC = gdn_fused_decode_tactic()
_GDN_FUSED_DECODE_MODEL_SHAPE = _GDN_FUSED_DECODE_TACTIC.get("model_shape", {})
_GDN_FUSED_DECODE_BATCHES = frozenset(
    _GDN_FUSED_DECODE_MODEL_SHAPE.get("batch_buckets", ()) if isinstance(_GDN_FUSED_DECODE_MODEL_SHAPE, dict) else ())
_GDN_FUSED_DECODE_LOCAL_GEOMETRIES = frozenset({
    (10240, 48),
    (5120, 24),
})


def flashinfer_gdn_enabled() -> bool:
    """Return whether the new backend is enabled for model execution."""
    return envs.VLLM_HPU_FLASHINFER_GDN


def flashinfer_gdn_prefill_enabled() -> bool:
    """Return whether the promoted GDN prefill tactic is enabled."""
    return envs.VLLM_HPU_FLASHINFER_GDN_PREFILL and bool(_GDN_PREFILL_TACTIC.get("promoted", False))


def flashinfer_gdn_fused_decode_enabled() -> bool:
    """Return whether the qualified fused direct-state recipe is enabled."""
    return envs.VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE and bool(_GDN_FUSED_DECODE_TACTIC.get("promoted", False))


def can_bind_gdn_active_state(batch: int, packed_width: int, value_heads: int) -> bool:
    """Gate the eager state-view binding to the direct reference composition."""
    use_reference = _BACKEND_POLICY == "pytorch" or (_BACKEND_POLICY == "auto" and not _PUBLIC_AUTO_PROMOTED
                                                     and not _BRIDGE_AUTO_ENABLED)
    return (flashinfer_gdn_enabled() and flashinfer_gdn_fused_decode_enabled() and use_reference
            and batch in _GDN_FUSED_DECODE_BATCHES and (packed_width, value_heads) in _GDN_FUSED_DECODE_LOCAL_GEOMETRIES
            and (value_heads != 24 or envs.VLLM_HPU_FLASHINFER_GDN_TP2))


def maybe_run_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
    prefill_num_seqs: int,
    prefill_seq_len: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Run the single-sequence Qwen3.8 prefill tactic on local TP heads.

    Unsupported shapes deliberately return ``None`` so other Qwen GDN
    variants retain the general HPU implementation.
    """
    if flashinfer_gdn_enabled():
        require_reference_allowed("vLLM GDN prefill")
    if not flashinfer_gdn_prefill_enabled():
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return None
    if tuple(q.shape) != tuple(k.shape) or q.shape[0] != 1 or v.shape[0] != 1:
        return None
    key_heads, value_heads = q.shape[2], v.shape[2]
    if (q.shape[1] != v.shape[1] or q.shape[3] != 128 or v.shape[3] != 128
            or (key_heads, value_heads) not in ((16, 48), (8, 24))):
        return None
    if value_heads == 24 and not envs.VLLM_HPU_FLASHINFER_GDN_TP2:
        return None
    if log_decay.shape != (1, q.shape[1], value_heads) or beta.shape != (1, q.shape[1], value_heads):
        return None
    if log_decay.dtype != torch.float32:
        return None
    if chunk_size != 128 or prefill_num_seqs != 1 or prefill_seq_len != q.shape[1]:
        return None

    tuning = _GDN_PREFILL_TACTIC.get("tuning", {})
    if not isinstance(tuning, dict):
        return None
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(tuning.get("compute_dtype"))
    if compute_dtype is None:
        return None
    return _chunk_gated_delta_rule_log_gate(
        q,
        k,
        v,
        log_decay,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        prefill_num_seqs=prefill_num_seqs,
        prefill_seq_len=prefill_seq_len,
        neumann_iters=int(tuning["neumann_iters"]),
        fused_state_matmul=bool(tuning["fused_state_matmul"]),
        recursive_solver_base=int(tuning["recursive_solver_base"]),
        compact_repeated_kkt=bool(tuning["compact_repeated_kkt"]),
        compile_qk_l2norm=bool(tuning["compile_qk_l2norm"]),
        flashqla_reformulation=bool(tuning["flashqla_reformulation"]),
        deferred_output_add=bool(tuning["deferred_output_add"]),
        compute_dtype=compute_dtype,
        solve_in_fp32=bool(tuning["solve_in_fp32"]),
        state_in_fp32=bool(tuning["state_in_fp32"]),
        preserve_compact_qk=bool(tuning["preserve_compact_qk"]),
        masked_triangular_decay=bool(tuning["masked_triangular_decay"]),
    )


def maybe_run_gdn_decode_packed(
    mixed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor | None,
    load_state_indices: torch.Tensor | None,
    store_state_indices: torch.Tensor | None,
    *,
    use_qk_l2norm: bool,
    scale: float | None = None,
    direct_state_layout: bool = False,
    direct_state_group_count: int | None = None,
    direct_state_group_offset: int | None = None,
    allow_indexed_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the packed GDN backend, or return ``None`` when it is disabled.

    Backend availability fallback is handled inside ``flashinfer_gaudi``
    before any state-mutating operation is launched.
    """
    if flashinfer_gdn_enabled():
        require_reference_allowed("vLLM GDN packed decode")
    if not flashinfer_gdn_enabled() or state_pool is None or load_state_indices is None:
        return None
    # The packed integration currently replaces vLLM's one-token recurrent
    # decode only. Speculative/MTP batches have more packed token rows than
    # request-state indices and must remain on the existing general path.
    if mixed_qkv.shape[0] != load_state_indices.numel():
        return None
    if store_state_indices is None:
        store_state_indices = load_state_indices

    use_reference = _BACKEND_POLICY == "pytorch" or (_BACKEND_POLICY == "auto" and not _PUBLIC_AUTO_PROMOTED
                                                     and not _BRIDGE_AUTO_ENABLED)
    selected_state_pool = state_pool
    use_direct_state = False
    if (use_reference and direct_state_layout and direct_state_group_count is not None and direct_state_group_count > 0
            and direct_state_group_offset is not None):
        group_span = (state_pool.shape[0] - 2) // direct_state_group_count
        if mixed_qkv.shape[0] <= group_span:
            state_start = direct_state_group_offset * group_span + 1
            selected_state_pool = state_pool.narrow(0, state_start, mixed_qkv.shape[0])
            use_direct_state = True

    # The indexed PyTorch compatibility path performs state gather/scatter
    # around the recurrence and loses decisively to vLLM's existing decode
    # implementation.  In auto mode, only select the measured contiguous
    # state fast path; explicit ``pytorch`` policy still exposes the complete
    # compatibility implementation for testing and unsupported layouts.
    if use_reference and _BACKEND_POLICY == "auto" and not use_direct_state and not allow_indexed_reference:
        return None

    if use_reference:
        output, updated_pool = packed_recurrent_decode(
            mixed_qkv,
            log_decay,
            beta,
            selected_state_pool,
            load_state_indices,
            store_state_indices,
            scale,
            use_qk_l2norm,
            use_direct_state,
        )
    else:
        output, updated_pool = gated_delta_rule_decode_packed(
            packed_qkv=mixed_qkv,
            log_decay=log_decay,
            beta=beta,
            state_pool=selected_state_pool,
            load_state_indices=load_state_indices,
            store_state_indices=store_state_indices,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
        )
    return output.unsqueeze(0), updated_pool


def maybe_run_gdn_fused_decode_step(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    ssm_state: torch.Tensor | None,
    load_state_indices: torch.Tensor | None,
    *,
    direct_conv_state: bool,
    direct_gdn_state: bool,
    direct_state_group_count: int | None,
    direct_state_group_offset: int | None,
    scale: float,
    state_is_active_view: bool = False,
    defer_state_writeback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the shape-gated direct-state decode composition on local TP heads."""
    if flashinfer_gdn_enabled():
        require_reference_allowed("vLLM fused GDN decode")
    use_reference = _BACKEND_POLICY == "pytorch" or (_BACKEND_POLICY == "auto" and not _PUBLIC_AUTO_PROMOTED
                                                     and not _BRIDGE_AUTO_ENABLED)
    if (not flashinfer_gdn_enabled() or not flashinfer_gdn_fused_decode_enabled() or not use_reference
            or ssm_state is None or load_state_indices is None):
        return None
    if not direct_conv_state or not direct_gdn_state:
        return None
    if direct_state_group_count is None or direct_state_group_count <= 0 or direct_state_group_offset is None:
        return None
    token_rows = mixed_qkv.shape[0]
    state_rows = load_state_indices.numel()
    ordinary_decode = token_rows == state_rows and token_rows in _GDN_FUSED_DECODE_BATCHES
    if not ordinary_decode:
        return None
    if defer_state_writeback and (not ordinary_decode or not state_is_active_view):
        return None
    packed_width = mixed_qkv.shape[1] if mixed_qkv.ndim == 2 else -1
    value_heads = A_log.numel()
    if value_heads == 24 and not envs.VLLM_HPU_FLASHINFER_GDN_TP2:
        return None
    if (packed_width, value_heads) not in _GDN_FUSED_DECODE_LOCAL_GEOMETRIES:
        return None
    if (mixed_qkv.dtype != torch.bfloat16 or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or tuple(mixed_qkv.shape[1:]) != (packed_width, ) or tuple(a.shape) != (token_rows, value_heads)
            or tuple(b.shape) != (token_rows, value_heads)):
        return None
    if (tuple(conv_state.shape) != (state_rows, 3, packed_width) or conv_state.dtype != torch.bfloat16
            or tuple(conv_weight.shape) != (packed_width, 4) or conv_weight.dtype != torch.bfloat16):
        return None
    if conv_bias is not None and (tuple(conv_bias.shape) != (packed_width, ) or conv_bias.dtype != torch.bfloat16):
        return None
    if (tuple(A_log.shape) != (value_heads, ) or tuple(dt_bias.shape) != (value_heads, ) or A_log.dtype != torch.float32
            or dt_bias.dtype != torch.bfloat16):
        return None

    if state_is_active_view:
        if not ordinary_decode:
            return None
        selected_ssm_state = ssm_state
    else:
        group_span = (ssm_state.shape[0] - 2) // direct_state_group_count
        if state_rows > group_span or not 0 <= direct_state_group_offset < direct_state_group_count:
            return None
        state_start = direct_state_group_offset * group_span + 1
        selected_ssm_state = ssm_state.narrow(0, state_start, state_rows)
    if (tuple(selected_ssm_state.shape) != (state_rows, value_heads, 128, 128)
            or selected_ssm_state.dtype != torch.float32):
        return None

    output, _, updated_state = qwen38_fused_decode_step_direct(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        conv_state,
        conv_weight,
        conv_bias,
        selected_ssm_state,
        scale,
        inplace_state=not defer_state_writeback,
        direct_state_update=(envs.VLLM_HPU_GDN_DIRECT_STATE_UPDATE and state_is_active_view and token_rows == 1
                             and value_heads == 24 and not defer_state_writeback),
    )
    return output.unsqueeze(0), updated_state


__all__ = [
    "flashinfer_gdn_enabled",
    "flashinfer_gdn_fused_decode_enabled",
    "flashinfer_gdn_prefill_enabled",
    "maybe_run_gdn_decode_packed",
    "maybe_run_gdn_fused_decode_step",
    "maybe_run_gdn_prefill",
]
