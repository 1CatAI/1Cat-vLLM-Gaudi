# SPDX-License-Identifier: Apache-2.0
"""vLLM-facing adapter for the in-tree Gaudi FlashInfer implementation."""

from __future__ import annotations

import torch

from flashinfer_gaudi._config import bridge_auto_enabled, get_backend_policy
from flashinfer_gaudi._reference import packed_recurrent_decode
from flashinfer_gaudi._tactics import public_gdn_auto_promoted
from flashinfer_gaudi.gdn_decode import gated_delta_rule_decode_packed
from vllm_gaudi import envs

_BACKEND_POLICY = get_backend_policy()
_PUBLIC_AUTO_PROMOTED = public_gdn_auto_promoted()
_BRIDGE_AUTO_ENABLED = bridge_auto_enabled()


def flashinfer_gdn_enabled() -> bool:
    """Return whether the new backend is enabled for model execution."""
    return envs.VLLM_HPU_FLASHINFER_GDN


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
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the packed GDN backend, or return ``None`` when it is disabled.

    Backend availability fallback is handled inside ``flashinfer_gaudi``
    before any state-mutating operation is launched.
    """
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
    if use_reference and _BACKEND_POLICY == "auto" and not use_direct_state:
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


__all__ = ["flashinfer_gdn_enabled", "maybe_run_gdn_decode_packed"]
