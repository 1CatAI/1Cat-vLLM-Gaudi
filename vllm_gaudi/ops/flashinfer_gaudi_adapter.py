# SPDX-License-Identifier: Apache-2.0
"""vLLM-facing adapter for the in-tree Gaudi FlashInfer implementation."""

from __future__ import annotations

import torch

from flashinfer_gaudi.gdn_decode import gated_delta_rule_decode_packed
from vllm_gaudi import envs


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
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the packed GDN backend, or return ``None`` when it is disabled.

    Backend availability fallback is handled inside ``flashinfer_gaudi``
    before any state-mutating operation is launched.
    """
    if not flashinfer_gdn_enabled() or state_pool is None or load_state_indices is None:
        return None
    if store_state_indices is None:
        store_state_indices = load_state_indices

    output, updated_pool = gated_delta_rule_decode_packed(
        packed_qkv=mixed_qkv,
        log_decay=log_decay,
        beta=beta,
        state_pool=state_pool,
        load_state_indices=load_state_indices,
        store_state_indices=store_state_indices,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )
    return output.unsqueeze(0), updated_pool


__all__ = ["flashinfer_gdn_enabled", "maybe_run_gdn_decode_packed"]
