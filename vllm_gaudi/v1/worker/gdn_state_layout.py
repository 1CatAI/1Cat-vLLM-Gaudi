# SPDX-License-Identifier: Apache-2.0
"""CPU ownership proofs for optional direct DFlash2 checkpoint writes."""

import torch


def has_contiguous_dflash_checkpoints(
    state_indices: torch.Tensor,
    group_offsets: dict[int, int],
    request_capacity: int,
    active_requests: int,
    padded_requests: int,
    tokens_per_request: int,
    slots_per_request: int,
    full_query: bool,
) -> bool:
    """Prove group-major contiguous ownership before using a static slice.

    Call only for compact state with prefix caching disabled. All requests
    must occupy base slots in input order starting at zero. A reordered,
    padded, noncompact, or partially filled verification batch fails closed.
    The proof is CPU-only and must run before H2D transfer or graph capture.
    """
    if state_indices.device.type != "cpu":
        raise ValueError("Checkpoint ownership must be validated on CPU metadata.")
    if (not full_query or not group_offsets or active_requests < 1 or active_requests != padded_requests
            or active_requests > request_capacity or tokens_per_request != 8 or slots_per_request != tokens_per_request
            or state_indices.ndim != 3 or state_indices.shape[1:] != (padded_requests, tokens_per_request)):
        return False
    offsets = list(group_offsets.values())
    if sorted(offsets) != list(range(len(offsets))):
        return False
    relative = torch.arange(active_requests * tokens_per_request,
                            dtype=state_indices.dtype).reshape(active_requests, tokens_per_request)
    for group_index, group_offset in group_offsets.items():
        if group_index < 0 or group_index >= state_indices.shape[0]:
            return False
        first_slot = group_offset * request_capacity * slots_per_request + 1
        if not torch.equal(state_indices[group_index], relative + first_slot):
            return False
    return True
