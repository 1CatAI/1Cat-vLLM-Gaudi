# SPDX-License-Identifier: Apache-2.0
"""Stable fixed-capacity candidate descriptors for bounded Reindex execution.

This is a preparation primitive, not a claim that a masked BMM skips work.
Consumers must use counts to suppress complete matrix tiles and retain the
source-slot map for tie resolution. No count is copied to the host.
"""

import torch


def bounded_reindex_bucket(rows):
    """Buckets qualified for the optional bounded-tile scorer and replay ABI."""
    return rows == 8


def unique_pool_tile_bound(positions, ratio, tile_rows=2048):
    """Scheduler-side bound for pools produced by the unique Full emitter.

    This contract does NOT apply to arbitrary externally supplied duplicate
    pools. Full's stable unique block selection and forward-only history bound
    the count by ceil(visible/8). Positions must already be owned host metadata;
    this function deliberately rejects tensors to prohibit implicit D2H.
    """
    if isinstance(positions, torch.Tensor) or ratio not in (1, 2) or tile_rows != 2048:
        raise ValueError("Tile bound requires host positions and the fixed 2048-row tile")
    if not 1 <= len(positions) <= 64 or any(not isinstance(p, int) or p < -1 or p >= 1048576 for p in positions):
        raise ValueError("Invalid scheduler position range")
    visible = max((p + 1) // ratio for p in positions)
    if visible <= 512:
        return 0
    return min(8, (visible + tile_rows - 1) // tile_rows)


def validate_compact_inputs(candidates, positions, ratio):
    if (candidates.ndim != 2 or candidates.shape[1] != 2048 or not 1 <= candidates.shape[0] <= 64
            or positions.shape != (candidates.shape[0], ) or candidates.dtype != torch.int32
            or positions.dtype != torch.int32 or candidates.device != positions.device
            or not candidates.is_contiguous() or not positions.is_contiguous() or ratio not in (1, 2)):
        raise ValueError("Reindex compaction requires contiguous I32 [B,2048]/[B], B=1..64, ratio=1/2")


def compact_reindex_reference(candidates, positions, ratio):
    """CPU oracle, preserving holes, duplicate IDs and original tie order.

    Counts are blocks, not key rows. The last retained block may contain
    invisible rows, so the consumer must still apply its per-row length mask.
    <=512 uses the existing direct-prefix emitter and needs no score tiles.
    """
    validate_compact_inputs(candidates, positions, ratio)
    if candidates.device.type != "cpu":
        raise ValueError("The compaction reference is CPU-only; no implicit device readback")
    compact = torch.full_like(candidates, -1)
    source_slots = torch.full_like(candidates, -1)
    counts = torch.zeros_like(positions)
    slots = torch.arange(2048, dtype=torch.int32)
    for request in range(candidates.shape[0]):
        visible = max(0, (int(positions[request]) + 1) // ratio)
        if visible <= 512:
            continue
        pool = candidates[request]
        # Compare before multiplication: malformed I32 IDs must not wrap into
        # valid key rows. Duplicates remain distinct route slots.
        valid = (pool >= 0) & (pool < (visible + 7) // 8)
        count = int(valid.sum())
        compact[request, :count] = pool[valid]
        source_slots[request, :count] = slots[valid]
        counts[request] = count
    return compact, source_slots, counts


def compact_reindex(candidates, positions, ratio):
    validate_compact_inputs(candidates, positions, ratio)
    if candidates.device.type != "hpu":
        raise ValueError("Native Reindex compaction requires HPU inputs")
    return torch.ops.custom_op.custom_deepseek_v41_reindex_compact_gaudi2(candidates, positions, ratio)
