# SPDX-License-Identifier: Apache-2.0
"""Compile index K unpack, MME scores and weighted head reduction together."""
from functools import lru_cache
from types import FunctionType

import torch

from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4


def index_scores(query, weights, packed, table, positions, rows, ratio, local_heads, preserve_rounding=True,
                 native_keys=False):
    if native_keys:
        key_rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
        keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(packed, table, key_rows, ratio)
        if rows.ndim == 1:
            keys = keys[0]
    else:
        safe = rows.clamp_min(0)
        page_rows = 128 // ratio
        physical = table[(safe.flatten() // page_rows).long()].reshape(safe.shape) * page_rows + safe % page_rows
        keys = unpack_fp4(packed.index_select(0, physical.flatten().long()).reshape(*physical.shape, 68), 128, 32)
    scores = (torch.einsum("thd,nd->thn", query, keys) if keys.ndim == 2 else
              torch.einsum("thd,tnd->thn", query, keys))

    def boundary(value):
        if preserve_rounding:
            return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(
                value.reshape(1, -1)).reshape(value.shape)
        return value

    scores = boundary(scores)
    scores = boundary(scores.relu() * weights.unsqueeze(-1))
    partial = boundary(scores.reshape(query.shape[0], 2, local_heads, -1).sum(2))
    reduced = boundary(partial.sum(1)).float()
    visible = ((positions + 1) // ratio).unsqueeze(-1)
    valid = (rows >= 0) & (rows < visible)
    return reduced.masked_fill(~valid, -torch.inf)


@lru_cache(maxsize=64)
def compiled_index_scores(signature):
    entry = FunctionType(index_scores.__code__.replace(co_name=f"prefill_index_scores_{signature}"),
                         index_scores.__globals__, argdefs=index_scores.__defaults__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def shared_index_scores(query, weights, packed, table, positions, ratio, local_heads, source_rows):
    """Decode a bounded common source once, retaining all score boundaries.

    This computes scores only. Reindex must gather them back into its original
    candidate order before top-k: sorting a common source directly changes tie
    resolution and is not equivalent to the candidate-pool contract.
    """
    if not 1 <= source_rows <= 16384 or not 1 <= query.shape[0] <= 128:
        raise ValueError("Shared index scores require a bounded query/source tile")
    outputs = []
    for start in range(0, source_rows, 2048):
        rows = torch.arange(start, min(start + 2048, source_rows), device=query.device, dtype=torch.int32)
        outputs.append(index_scores(query, weights, packed, table, positions, rows, ratio, local_heads,
                                    True, True))
    return torch.cat(outputs, -1)


@lru_cache(maxsize=64)
def compiled_shared_index_scores(signature):
    entry = FunctionType(shared_index_scores.__code__.replace(co_name=f"shared_index_scores_{signature}"),
                         shared_index_scores.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def shared_reindex(query, weights, packed, table, positions, blocks, ratio, local_heads, source_rows):
    """Bounded common scoring followed by the original candidate-slot top-k.

    Preserve both the candidate order and the source-tile merge order. In
    particular, neither source deduplication nor a global top-k is a valid
    replacement for the existing tie behavior.
    """
    common = shared_index_scores(query, weights, packed, table, positions, ratio, local_heads, source_rows)
    rows = blocks.unsqueeze(-1) * 8 + torch.arange(8, device=query.device, dtype=torch.int32)
    rows = torch.where(blocks.unsqueeze(-1) >= 0, rows, -1).flatten(1)
    best_values = best_rows = None
    for start in range(0, rows.shape[-1], 2048):
        current = rows[:, start:start + 2048]
        scores = common.gather(1, current.clamp(0, source_rows - 1).long())
        scores = scores.masked_fill((current < 0) | (current >= source_rows), -torch.inf)
        values, offsets = scores.topk(min(512, scores.shape[-1]), dim=-1, sorted=False)
        selected = current.gather(1, offsets)
        if best_values is not None:
            values = torch.cat((best_values, values), -1)
            selected = torch.cat((best_rows, selected), -1)
            values, offsets = values.topk(min(512, values.shape[-1]), dim=-1, sorted=False)
            selected = selected.gather(1, offsets)
        best_values, best_rows = values, selected
    count = ((positions + 1) // ratio).unsqueeze(-1)
    # Source capacity is a safe sentinel strictly beyond every eligible row.
    selected = torch.where((best_rows >= 0) & (best_rows < count), best_rows, source_rows)
    selected = selected.sort(-1).values
    return torch.where(selected < source_rows, selected, -1).int()


@lru_cache(maxsize=64)
def compiled_shared_reindex(signature):
    entry = FunctionType(shared_reindex.__code__.replace(co_name=f"shared_reindex_{signature}"),
                         shared_reindex.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
