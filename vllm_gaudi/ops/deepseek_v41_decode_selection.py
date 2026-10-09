# SPDX-License-Identifier: Apache-2.0
"""Batch independent local CSA2 selections without changing merge order."""

import torch


def threshold_decode_selection(score_tiles, rows, positions, ratio, *, collect_blocks=False):
    """Reuse the common ordered selection kernel after existing causal scoring.

    Scores retain the checkpoint BF16 boundary. Source offsets are mapped
    back to logical IDs after selection, including Reindex candidate pools.
    No scoring, TP communication, or per-query visibility is changed.
    """
    from vllm_gaudi.ops.deepseek_v41_indexer import ordered_index_ids

    scores = torch.cat(score_tiles, -1).contiguous()
    tokens, columns = scores.shape
    if (not 1 <= tokens <= 6 or columns < 512 or columns > 32768 or columns % 64
            or rows.shape[-1] != columns or rows.ndim not in (1, 2)
            or (rows.ndim == 2 and rows.shape[0] != tokens) or positions.shape != (tokens,)):
        raise ValueError("Invalid causal decode threshold selection geometry")
    # Visibility has already been applied by the unchanged score producer.
    # The native selector here selects offsets in that fixed score tensor.
    extent = torch.full_like(positions, columns - 1)
    unused = torch.empty((tokens, 2048), dtype=torch.int32, device=scores.device)
    if rows.ndim == 1:
        # Full's source is the unchanged arange(0, columns). The common C1
        # emitter already returns these logical IDs; an i64 gather is redundant.
        indices = ordered_index_ids(scores, extent, unused, 1)
    else:
        # Reindex source rows are blocks*8 + arange(8). Reuse C1's native
        # logical-ID mapping in its emitter, preserving the supplied block
        # order and all causal score masks, rather than gather offsets later.
        candidates = torch.where(rows[:, ::8] >= 0, rows[:, ::8] // 8, -1).int().contiguous()
        candidates = torch.nn.functional.pad(candidates, (0, 2048 - candidates.shape[-1]), value=-1)
        indices = ordered_index_ids(scores, extent, candidates, 1, reindex=True)
    block_scores = block_ids = None
    if collect_blocks:
        grouped = scores.reshape(tokens, -1, 8).amax(-1)
        block_ids = torch.arange(columns // 8, dtype=torch.int32, device=positions.device)
        newest = (((positions + 1) // ratio - 1) // 8).unsqueeze(-1)
        grouped = grouped.masked_fill(block_ids == newest, torch.inf).contiguous()
        if grouped.shape[-1] > 2048:
            block_ids = ordered_index_ids(grouped, extent, unused, 1, blocks=True)
            block_scores = grouped.gather(1, block_ids.clamp_min(0).long())
            block_scores = torch.where(block_ids >= 0, block_scores, -torch.inf)
        else:
            block_scores = grouped
            block_ids = block_ids.expand(tokens, -1)
    return indices, block_scores, block_ids

def can_batch_full_r1(owner, query, rows, prefix_scores, native_scores, active_columns):
    return (getattr(owner, "decode_batched_selection", False) and owner.tensor_parallel_size == 4
            and not native_scores and owner.ratio == 1 and owner.layer == owner.candidate_source
            and query.device.type == "hpu" and query.shape[0] == 1 and rows.ndim == 1
            and owner.search_length == rows.shape[-1] == 32768 and prefix_scores is not None
            and 0 < active_columns <= rows.shape[-1] and active_columns % 2048 == 0)


def _select(scores, rows, width):
    # Keep the parent's score-only sort and subsequent ID gather. Native
    # carried-ID sorting adds a second operand and different buffer copies.
    values, offsets = scores.topk(min(width, scores.shape[-1]), dim=-1, sorted=False)
    return values, rows.gather(1, offsets)


def batched_decode_selection(score_tiles, rows, positions, ratio, *, width=512,
                             collect_blocks=False, selected_tiles=None):
    """Batch local TopKs, retaining every original sequential merge boundary."""
    tiles = len(score_tiles)
    if not tiles:
        raise ValueError("Decode selection requires at least one source tile")
    tokens, columns = score_tiles[0].shape
    selected_tiles = tiles if selected_tiles is None else selected_tiles
    if (not 1 <= tokens <= 6 or columns != 2048 or not 1 <= selected_tiles <= tiles
            or ratio not in (1, 2) or not 1 <= width <= columns
            or any(score.shape != (tokens, columns) for score in score_tiles)
            or rows.ndim not in (1, 2) or rows.shape[-1] != tiles * columns
            or (rows.ndim == 2 and rows.shape[0] != tokens)
            or positions.shape != (tokens,)):
        raise ValueError("Invalid bounded decode selection geometry")
    source = rows.expand(tokens, -1) if rows.ndim == 1 else rows
    ids = source.reshape(tokens, tiles, columns).transpose(0, 1).contiguous()
    scores = torch.stack(score_tiles, 0)
    local_scores, local_ids = _select(scores[:selected_tiles].reshape(-1, columns),
                                     ids[:selected_tiles].reshape(-1, columns), width)
    local_scores = local_scores.reshape(selected_tiles, tokens, width)
    local_ids = local_ids.reshape(selected_tiles, tokens, width)
    best_scores, best_ids = local_scores[0], local_ids[0]
    for tile in range(1, selected_tiles):
        best_scores, best_ids = _select(torch.cat((best_scores, local_scores[tile]), -1),
                                        torch.cat((best_ids, local_ids[tile]), -1), width)
    block_scores = block_ids = None
    if collect_blocks:
        block_columns = columns // 8
        grouped = scores.reshape(tiles, tokens, block_columns, 8).amax(-1)
        blocks = torch.arange(tiles * block_columns, dtype=torch.int32,
                              device=positions.device).reshape(tiles, 1, block_columns).expand(-1, tokens, -1)
        newest = (((positions + 1) // ratio - 1) // 8).reshape(1, tokens, 1)
        grouped = grouped.masked_fill(blocks == newest, torch.inf)
        local_scores, local_ids = _select(grouped.reshape(-1, block_columns),
                                         blocks.contiguous().reshape(-1, block_columns), block_columns)
        local_scores = local_scores.reshape(tiles, tokens, block_columns)
        local_ids = local_ids.reshape(tiles, tokens, block_columns)
        block_scores, block_ids = local_scores[0], local_ids[0]
        for tile in range(1, tiles):
            values = torch.cat((block_scores, local_scores[tile]), -1)
            ids = torch.cat((block_ids, local_ids[tile]), -1)
            block_scores, block_ids = _select(values, ids, min(2048, values.shape[-1]))
    return best_ids, block_scores, block_ids
