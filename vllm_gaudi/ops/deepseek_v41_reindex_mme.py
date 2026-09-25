# SPDX-License-Identifier: Apache-2.0
"""Bounded MME reindex scoring with the native C1 head reduction order.

Candidate slots retain their order, including holes. Only the 2048 candidate
blocks are visited; full context capacity never becomes a decoded KV tensor.
This experimental entry is not enabled in model dispatch until qualified.
"""

import torch


def _bf16_boundary(value):
    return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(value.to(torch.bfloat16).reshape(
        1, -1)).reshape(value.shape)


def reindex_mme_scores(query, weights, cache, pages, positions, candidates, ratio, tile=2048, *, tiled_keys=False):
    if tile not in (512, 1024, 2048) or ratio not in (1, 2):
        raise ValueError("Unsupported reindex tile or compression ratio")
    if query.ndim != 3 or query.shape[1:] != (32, 128):
        raise ValueError("Reindex queries must have shape [requests,32,128]")
    if candidates.shape != (query.shape[0], 2048):
        raise ValueError("Reindex candidate pool must contain 2048 blocks per request")
    visible = ((positions + 1) // ratio).unsqueeze(-1)
    offsets = torch.arange(8, dtype=torch.int32, device=query.device)
    outputs = []
    for start in range(0, 2048, tile // 8):
        blocks = candidates[:, start:start + tile // 8]
        rows = (blocks.unsqueeze(-1) * 8 + offsets).flatten(1)
        valid = ((blocks >= 0).unsqueeze(-1).expand(-1, -1, 8).flatten(1) & (rows < visible) & (visible > 512))
        rows = torch.where(valid, rows, -1).contiguous()
        gather = (torch.ops.custom_op.custom_deepseek_v41_index_keys_tiled_gaudi2
                  if tiled_keys else torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2)
        keys = gather(cache, pages, rows, ratio)
        dots = _bf16_boundary(torch.bmm(query, keys.transpose(1, 2)))
        scores = torch.ops.custom_op.custom_deepseek_v41_index_reduce_gaudi2(dots.contiguous(), weights)
        outputs.append(scores.masked_fill(~valid, -torch.inf))
    return torch.cat(outputs, -1)


def reindex_mme_select(query, weights, cache, pages, positions, candidates, *, ratio, tile=2048, tiled_keys=False):
    scores = reindex_mme_scores(query, weights, cache, pages, positions, candidates, ratio, tile, tiled_keys=tiled_keys)
    ops = torch.ops.custom_op
    stats = ops.custom_deepseek_v41_index_threshold_gaudi2(scores, positions, ratio, 1, 0)
    selected = ops.custom_deepseek_v41_index_emit_gaudi2(scores, positions, candidates, stats, ratio, 1, 0)
    selected = torch.where(selected >= 0, selected, 2147483647).sort(-1).values
    return torch.where(selected < 2147483647, selected, -1)


def bounded_reindex_mme_select(query, weights, cache, pages, positions, candidates, *, ratio, tiled_keys=False):
    """Fixed short-prefix and whole-pool graphs, selected by native publication.

    Ordinary compilation executes both tactics. Only the prepared native plan
    may omit tiles, using scheduler-owned bounds for unique Full-emitter pools.
    The final mandatory mask prevents stale optional outputs from escaping.
    """
    ops = torch.ops.custom_op
    score_tile = (ops.custom_deepseek_v41_reindex_tiled_gaudi2
                  if tiled_keys else ops.custom_deepseek_v41_reindex_tile_gaudi2)
    compact, _, counts = ops.custom_deepseek_v41_reindex_compact_gaudi2(candidates, positions, ratio)
    offsets = torch.arange(8, dtype=torch.int32, device=query.device)
    visible = ((positions + 1) // ratio).unsqueeze(-1)
    scores = []
    for ordinal in range(4):
        blocks = compact[:, ordinal * 256:(ordinal + 1) * 256]
        rows = (blocks.unsqueeze(-1) * 8 + offsets).flatten(1)
        valid = (rows >= 0) & (rows < visible) & (visible > 512)
        rows = torch.where(valid, rows, -1).contiguous()
        score = score_tile(query, weights, cache, pages, rows, ratio, ordinal)
        scores.append(score)
    scores.append(torch.full((query.shape[0], 8192), -torch.inf, dtype=torch.float32, device=query.device))
    scores = torch.cat(scores, -1)
    rows = (compact.unsqueeze(-1) * 8 + offsets).flatten(1)
    valid = ((torch.arange(16384, device=query.device)[None] < counts[:, None] * 8) & (rows >= 0) & (rows < visible) &
             (visible > 512))
    full_rows = torch.where(valid, rows, -1).contiguous()
    whole = score_tile(query, weights, cache, pages, full_rows, ratio, 8)
    scores = torch.where(visible.max() > 8192, whole, scores)
    scores = scores.masked_fill(~valid, -torch.inf)
    stats = ops.custom_deepseek_v41_index_threshold_gaudi2(scores, positions, ratio, 1, 0)
    selected = ops.custom_deepseek_v41_index_emit_gaudi2(scores, positions, compact, stats, ratio, 1, 0)
    selected = torch.where(selected >= 0, selected, 2147483647).sort(-1).values
    return torch.where(selected < 2147483647, selected, -1)
