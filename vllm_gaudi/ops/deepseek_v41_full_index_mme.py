# SPDX-License-Identifier: Apache-2.0
"""Bounded concurrent Full scoring on MME, with the canonical long scorer.

Both branches have fixed addresses and shapes. Device positions suppress the
packed TPC dot products for requests in the bounded prefix. MME tiles are still
executed for the entire batch: this does not claim conditional MME execution.
The packed page pool remains canonical and no persistent decoded mirror is used.
"""

import torch

from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
from vllm_gaudi.ops.deepseek_v41_reindex_mme import _bf16_boundary

FULL_MME_PREFIX_TOKENS = 4096


def full_index_mme_select(query,
                          weights,
                          cache,
                          pages,
                          positions,
                          candidates,
                          *,
                          ratio,
                          capacity,
                          publish_candidates=False,
                          key_ready=None,
                          tiled_keys=False):
    if ratio not in (1, 2) or capacity < FULL_MME_PREFIX_TOKENS // ratio:
        raise ValueError("Full MME scoring requires ratio 1/2 and the complete bounded prefix")
    ops = torch.ops.custom_op
    if key_ready is not None:
        # Cache mutation must precede key gather itself, not only the query
        # consumer. -1 denotes a completed compression pair with no write.
        positions = torch.where(key_ready >= -1, positions, -1).contiguous()
    hot = (positions >= 0) & (positions < FULL_MME_PREFIX_TOKENS)
    hot_positions = torch.where(hot, positions, -1).contiguous()
    long_positions = torch.where(hot, -1, positions).contiguous()
    visible = ((hot_positions + 1) // ratio)[:, None]
    scores = []
    for begin in range(0, FULL_MME_PREFIX_TOKENS // ratio, 2048):
        rows = torch.arange(begin, begin + 2048, device=query.device, dtype=torch.int32)[None]
        valid = (rows < visible) & (visible > 512)
        rows = torch.where(valid, rows, -1).contiguous()
        gather = (ops.custom_deepseek_v41_index_keys_tiled_gaudi2
                  if tiled_keys else ops.custom_deepseek_v41_index_keys_gaudi2)
        keys = gather(cache, pages, rows, ratio)
        dots = _bf16_boundary(torch.bmm(query, keys.transpose(1, 2)))
        score = ops.custom_deepseek_v41_index_reduce_gaudi2(dots.contiguous(), weights)
        scores.append(score.masked_fill(~valid, -torch.inf))
    scores = torch.cat(scores, -1) if len(scores) > 1 else scores[0]
    stats = ops.custom_deepseek_v41_index_threshold_gaudi2(scores, hot_positions, ratio, 0, 0)
    hot_selected = ops.custom_deepseek_v41_index_emit_gaudi2(scores, hot_positions, candidates, stats, ratio, 0, 0)
    selected, blocks = runtime_index_select(query,
                                            weights,
                                            cache,
                                            pages,
                                            long_positions,
                                            candidates,
                                            ratio=ratio,
                                            capacity=capacity,
                                            publish_candidates=publish_candidates)
    selected = torch.where(hot[:, None], hot_selected, selected)
    if publish_candidates:
        # At most 512 visible blocks; the emitter's direct prefix branch
        # does not inspect scores or metadata. Reuse initialized tensors
        # rather than introducing unread, uninitialized graph inputs.
        hot_blocks = ops.custom_deepseek_v41_index_emit_gaudi2(scores, hot_positions, candidates, stats, ratio, 0, 1)
        blocks = torch.where(hot[:, None], hot_blocks, blocks)
    return selected, blocks
