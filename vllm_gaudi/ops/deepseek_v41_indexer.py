# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape native C1 CSA2 selection with device-valued context length.

The paged metadata/custom-op boundary follows vLLM 2c88fb131c7a,
model_executor/layers/sparse_attn_indexer.py. CUDA scoring instructions are
not reused. This experimental TPC implementation preserves the existing
FP4 roundtrip and two BF16 shard-sum boundaries.
"""

import torch


def runtime_index_select(query, weights, cache, pages, positions, candidates, *, ratio, capacity, reindex=False,
                         publish_candidates=False):
    """Return fixed top-512 rows and, for the source layer, top-2048 blocks.

All source and output buffers keep their shapes as positions change. The
native kernels determine valid work from positions; stale score workspace
outside that range is never eligible. Ties use increasing source slot order.
"""
    if ratio not in (1, 2) or capacity < 512 or capacity > 1048576 or capacity % 64:
        raise ValueError("Runtime indexer requires ratio 1/2 and a bounded, aligned capacity")
    if reindex and publish_candidates:
        raise ValueError("A candidate consumer cannot publish a new candidate pool")
    ops = torch.ops.custom_op
    columns = 16384 if reindex else capacity
    scores, block_scores = ops.custom_deepseek_v41_index_scores_gaudi2(
        query, weights, cache, pages, positions, candidates, ratio, int(reindex), columns)
    stats = ops.custom_deepseek_v41_index_threshold_gaudi2(scores, positions, ratio, int(reindex), 0)
    selected = ops.custom_deepseek_v41_index_emit_gaudi2(scores, positions, candidates, stats, ratio, int(reindex), 0)
    # Reindex pools need not be ordered. Preserve the public logical-row order
    # regardless of the physical order of candidate slots.
    selected = torch.where(selected >= 0, selected, 2147483647).sort(-1).values
    selected = torch.where(selected < 2147483647, selected, -1)
    blocks = None
    if publish_candidates:
        stats = ops.custom_deepseek_v41_index_threshold_gaudi2(block_scores, positions, ratio, 0, 1)
        blocks = ops.custom_deepseek_v41_index_emit_gaudi2(block_scores, positions, candidates, stats, ratio, 0, 1)
    return selected, blocks
