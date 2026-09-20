# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape native CSA2 selection with device-valued context lengths."""

import torch


# A 2052-token prompt followed by 256 decode steps stays below this bound.
# The bounded mirror accelerates that common production profile while longer
# requests retain the capacity-independent packed scorer and the 1M contract.
INDEX_MME_HOT_TOKENS = 2560


def _mme_hot_scores(query, weights, decoded_keys, positions):
    """Reproduce the checkpoint's two BF16 TP-shard sum boundaries on MME."""
    raw_scores = torch.einsum("bhd,nd->bhn", query, decoded_keys).contiguous()
    # Keep the large dot product on MME and fuse the following ReLU, per-head
    # weight, and the two checkpoint BF16 shard boundaries into one TPC pass.
    # The native reducer also writes -inf past the device-valued visible row,
    # avoiding all generic reduction intermediates without changing selection.
    return torch.ops.custom_op.custom_deepseek_v41_index_reduce_bf16_gaudi2(
        raw_scores, weights, positions, 1)


def runtime_index_select(query, weights, cache, pages, positions, candidates, *, ratio, capacity,
                         reindex=False, publish_candidates=False, decoded_hot=None):
    """Return ordered top-512 rows and, for a Full layer, top-2048 blocks.

    Tensor shapes remain fixed while the native kernels derive useful work
    from ``positions``.  Padding outside the visible prefix is never eligible.
    """
    if ratio not in (1, 2) or capacity < 512 or capacity > 1048576 or capacity % 64:
        raise ValueError("Runtime indexer requires ratio 1/2 and a bounded aligned capacity")
    if reindex and publish_candidates:
        raise ValueError("A candidate consumer cannot publish a new candidate pool")
    ops = torch.ops.custom_op
    columns = 16384 if reindex else capacity
    if decoded_hot is None:
        scores, block_scores = ops.custom_deepseek_v41_index_scores_gaudi2(
            query, weights, cache, pages, positions, candidates, ratio, int(reindex), columns)
    else:
        if (ratio != 1 or decoded_hot.dtype != torch.bfloat16 or decoded_hot.ndim != 2
                or decoded_hot.shape != (INDEX_MME_HOT_TOKENS, 128)):
            raise ValueError("MME index hot cache requires the fixed ratio-1 BF16 geometry")
        scores = _mme_hot_scores(query, weights, decoded_hot, positions)
        # Below 16K every visible block survives the candidate publication.
        # Its device-valued direct branch does not read these placeholders.
        block_scores = torch.empty((query.shape[0], INDEX_MME_HOT_TOKENS // 8),
                                   dtype=torch.float32,
                                   device=query.device)
    # The decoded hot bucket is a causal prefix shorter than 16K rows.  The
    # Full layer's candidate blocks are therefore exactly the identity block
    # prefix, so a Reindex owner can select the same logical rows directly.
    # This keeps the exact ordered result while avoiding candidate reads and
    # the badly imbalanced 16K worker partition.  Once the request leaves the
    # hot bucket ``decoded_hot`` is absent and the normal Reindex contract is
    # restored before the first consumer runs.
    select_reindex = int(reindex and decoded_hot is None)
    stats = ops.custom_deepseek_v41_index_threshold_gaudi2(
        scores, positions, ratio, select_reindex, 0)
    selected = ops.custom_deepseek_v41_index_emit_gaudi2(
        scores, positions, candidates, stats, ratio, select_reindex, 0)
    # The emitter partitions the score vector by increasing source offset and
    # starts every worker at the exact prefix count of retained entries.  Full
    # output is therefore already in increasing logical-row order.  Reindex
    # candidate blocks are also published in increasing logical order, so its
    # mapped rows retain that order.  Sorting either result repeats work and
    # adds a bitonic-sort kernel to every selection layer.
    blocks = None
    # No hot-bucket consumer observes candidate_pool: all of them use the
    # equivalent direct prefix above.  Skip both publication kernels.  At the
    # first non-hot token the source layer runs before Reindex layers and
    # refreshes candidate_pool for the same generation.
    if publish_candidates and decoded_hot is None:
        stats = ops.custom_deepseek_v41_index_threshold_gaudi2(
            block_scores, positions, ratio, 0, 1)
        blocks = ops.custom_deepseek_v41_index_emit_gaudi2(
            block_scores, positions, candidates, stats, ratio, 0, 1)
    return selected, blocks
