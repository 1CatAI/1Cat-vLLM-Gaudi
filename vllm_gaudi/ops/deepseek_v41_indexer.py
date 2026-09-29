# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape native CSA2 selection with device-valued context lengths.

The paged metadata/custom-op boundary follows vLLM 2c88fb131c7a,
model_executor/layers/sparse_attn_indexer.py. CUDA scoring instructions are
not reused. This experimental TPC implementation preserves the existing
FP4 roundtrip and two BF16 shard-sum boundaries.
"""

import torch

# A 2052-token prompt followed by 256 decode steps stays below this bound.
# The bounded mirror accelerates that common production profile while longer
# requests retain the capacity-independent packed scorer and the 1M contract.
INDEX_MME_HOT_TOKENS = 2560


def _mme_hot_scores(query, weights, decoded_keys, positions, ratio):
    """Reproduce the checkpoint's two BF16 TP-shard sum boundaries on MME.

    ``decoded_keys`` follows the compressed row geometry, so ratio-2 Full
    layers execute half as many MME columns as ratio-1 layers while retaining
    the same token-valued visibility boundary in the native reducer.
    """
    raw_scores = torch.einsum("bhd,nd->bhn", query, decoded_keys).contiguous()
    # Keep the large dot product on MME and fuse the following ReLU, per-head
    # weight, and the two checkpoint BF16 shard boundaries into one TPC pass.
    # The native reducer also writes -inf past the device-valued visible row,
    # avoiding all generic reduction intermediates without changing selection.
    return torch.ops.custom_op.custom_deepseek_v41_index_reduce_bf16_gaudi2(raw_scores, weights, positions, ratio)


def runtime_index_select(query,
                         weights,
                         cache,
                         pages,
                         positions,
                         candidates,
                         *,
                         ratio,
                         capacity,
                         reindex=False,
                         publish_candidates=False,
                         decoded_hot=None,
                         ordered_candidates=False):
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
        scores, block_scores = ops.custom_deepseek_v41_index_scores_gaudi2(query, weights, cache, pages, positions,
                                                                           candidates, ratio, int(reindex), columns)
    else:
        if query.shape[0] != 1 or pages.ndim != 1:
            raise ValueError("Decoded index hot mirror belongs to one request; batch decode requires paged state")
        hot_rows = INDEX_MME_HOT_TOKENS // ratio
        if (decoded_hot.dtype != torch.bfloat16 or decoded_hot.ndim != 2 or decoded_hot.shape != (hot_rows, 128)):
            raise ValueError("MME index hot cache requires the fixed ratio-aware BF16 geometry")
        scores = _mme_hot_scores(query, weights, decoded_hot, positions, ratio)
        # Below 16K every visible block survives the candidate publication.
        # Its device-valued direct branch does not read these placeholders.
        block_scores = torch.empty((query.shape[0], hot_rows // 8), dtype=torch.float32, device=query.device)
    # The decoded hot bucket is a causal prefix shorter than 16K rows.  The
    # Full layer's candidate blocks are therefore exactly the identity block
    # prefix, so a Reindex owner can select the same logical rows directly.
    # This keeps the exact ordered result while avoiding candidate reads and
    # the badly imbalanced 16K worker partition.  Once the request leaves the
    # hot bucket ``decoded_hot`` is absent and the normal Reindex contract is
    # restored before the first consumer runs.
    select_reindex = int(reindex and decoded_hot is None)
    stats = ops.custom_deepseek_v41_index_threshold_gaudi2(scores, positions, ratio, select_reindex, 0)
    selected = ops.custom_deepseek_v41_index_emit_gaudi2(scores, positions, candidates, stats, ratio, select_reindex, 0)
    # Full output is in logical-row order. A production Reindex pool can
    # carry the same guarantee from its Full publisher; external/preserved
    # pools may be permuted. Preserve their public logical ordering unless
    # the caller supplies the stronger producer contract.
    if select_reindex and not ordered_candidates:
        selected = torch.where(selected >= 0, selected, 2147483647).sort(-1).values
        selected = torch.where(selected < 2147483647, selected, -1)
    blocks = None
    # No hot-bucket consumer observes candidate_pool: all of them use the
    # equivalent direct prefix above.  Skip both publication kernels.  At the
    # first non-hot token the source layer runs before Reindex layers and
    # refreshes candidate_pool for the same generation.
    if publish_candidates and decoded_hot is None:
        stats = ops.custom_deepseek_v41_index_threshold_gaudi2(block_scores, positions, ratio, 0, 1)
        blocks = ops.custom_deepseek_v41_index_emit_gaudi2(block_scores, positions, candidates, stats, ratio, 0, 1)
    return selected, blocks
