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


def _mme_hot_scores(query, weights, decoded_keys, positions, ratio, local_heads):
    """Retain the BF16 boundary of each TP shard's head sum on MME.

    ``decoded_keys`` follows the compressed row geometry, so ratio-2 Full
    layers execute half as many MME columns as ratio-1 layers while retaining
    the same token-valued visibility boundary in the native reducer.
    """
    raw_scores = torch.einsum("bhd,nd->bhn", query, decoded_keys).contiguous()
    # Keep the large dot product on MME and fuse the following ReLU, per-head
    # weight, and the two checkpoint BF16 shard boundaries into one TPC pass.
    # The native reducer also writes -inf past the device-valued visible row,
    # avoiding all generic reduction intermediates without changing selection.
    return torch.ops.custom_op.custom_deepseek_v41_index_reduce_bf16_gaudi2(
        raw_scores, weights, positions, ratio, local_heads
    )


def ordered_index_ids(scores, positions, candidates, ratio, *, reindex=False, blocks=False):
    """Shared threshold/bitmap emission with TP2 C1's ordered tie rule."""
    ops = torch.ops.custom_op
    scores = scores.contiguous()
    metadata = ops.custom_deepseek_v41_index_threshold_gaudi2(scores, positions, ratio, int(reindex), int(blocks))
    return ops.custom_deepseek_v41_index_emit_gaudi2(
        scores, positions, candidates, metadata, ratio, int(reindex), int(blocks)
    )


def _bounded_mme_scores(
    query, weights, cache, pages, positions, candidates, ratio, local_heads, search_rows, reindex, decoded_keys
):
    """Use the same paged-key/MME producer for any TP shard geometry."""
    if reindex:
        rows = candidates.unsqueeze(-1) * 8 + torch.arange(8, dtype=torch.int32, device=positions.device)
        rows = torch.where(candidates.unsqueeze(-1) >= 0, rows, -1).flatten(1)
    else:
        rows = torch.arange(search_rows, dtype=torch.int32, device=positions.device)
    if not reindex and decoded_keys is not None:
        from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_source_scores

        scores = mirror_source_scores(query, weights, decoded_keys, positions, rows, ratio, local_heads)
    else:
        from vllm_gaudi.ops.deepseek_v41_decode_index import native_index_tile
        from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_index_tile

        score = native_index_tile if decoded_keys is None else mirror_index_tile
        scores = torch.cat(
            [
                (
                    score(query, weights, cache, pages, positions, rows[..., first : first + 2048], ratio, local_heads)
                    if decoded_keys is None
                    else score(
                        query, weights, decoded_keys, positions, rows[..., first : first + 2048], ratio, local_heads
                    )
                )
                for first in range(0, rows.shape[-1], 2048)
            ],
            -1,
        ).contiguous()
    return scores


def runtime_index_select(
    query,
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
    ordered_candidates=False,
    local_heads=16,
    search_rows=None,
    decoded_keys=None,
):
    """Return ordered top-512 rows and, for a Full layer, top-2048 blocks.

    Tensor shapes remain fixed while the native kernels derive useful work
    from ``positions``.  Padding outside the visible prefix is never eligible.
    """
    if ratio not in (1, 2) or capacity < 512 or capacity > 1048576 or capacity % 64:
        raise ValueError("Runtime indexer requires ratio 1/2 and a bounded aligned capacity")
    if reindex and publish_candidates:
        raise ValueError("A candidate consumer cannot publish a new candidate pool")
    if local_heads < 1 or query.shape[1] % local_heads:
        raise ValueError("Index heads must divide into complete TP shards")
    ops = torch.ops.custom_op
    columns = 16384 if reindex else capacity
    if search_rows is not None:
        scores = _bounded_mme_scores(
            query, weights, cache, pages, positions, candidates, ratio, local_heads, search_rows, reindex, decoded_keys
        )
        block_scores = None
        if publish_candidates:
            block_scores = scores.reshape(scores.shape[0], -1, 8).amax(-1)
            block_ids = torch.arange(block_scores.shape[-1], dtype=torch.int32, device=positions.device)
            newest = (((positions + 1) // ratio - 1) // 8).unsqueeze(-1)
            block_scores = block_scores.masked_fill(block_ids == newest, torch.inf)
    elif decoded_hot is None:
        scores, block_scores = ops.custom_deepseek_v41_index_scores_gaudi2(
            query, weights, cache, pages, positions, candidates, ratio, int(reindex), columns, local_heads
        )
    else:
        if query.shape[0] != 1 or pages.ndim != 1:
            raise ValueError("Decoded index hot mirror belongs to one request; batch decode requires paged state")
        hot_rows = INDEX_MME_HOT_TOKENS // ratio
        if decoded_hot.dtype != torch.bfloat16 or decoded_hot.ndim != 2 or decoded_hot.shape != (hot_rows, 128):
            raise ValueError("MME index hot cache requires the fixed ratio-aware BF16 geometry")
        scores = _mme_hot_scores(query, weights, decoded_hot, positions, ratio, local_heads)
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
    select_reindex = reindex and (search_rows is not None or decoded_hot is None)
    selected = ordered_index_ids(scores, positions, candidates, ratio, reindex=select_reindex)
    # The emitter partitions the score vector by increasing source offset and
    # starts every worker at the exact prefix count of retained entries.  Full
    # output is therefore already in increasing logical-row order.  Reindex
    # candidate blocks are also published in increasing logical order, so its
    # mapped rows retain that order.  Sorting either result repeats work and
    # adds a bitonic-sort kernel to every selection layer.
    if select_reindex and not ordered_candidates:
        selected = torch.where(selected >= 0, selected, 2147483647).sort(-1).values
        selected = torch.where(selected < 2147483647, selected, -1)
    blocks = None
    # No hot-bucket consumer observes candidate_pool: all of them use the
    # equivalent direct prefix above.  Skip both publication kernels.  At the
    # first non-hot token the source layer runs before Reindex layers and
    # refreshes candidate_pool for the same generation.
    if publish_candidates and (search_rows is not None or decoded_hot is None):
        blocks = ordered_index_ids(block_scores, positions, candidates, ratio, blocks=True)
    return selected, blocks
