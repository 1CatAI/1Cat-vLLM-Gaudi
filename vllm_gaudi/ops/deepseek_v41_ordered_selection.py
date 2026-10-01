# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for tools using the shared native runtime indexer."""
from vllm_gaudi.ops.deepseek_v41_indexer import ordered_index_ids as ordered_index_ids, runtime_index_select


def ordered_index_select(owner, positions, q, weights, candidates):
    """Delegate to the shared scorer with the owner's actual shard geometry."""
    return runtime_index_select(
        q.contiguous(), weights.contiguous(), owner.cache.index, owner.shared.block_table,
        positions, candidates, ratio=owner.ratio, capacity=owner.length // owner.ratio,
        reindex=owner.layer > owner.candidate_source,
        publish_candidates=owner.layer == owner.candidate_source, local_heads=owner.index_heads,
        search_rows=owner.decode_visible_rows or owner.search_length // owner.ratio,
        decoded_keys=owner.cache.index_mirror if owner._uses_index_mirror(q) else None)
