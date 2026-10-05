# SPDX-License-Identifier: Apache-2.0
"""Cold replication of the small BF16 index head-gain projection.

Query projection weights and score work remain sharded exactly as before.
The replica removes only the gain collective from single-token decode.
"""

import torch


def prepare_index_gain_weight(owner):
    """Materialize rank-ordered head weights before recording any replay."""
    if not owner.owns_index:
        return
    if owner._index_gain_weight is not None:
        raise ValueError("Index gain weight must be invalidated before preparing again")
    projection = owner.weights.indexer.weights_proj
    weight = projection.weight
    tp_size = owner.tensor_parallel_size
    if (tp_size < 1 or weight.ndim != 2 or weight.shape[0] != owner.index_heads
            or weight.dtype != torch.bfloat16 or getattr(projection, "bias", None) is not None
            or hasattr(projection, "scale")):
        raise ValueError("Replicated index gain requires an unscaled BF16 head projection")
    replica = weight if tp_size == 1 else owner.gather(weight.contiguous(), 0)
    if replica.shape != (owner.index_heads * tp_size, weight.shape[1]):
        raise ValueError("Index gain gather did not return rank-ordered head rows")
    owner._index_gain_weight = replica.contiguous()


def uses_replicated_index_gain(owner, value, *, prefill, request_batch, local_queries):
    # Preserve token-sharded prefill and DSpark/request-batch graph contracts.
    return (not prefill and not request_batch and not local_queries and value.shape[0] == 1
            and getattr(owner, "_index_gain_weight", None) is not None)
