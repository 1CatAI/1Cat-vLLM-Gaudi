# SPDX-License-Identifier: Apache-2.0
"""Exchange index queries and head gains through one rank-ordered packet."""

import torch


def gather_index_query_packet(query, weights, gather, tp_size=4):
    """Preserve each rank's query/gain bits and the existing head order.

    The returned tensors have the same dense layouts as the two original
    gathers. Both operands belong to one score operation and have identical
    process-group and input lifetimes; no additional synchronization is needed.
    """
    batch, heads, width = query.shape
    if weights.shape != (batch, heads) or weights.dtype != query.dtype:
        raise ValueError("Index query and gains must have matching head geometry and dtype")
    columns = heads * width
    packet = torch.cat((query.reshape(batch, columns), weights), dim=-1)
    received = gather(packet, 1).reshape(batch, tp_size, columns + heads)
    queries = received[..., :columns].reshape(batch, tp_size * heads, width).contiguous()
    gains = received[..., columns:].reshape(batch, tp_size * heads).contiguous()
    return queries, gains
