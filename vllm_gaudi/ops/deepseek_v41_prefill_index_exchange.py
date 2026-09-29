# SPDX-License-Identifier: Apache-2.0
"""Route prefill index heads directly to the query owner, without replication."""
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class PrefillIndexQueryPartition:
    query: torch.Tensor
    weights: torch.Tensor
    tokens: int
    rank: int

    def validate(self, tokens, rank):
        rows = (tokens + 3) // 4
        if (self.tokens != tokens or self.rank != rank or self.query.shape != (rows, 32, 128)
                or self.weights.shape != (rows, 32)):
            raise ValueError("Prefill index query ownership changed before selection")


def exchange_prefill_index_queries(query, weights, rank, *, group):
    """Preserve TP head order and BF16 bytes in one Q/weight exchange.

    The caller applies RoPE and the existing FP4 roundtrip before this helper.
    No arithmetic or head reduction moves across the collective. Tail queries
    are padded here; selection supplies invalid positions for the same rows.
    Each invocation owns its buffers until their ordinary device consumers end.
    """
    if (query.ndim != 3 or query.shape[1:] != (8, 128) or query.shape[0] < 1 or weights.shape != query.shape[:2]
            or query.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16 or query.device != weights.device
            or query.requires_grad or weights.requires_grad):
        raise ValueError("Prefill index exchange requires BF16 [T,8,128] Q and [T,8] weights")
    if not 0 <= rank < 4 or dist.get_world_size(group) != 4 or dist.get_rank(group) != rank:
        raise ValueError("Prefill index exchange requires the four-rank TP group and its local rank")
    tokens = query.shape[0]
    rows = (tokens + 3) // 4
    # One packet carries the already-rounded head weight with its Q vector.
    packet = torch.cat((query, weights.unsqueeze(-1)), -1)
    if rows * 4 != tokens:
        packet = F.pad(packet, (0, 0, 0, 0, 0, rows * 4 - tokens))
    received = torch.empty_like(packet)
    dist.all_to_all_single(received, packet, group=group)
    peers = received.reshape(4, rows, 8, 129)
    local_query = peers[..., :128].permute(1, 0, 2, 3).reshape(rows, 32, 128).contiguous()
    local_weights = peers[..., 128].permute(1, 0, 2).reshape(rows, 32).contiguous()
    return PrefillIndexQueryPartition(local_query, local_weights, tokens, rank)
