# SPDX-License-Identifier: Apache-2.0
"""Exact TP greedy selection using one score/token pair per vocabulary shard."""

import torch


def local_greedy_candidate(logits, tp_rank):
    local_ids = logits.argmax(-1, keepdim=True)
    scores = logits.gather(-1, local_ids)
    global_ids = local_ids.to(torch.float32) + tp_rank * logits.shape[-1]
    return torch.cat((scores, global_ids), dim=-1)


def select_greedy_candidate(gathered):
    # TP shards are ordered by rank, so equal scores retain the lowest global
    # token ID. Both reductions use the backend's existing argmax primitive.
    candidates = gathered.unflatten(-1, (-1, 2))
    winner = candidates[..., 0].argmax(-1, keepdim=True)
    return candidates[..., 1].gather(-1, winner).to(torch.int64)
