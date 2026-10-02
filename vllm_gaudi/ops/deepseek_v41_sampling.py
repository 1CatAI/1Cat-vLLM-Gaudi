# SPDX-License-Identifier: Apache-2.0
"""TP sampling with request-owned randomness and device probability filtering."""

import hashlib

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


def request_uniform(req_id, seed, ordinal):
    """All TP ranks derive the same fresh draw without replaying a captured RNG."""
    owner = str(seed) if seed is not None else req_id
    digest = hashlib.blake2b(f"{owner}:{ordinal}".encode(), digest_size=8).digest()
    # Midpoints avoid exact zero and one, including after conversion to FP32.
    value = (int.from_bytes(digest, "big") >> 12) + 0.5
    return min(max(value / (1 << 52), 2**-24), 1 - 2**-24)


def sample_probabilities(logits, controls, *, filtered):
    """Inverse-CDF temperature/top-p/top-k sampling; controls are mutable inputs.

    Each row contains temperature, nucleus probability, uniform draw and top-k.
    Padded and greedy rows remain valid in a mixed request bucket.
    """
    temperature, top_p, uniform, top_k = controls.unbind(-1)
    scaled = logits.float() / temperature.clamp_min(1e-5).unsqueeze(-1)
    if filtered:
        scaled, order = scaled.sort(dim=-1, descending=True)
    probabilities = scaled.softmax(-1)
    if filtered:
        cumulative = probabilities.cumsum(-1)
        keep = (cumulative - probabilities) < top_p.unsqueeze(-1)
        indices = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.int32)
        keep = keep & ((top_k.unsqueeze(-1) <= 0) | (indices < top_k.unsqueeze(-1)))
        probabilities = torch.where(keep, probabilities, 0.0)
    cumulative = probabilities.cumsum(-1)
    threshold = uniform.unsqueeze(-1) * cumulative[..., -1:]
    selected = (cumulative < threshold).sum(-1, keepdim=True).clamp_max(logits.shape[-1] - 1)
    if filtered:
        selected = order.gather(-1, selected)
    return torch.where(temperature.unsqueeze(-1) == 0, logits.argmax(-1, keepdim=True), selected).to(torch.int32)


def commit_sampled_token(selected, record):
    updated = torch.cat((record[:1] + 1, torch.ones_like(record[1:3]), selected.reshape(1).to(torch.int32)))
    record.copy_(updated)
    return record
