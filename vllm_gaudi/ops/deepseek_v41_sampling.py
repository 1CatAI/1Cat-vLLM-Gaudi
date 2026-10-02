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


def local_nucleus_candidates(logits, controls, tp_rank, *, candidates, sorted_candidates=True):
    """Keep exact shard mass, independent of the bounded candidate selection."""
    temperature = controls[:, :1].clamp_min(1e-5)
    scaled = logits.float() / temperature
    maximum = scaled.amax(-1, keepdim=True)
    mass = (scaled - maximum).exp().sum(-1, keepdim=True)
    values, ids = scaled.topk(candidates, dim=-1, sorted=sorted_candidates)
    ids = ids.to(torch.float32) + tp_rank * logits.shape[-1]
    return torch.cat((maximum, mass, values, ids), -1)


def sample_nucleus_candidates(gathered, controls, *, candidates):
    """Return a draw and a conservative proof that no nucleus row was omitted.

    The denominator includes every vocabulary row, not just the candidates.
    Every shard's final candidate must lie strictly below the last retained
    row. Equal scores at that boundary are rejected, as are incomplete mass,
    invalid probabilities and unsupported top-k extents. A rejected draw must
    be replaced by the full sampler with the *same* control/uniform inputs.
    """
    shards = gathered.unflatten(-1, (-1, 2 + 2 * candidates))
    maximum = shards[..., 0].amax(-1, keepdim=True)
    denominator = (shards[..., 1] * (shards[..., 0] - maximum).exp()).sum(-1, keepdim=True)
    values = shards[..., 2:2 + candidates].flatten(-2)
    ids = shards[..., 2 + candidates:].flatten(-2)
    ordered, order = values.sort(-1, descending=True)
    probabilities = (ordered - maximum).exp() / denominator
    before = probabilities.cumsum(-1) - probabilities
    top_p, uniform, top_k = controls[:, 1:2], controls[:, 2:3], controls[:, 3:4]
    ordinal = torch.arange(values.shape[-1], dtype=torch.int32, device=values.device)
    keep = (before < top_p) & ((top_k <= 0) | (ordinal < top_k))
    filtered = torch.where(keep, probabilities, 0.)
    cumulative = filtered.cumsum(-1)
    selected = (cumulative < uniform * cumulative[:, -1:]).sum(-1, keepdim=True)
    selected = selected.clamp_max(values.shape[-1] - 1)
    token = ids.gather(-1, order.gather(-1, selected)).to(torch.int32)
    last_retained = torch.where(keep, ordered, float('inf')).amin(-1, keepdim=True)
    shard_cutoff = shards[..., 2:2 + candidates].amin(-1).amax(-1, keepdim=True)
    complete = (probabilities.sum(-1, keepdim=True) >= top_p) & (shard_cutoff < last_retained)
    complete = complete & torch.isfinite(denominator) & (denominator > 0) & (top_p < 1)
    # Full sort has backend-specific equal-score order. Avoid asserting exact
    # token equality when a retained tie can change inverse-CDF row ordering.
    ties = (ordered[:, 1:] == ordered[:, :-1]) & keep[:, 1:]
    complete = complete & ~ties.any(-1, keepdim=True)
    return token, complete


def sample_replay_candidates(logits, controls, tp_rank, all_gather, *, candidates=128, sorted_candidates=True):
    """One peer point for bounded sampling and the existing greedy contract.

    A token outside the vocabulary is a request-owned full-sampler fallback
    marker. It must be resolved before staging any next-token input.
    """
    packet = torch.cat((local_nucleus_candidates(logits, controls, tp_rank, candidates=candidates,
                                               sorted_candidates=sorted_candidates),
                        local_greedy_candidate(logits, tp_rank)), -1)
    shards = all_gather(packet, -1).unflatten(-1, (-1, packet.shape[-1]))
    token, complete = sample_nucleus_candidates(shards[..., :-2].flatten(-2), controls,
                                                candidates=candidates)
    greedy = select_greedy_candidate(shards[..., -2:].flatten(-2)).to(torch.int32)
    vocabulary = logits.shape[-1] * shards.shape[-2]
    sampled = token + (~complete).to(torch.int32) * vocabulary
    return torch.where(controls[:, :1] == 0, greedy, sampled)
