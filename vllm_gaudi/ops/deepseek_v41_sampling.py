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


def pack_sample_status(selected, covered):
    """One C1 integer for the established native token-copy ABI."""
    return selected * 2 + covered.to(torch.int32)


def unpack_sample_status(values):
    if len(values) != 1 or values[0] < 0:
        raise RuntimeError("Invalid bounded-sampling completion certificate")
    return int(values[0]) // 2, bool(int(values[0]) & 1)


def local_nucleus_packet(logits, controls, tp_rank, width):
    """Keep the full local partition function, exchanging only bounded candidates."""
    scaled = logits.float() / controls[:, :1].clamp_min(1e-5)
    maximum = scaled.amax(-1, keepdim=True)
    total = (scaled - maximum).exp().sum(-1, keepdim=True)
    values, ids = scaled.topk(width, dim=-1, sorted=True)
    global_ids = ids.to(torch.float32) + tp_rank * logits.shape[-1]
    greedy = scaled.argmax(-1, keepdim=True).float() + tp_rank * logits.shape[-1]
    return torch.cat((maximum, total, greedy, values, global_ids), -1)


def sample_nucleus_packet(packet, controls, *, tp_size, width):
    """Return a candidate and a coverage certificate; uncertified rows must fall back.

    The full-vocabulary partition function is retained. Each shard's Kth
    score must be strictly below the last retained nucleus score. Thus every
    omitted score is outside the nucleus, including the boundary token.
    Equal retained scores fall back to preserve the existing sort's tie order.
    """
    shards = packet.reshape(packet.shape[0], tp_size, 3 + 2 * width)
    maximum = shards[..., 0].amax(-1, keepdim=True)
    total = (shards[..., 1] * (shards[..., 0] - maximum).exp()).sum(-1, keepdim=True)
    values = shards[..., 3:3 + width].flatten(1)
    ids = shards[..., 3 + width:].flatten(1).to(torch.int64)
    values, order = values.sort(-1, descending=True)
    ids = ids.gather(-1, order)
    probabilities = (values - maximum).exp() / total
    cumulative = probabilities.cumsum(-1)
    keep = cumulative - probabilities < controls[:, 1:2]
    offsets = torch.arange(tp_size * width, device=packet.device, dtype=torch.int32)
    keep = keep & ((controls[:, 3:4] <= 0) | (offsets < controls[:, 3:4]))
    retained = keep.sum(-1, keepdim=True).clamp_min(1)
    boundary = values.gather(-1, retained - 1)
    covered = (shards[..., 2 + width] < boundary).all(-1, keepdim=True)
    ties = ((values[:, 1:] == values[:, :-1]) & keep[:, 1:]).any(-1, keepdim=True)
    covered = covered & ~ties & torch.isfinite(total) & (total > 0)
    # The unfiltered sampler visits vocabulary order, rather than sorted
    # scores. Until its separate rank/CDF path is qualified, use full fallback.
    covered = covered & ((controls[:, 1:2] < 1) | (controls[:, 3:4] > 0))
    probabilities = torch.where(keep, probabilities, 0.)
    cumulative = probabilities.cumsum(-1)
    threshold = controls[:, 2:3] * cumulative[:, -1:]
    selected = (cumulative < threshold).sum(-1, keepdim=True).clamp_max(tp_size * width - 1)
    sampled = ids.gather(-1, selected).to(torch.int32)
    greedy = torch.where(shards[..., 0] == maximum, shards[..., 2], float(2**24)).amin(-1, keepdim=True)
    greedy = greedy.to(torch.int32)
    return torch.where(controls[:, :1] == 0, greedy, sampled), covered | (controls[:, :1] == 0)


def device_sampling_controls(parameters, seed, counter):
    """A counter-based uniform draw using only fixed-address device inputs.

    Integer avalanche permutations visit every uint32 value once per seed.
    The FP32 midpoint draw uses the same open-interval convention as the
    request-owned host sampler; the reproducible sequence is device owned.
    This primitive is enabled only by the separately qualified device sampler.
    """
    value = counter ^ seed
    value = (value ^ ((value >> 16) & 0xffff)) * -2048144789
    value = (value ^ ((value >> 13) & 0x7ffff)) * -1028477387
    value = value ^ ((value >> 16) & 0xffff)
    uniform = (((value >> 8) & 0xffffff).float() + .5) * (2**-24)
    uniform = uniform.clamp(2**-24, 1 - 2**-24).reshape(-1, 1)
    controls = torch.cat((parameters[:, :2], uniform, parameters[:, 2:3]), -1)
    return controls


def device_sampling_draw(parameters, seed, counter):
    controls = device_sampling_controls(parameters, seed, counter)
    counter.copy_(counter + 1)
    return controls
