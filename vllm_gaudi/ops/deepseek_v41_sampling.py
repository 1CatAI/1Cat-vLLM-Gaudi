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


def partitioned_vocab_topk(scaled, width):
    """Exact top-K from eight local selections; only the final K is sorted."""
    rows, columns = scaled.shape
    partitions = 8
    local_columns = (columns + partitions * 64 - 1) // (partitions * 64) * 64
    if not 1 <= rows <= 6 or not 64 <= local_columns <= 4096 or width > local_columns:
        raise ValueError("Partitioned vocabulary selection requires C1-C6 and up to32768 vocabulary columns")
    padded = torch.nn.functional.pad(scaled, (0, partitions * local_columns - columns), value=-float("inf"))
    values, ids = torch.ops.custom_op.custom_deepseek_v41_vocab_partition_topk_gaudi2(
        padded.reshape(rows * partitions, local_columns).contiguous(), width)
    offsets = torch.arange(partitions, device=scaled.device, dtype=torch.int32).reshape(1, partitions, 1)
    ids = (ids.reshape(rows, partitions, width) + offsets * local_columns).reshape(rows, partitions * width)
    values, order = values.reshape(rows, partitions * width).topk(width, dim=-1, sorted=True)
    return values, ids.gather(-1, order.long())


def _stream_filter_partition(scaled, controls, maximum, total, width):
    """Compact a conservative score threshold; certify omissions explicitly.

    This is a selection fast path, not an unconditional approximation to q.
    Overflow or insufficient mass must use the asynchronous exact repair.
    """
    rows, columns = scaled.shape
    partitions = 8
    local_columns = (columns + partitions * 128 - 1) // (partitions * 128) * 128
    padded = torch.nn.functional.pad(scaled, (0, partitions * local_columns - columns), value=-float("inf"))
    cutoff = maximum + total.log() + ((1 - controls[:, 1:2]).clamp_min(1e-7) / (width * 2)).log()
    values, ids, counts = torch.ops.custom_op.custom_deepseek_v41_vocab_filter_gaudi2(
        padded.reshape(rows * partitions, local_columns).contiguous(), cutoff.contiguous(), width)
    offsets = torch.arange(partitions, device=scaled.device, dtype=torch.int32).reshape(1, partitions, 1)
    ids = (ids.reshape(rows, partitions, width) + offsets * local_columns).reshape(rows, partitions * width)
    values, order = values.reshape(rows, partitions * width).topk(width, dim=-1, sorted=True)
    ids = ids.gather(-1, order.long())
    omitted = torch.maximum(cutoff, values[:, -1:])
    # An overflowed shard can still be entirely below the global nucleus.
    # Its true maximum bounds every omitted score; infinity would needlessly
    # force repair for diffuse but negligible vocabulary shards.
    omitted = torch.where((counts.reshape(rows, partitions) > width).any(-1, keepdim=True), maximum, omitted)
    return values, ids, omitted


def stream_filter_candidates(scaled, controls, maximum, total, width):
    """Scan a full vocabulary with the existing bounded shard kernel.

    Its eight input planes hold at most 4096 columns each. Split larger
    vocabularies into fixed 32K pieces, then merge their compact candidates;
    all pieces use the full-row probability mass and the same cutoff.
    """
    columns = scaled.shape[-1]
    if columns <= 32768:
        return _stream_filter_partition(scaled, controls, maximum, total, width)
    values, ids, omissions = [], [], []
    for start in range(0, columns, 32768):
        scores, offsets, omitted = _stream_filter_partition(
            scaled[:, start:start + 32768].contiguous(), controls, maximum, total, width)
        values.append(scores)
        ids.append(offsets + start)
        omissions.append(omitted)
    scores, order = torch.cat(values, dim=-1).topk(width, dim=-1, sorted=True)
    selected = torch.cat(ids, dim=-1).gather(-1, order.long())
    omitted = torch.maximum(torch.stack(omissions, dim=1).amax(dim=1), scores[:, -1:])
    return scores, selected, omitted


def lane_sampling_candidates(scaled, width):
    """One score scan with certified omissions, without scalar score reloads.

    Each 64-wide lane keeps its two largest scores. A bound on every omitted
    score is carried to the existing nucleus certificate; uncertified rows
    use the same random draw in the asynchronous full-sort repair.
    """
    rows, columns = scaled.shape
    partitions = max(8, (columns + 4095) // 4096)
    partitions = (partitions + 7) // 8 * 8
    local_columns = (columns + partitions * 64 - 1) // (partitions * 64) * 64
    if scaled.device.type != "hpu" or columns < 1024:
        values, ids = scaled.topk(width, dim=-1, sorted=True)
        return values, ids, values[:, -1:]
    if not 1 <= rows <= 6 or partitions > 32 or local_columns < 128:
        raise ValueError("Lane candidate capacity requires C1-C6 and at most 131072 vocabulary columns")
    padded = torch.nn.functional.pad(scaled, (0, partitions * local_columns - columns), value=-float("inf"))
    values, ids, omitted = torch.ops.custom_op.custom_deepseek_v41_vocab_lane_candidates_gaudi2(
        padded.reshape(rows * partitions, local_columns).contiguous())
    offsets = torch.arange(partitions, device=scaled.device, dtype=torch.int32).reshape(1, partitions, 1)
    ids = (ids.reshape(rows, partitions, 128) + offsets * local_columns).reshape(rows, partitions * 128)
    values, order = values.reshape(rows, partitions * 128).topk(width, dim=-1, sorted=True)
    ids = ids.gather(-1, order.long())
    omitted = torch.maximum(omitted.reshape(rows, partitions).amax(-1, keepdim=True), values[:, -1:])
    return values, ids, omitted


def nucleus_packet_fields(packet, tp_size, width):
    """Decode old top-K and explicitly certified threshold packets."""
    prefix = packet.shape[-1] // tp_size - 2 * width
    if prefix not in (3, 4):
        raise ValueError("Nucleus packet does not match its sampling capacity")
    shards = packet.reshape(packet.shape[0], tp_size, prefix + 2 * width)
    omitted = shards[..., 3] if prefix == 4 else shards[..., 2 + width]
    return shards, shards[..., prefix:prefix + width].flatten(1), shards[..., prefix + width:].flatten(1), omitted


def local_nucleus_packet(logits, controls, tp_rank, width, *, radix=False, lane_candidates=False, threshold_state=None, shared_max=False):
    """Keep the full local partition function, exchanging only bounded candidates."""
    scaled = logits.float() / controls[:, :1].clamp_min(1e-5)
    maximum = scaled.amax(-1, keepdim=True)
    total = (scaled - maximum).exp().sum(-1, keepdim=True)
    from vllm_gaudi import envs

    stream_filter = envs.VLLM_HPU_DSV41_DSPARK and envs.VLLM_HPU_DSV41_DSPARK_STREAM_FILTER_SAMPLING
    if lane_candidates:
        values, ids, omitted = lane_sampling_candidates(scaled, width)
    elif stream_filter:
        values, ids, omitted = stream_filter_candidates(scaled, controls, maximum, total, width)
    elif envs.VLLM_HPU_DSV41_DSPARK and envs.VLLM_HPU_DSV41_DSPARK_PARTITION_RADIX_SAMPLING:
        values, ids = partitioned_vocab_topk(scaled, width)
    elif threshold_state is not None and logits.shape[0] == 1 and logits.shape[-1] > 512:
        position, scratch_ids = threshold_state
        metadata = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
            scaled.contiguous(), position, 1, 0, 0, 1)
        candidates = torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
            scaled.contiguous(), position, scratch_ids, metadata, 1, 0, 0).clamp_min(0).long()
        subset = scaled.gather(-1, candidates)
        values, offsets = subset.topk(width, dim=-1, sorted=True)
        ids = candidates.gather(-1, offsets)
        total = torch.where(metadata[:, 1:2] >= width, total, float("nan"))
    elif radix:
        values, ids = torch.ops.custom_op.custom_deepseek_v41_vocab_radix_topk_gaudi2(scaled.contiguous(), width)
        values, order = values.sort(dim=-1, descending=True)
        ids = ids.gather(-1, order)
    else:
        values, ids = scaled.topk(width, dim=-1, sorted=True)
    global_ids = ids.to(torch.float32) + tp_rank * logits.shape[-1]
    greedy = (global_ids[:, :1] if lane_candidates else
              scaled.argmax(-1, keepdim=True).float() + tp_rank * logits.shape[-1])
    if stream_filter or lane_candidates:
        return torch.cat((maximum, total, greedy, omitted, values, global_ids), -1)
    return torch.cat((maximum, total, greedy, values, global_ids), -1)


def sample_nucleus_packet(packet, controls, *, tp_size, width):
    """Return a candidate and a coverage certificate; uncertified rows must fall back.

    The full-vocabulary partition function is retained. Each shard's Kth
    score must be strictly below the last retained nucleus score. Thus every
    omitted score is outside the nucleus, including the boundary token.
    Equal retained scores fall back to preserve the existing sort's tie order.
    """
    shards, values, ids, omitted = nucleus_packet_fields(packet, tp_size, width)
    maximum = shards[..., 0].amax(-1, keepdim=True)
    total = (shards[..., 1] * (shards[..., 0] - maximum).exp()).sum(-1, keepdim=True)
    ids = ids.to(torch.int64)
    values, order = values.sort(-1, descending=True)
    ids = ids.gather(-1, order)
    probabilities = (values - maximum).exp() / total
    cumulative = probabilities.cumsum(-1)
    keep = cumulative - probabilities < controls[:, 1:2]
    offsets = torch.arange(tp_size * width, device=packet.device, dtype=torch.int32)
    keep = keep & ((controls[:, 3:4] <= 0) | (offsets < controls[:, 3:4]))
    retained = keep.sum(-1, keepdim=True).clamp_min(1)
    boundary = values.gather(-1, retained - 1)
    covered = (omitted < boundary).all(-1, keepdim=True)
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



def sample_nucleus_packet_fused(packet, controls, *, tp_size, width):
    """Same FP32 sort/exp/normalizer/cumsums; fused layout, mask and ID selection.

    The completion certificate is I32 0/1, directly consumed by the existing
    scalar status ABI. No RNG, host upload or state mutation is introduced.
    """
    if packet.shape[0] != controls.shape[0]:
        return sample_nucleus_packet(packet, controls, tp_size=tp_size, width=width)
    values, ids, maxima, totals, cuts = torch.ops.custom_op.custom_deepseek_v41_sampling_unpack_gaudi2(
        packet.contiguous(), tp_size, width)
    maximum = maxima.amax(-1, keepdim=True)
    total = (totals * (maxima - maximum).exp()).sum(-1, keepdim=True)
    values, order = values.sort(-1, descending=True)
    ids = ids.gather(-1, order)
    probabilities = (values - maximum).exp() / total
    cumulative = probabilities.cumsum(-1)
    retained, covered = torch.ops.custom_op.custom_deepseek_v41_sampling_mask_gaudi2(
        values, probabilities, cumulative, cuts, controls, total, tp_size)
    cumulative = retained.cumsum(-1)
    selected = torch.ops.custom_op.custom_deepseek_v41_sampling_select_gaudi2(
        cumulative, ids, controls, packet, maximum, tp_size)
    return selected, covered

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


def pack_sample_status(selected, covered):
    """One C1 integer for the established native token-copy ABI."""
    return selected * 2 + covered.to(torch.int32)


def unpack_sample_status(values):
    if len(values) != 1 or values[0] < 0:
        raise RuntimeError("Invalid bounded-sampling completion certificate")
    return int(values[0]) // 2, bool(int(values[0]) & 1)


def commit_replay_inputs(input_ids, positions, selected):
    """Advance private C1 replay roots after their last current-token consumer.

    These allocations belong to StageVariant, never to PositionBank or a
    scheduler request. Ordinary continuation resolves the certificate before
    replay. Device lookahead instead preserves this draw in a separate frame,
    drains a rejected invocation, and repairs it before publishing its output.
    Full sampling repair overwrites the returned token alias.
    """
    if (input_ids.dtype != torch.int32 or positions.dtype != torch.int32
            or input_ids.shape != (1,) or positions.shape != (1,)
            or selected.shape != (1, 1) or selected.dtype != torch.int32):
        raise ValueError("Replay input feedback requires private C1 I32 roots")
    next_position = positions + 1
    input_ids.copy_(selected.reshape(1))
    positions.copy_(next_position)
    return input_ids.reshape(1, 1), positions
