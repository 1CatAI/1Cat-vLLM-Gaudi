# SPDX-License-Identifier: Apache-2.0
"""Probability-correct fixed-capacity speculative rejection control.

Inputs are the actual normalized proposal and target distributions after
temperature/nucleus filtering, not confidence-head scores or greedy matches.
Request-owned uniforms must be fresh mutable inputs to a captured graph.
"""
import torch


class SpeculativeRequestSampling:
    """Fixed-address probability/RNG state belonging to one request.

    Request admission creates this state once. Binding a different request
    must bind its own tensors; no request-global q or RNG counter is shared.
    This is not enabled by the runner until sampled round capture, fallback
    and cancellation/reorder have passed qualification together.
    """

    def __init__(self, parameters, seed, local_vocab, device):
        if len(parameters) != 3 or local_vocab <= 0:
            raise ValueError("Speculative sampling needs temperature/top_p/top_k and a vocabulary shard")
        self.parameters = torch.tensor(parameters, dtype=torch.float32, device=device).reshape(1, 3).repeat(11, 1)
        signed_seed = (int(seed) + (1 << 31)) % (1 << 32) - (1 << 31)
        self.seed = torch.tensor([signed_seed], dtype=torch.int32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.offsets = torch.arange(11, dtype=torch.int32, device=device)
        self.proposal = torch.zeros((5, local_vocab), dtype=torch.float32, device=device)
        self.proposal_valid = torch.zeros(1, dtype=torch.bool, device=device)

    def next_draws(self):
        return speculative_sampling_draws(self.parameters, self.seed, self.counter, self.offsets)


def speculative_sampling_draws(parameters, seed, counter, offsets):
    """Explicit mutable request inputs for replay-safe independent draws."""
    from vllm_gaudi.ops.deepseek_v41_sampling import device_sampling_controls

    controls = device_sampling_controls(parameters, seed, counter + offsets)
    counter.copy_(counter + 11)
    return controls[6:], controls[:5, 2], controls[5:6, 2], controls[:6]


def sample_proposal_sharded(logits, controls, *, tp_rank, tp_size, width, all_gather):
    """Sample and retain the exact probability of the actual bounded proposal.

    A proposal may be truncated when its nucleus does not fit the packet.
    This is a different q, so the returned normalized local fragment must
    accompany the token into rejection. Target p cannot use this fallback.
    FP32 scores use a bit-preserving wire, matching verify collectives.
    """
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet

    packet = local_nucleus_packet(logits, controls, tp_rank, width)
    if packet.device.type == "hpu":
        packet = all_gather(packet.contiguous().view(torch.bfloat16), dim=-1).contiguous().view(torch.float32)
    else:
        packet = all_gather(packet, dim=-1)
    token, covered = sample_nucleus_packet(packet, controls, tp_size=tp_size, width=width)
    probability, _ = bounded_proposal_distribution(packet, controls, tp_rank=tp_rank, tp_size=tp_size,
                                                   width=width, local_vocab=logits.shape[-1])
    offset = token.long() - tp_rank * logits.shape[-1]
    owned = (offset >= 0) & (offset < logits.shape[-1])
    greedy = torch.zeros_like(probability).scatter(-1, offset.clamp(0, logits.shape[-1] - 1), owned.float())
    probability = torch.where(controls[:, :1] == 0, greedy, probability)
    return token.reshape(-1), probability, covered.reshape(-1)


def bounded_proposal_distribution(packet, controls, *, tp_rank, tp_size, width, local_vocab):
    """Actual proposal probabilities on one vocabulary shard.

    Reuse the C1 packet's full partition function and nucleus boundary. An
    uncertified packet remains a valid truncated *proposal* distribution;
    it cannot be used as the target distribution. Returning its actual mass
    is essential for unbiased rejection, including omitted or tied scores.
    """
    from vllm_gaudi.ops.deepseek_v41_sampling import nucleus_packet_fields

    shards, values, ids, omitted = nucleus_packet_fields(packet, tp_size, width)
    maximum = shards[..., 0].amax(-1, keepdim=True)
    total = (shards[..., 1] * (shards[..., 0] - maximum).exp()).sum(-1, keepdim=True)
    ids = ids.to(torch.int64)
    values, order = values.sort(-1, descending=True)
    ids = ids.gather(-1, order)
    probability = (values - maximum).exp() / total
    cumulative = probability.cumsum(-1)
    keep = cumulative - probability < controls[:, 1:2]
    lane = torch.arange(tp_size * width, dtype=torch.int32, device=packet.device)
    keep = keep & ((controls[:, 3:4] <= 0) | (lane < controls[:, 3:4]))
    retained = keep.sum(-1, keepdim=True).clamp_min(1)
    boundary = values.gather(-1, retained - 1)
    covered = (omitted < boundary).all(-1, keepdim=True)
    ties = ((values[:, 1:] == values[:, :-1]) & keep[:, 1:]).any(-1, keepdim=True)
    covered = covered & ~ties & torch.isfinite(total) & (total > 0)
    covered = covered & ((controls[:, 1:2] < 1) | (controls[:, 3:4] > 0))
    probability = torch.where(keep, probability, 0.)
    probability = probability / probability.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    offset = ids - tp_rank * local_vocab
    owned = (offset >= 0) & (offset < local_vocab)
    # Scatter-add makes masked foreign candidates harmless even when their
    # clamped destination equals a genuine local token. Scatter overwrite
    # would silently lose that token's proposal probability.
    local = torch.zeros((packet.shape[0], local_vocab), dtype=torch.float32, device=packet.device)
    local = local.scatter_add(-1, offset.clamp(0, local_vocab - 1), torch.where(owned, probability, 0.))
    return local, covered


def filtered_distribution(logits, temperature, top_p, top_k):
    """Return the normalized distribution used by an actual sampled proposal.

    Controls have one entry per row. Retain the first token crossing the
    nucleus threshold, matching the ordinary inverse-CDF sampler. Keep the
    distribution in vocabulary order for rejection/correction; confidence
    scores cannot substitute for these probabilities.
    """
    scaled = logits.float() / temperature.clamp_min(1e-5).unsqueeze(-1)
    sorted_logits, order = scaled.sort(dim=-1, descending=True)
    probabilities = sorted_logits.softmax(-1)
    cumulative = probabilities.cumsum(-1)
    keep = (cumulative - probabilities) < top_p.unsqueeze(-1)
    lane = torch.arange(logits.shape[-1], dtype=torch.int32, device=logits.device)
    keep = keep & ((top_k.unsqueeze(-1) <= 0) | (lane < top_k.unsqueeze(-1)))
    retained = torch.where(keep, probabilities, 0.0)
    retained = retained / retained.sum(-1, keepdim=True)
    filtered = torch.zeros_like(retained).scatter(-1, order, retained)
    greedy = torch.zeros_like(filtered).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.where(temperature.unsqueeze(-1) == 0, greedy, filtered)


def full_sampling_peer_logits(logits, tp_rank, tp_size):
    """Exact large fallback on the existing bounded native peer transport.

    Bound each transfer to the replay transport's established wire capacity.
    Reassemble in vocabulary rank order, not transfer order. This is used
    only by the separately gated full repair, never by covered sampling.
    """
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives

    _, gather = stage_collectives(tp_rank, True, tp_size, native_fp32_gather=True)
    columns = 16384 // logits.shape[0] // 64 * 64
    parts = []
    widths = []
    for first in range(0, logits.shape[-1], columns):
        value = logits[:, first:first + columns].float().contiguous()
        widths.append(value.shape[-1])
        if value.device.type == "hpu":
            wire_view = torch.ops.custom_op.custom_deepseek_v41_sampling_wire_view_gaudi2
            wire = wire_view(value, False).reshape(1, -1)
            wire_elements = wire.numel()
            wire = torch.nn.functional.pad(wire, (0, -wire_elements % 128))
            shards = torch.ops.vllm_gaudi.tp_peer_allgather(wire, tp_size).reshape(tp_size, wire.numel())
            parts.append(torch.cat(tuple(wire_view(shards[rank, :wire_elements].contiguous().reshape(
                value.shape[0], value.shape[1] * 2), True) for rank in range(tp_size)), -1))
        else:
            parts.append(gather(value, dim=-1))
    return torch.cat(tuple(torch.cat(tuple(value[:, rank * width:(rank + 1) * width]
                                          for value, width in zip(parts, widths, strict=True)), -1)
                           for rank in range(tp_size)), -1)


def full_sampling_distribution_sharded(logits, controls, *, tp_rank, all_gather,
                                       return_coverage=False, force_legacy=False, known_stochastic=False,
                                       weighted=False):
    """Exact fallback p/q with the same filtering as ordinary sampling.

    Transport FP32 logits without BF16 rounding. This large exchange is an
    explicit fallback, not a bounded packet or a native small-message plan.
    Each owner keeps only its normalized probability shard afterwards.
    """
    from vllm_gaudi import envs

    width = envs.VLLM_HPU_DSV41_DSPARK_SAMPLING_WIDTH
    if width not in (64, 128, 256):
        raise ValueError("DSpark bounded sampling width must be 64, 128 or 256")
    if (envs.VLLM_HPU_DSV41_DSPARK_TP_PACKET_TARGET and logits.device.type == "hpu"
            and logits.shape[0] == 6 and return_coverage and known_stochastic and not force_legacy
            and not weighted):
        from vllm.distributed import get_tensor_model_parallel_world_size

        return sample_target_packet_main(logits, controls, tp_rank=tp_rank,
                                         tp_size=get_tensor_model_parallel_world_size())
    if envs.VLLM_HPU_DSV41_DSPARK_SHARDED_BOUNDED_SAMPLING and not force_legacy:
        from vllm.distributed import get_tensor_model_parallel_world_size

        values = sample_bounded_or_full_sharded(logits, controls, tp_rank=tp_rank,
                                             tp_size=get_tensor_model_parallel_world_size(), all_gather=all_gather,
                                             width=width, radix=envs.VLLM_HPU_DSV41_DSPARK_RADIX_SAMPLING)
        return (*values, torch.ones_like(values[0], dtype=torch.bool)) if return_coverage else values
    if (logits.device.type == "hpu" and envs.VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR
            and not envs.VLLM_HPU_DSV41_DSPARK_FULL_HCCL_MAIN):
        from vllm.distributed import get_tensor_model_parallel_world_size

        full = full_sampling_peer_logits(logits, tp_rank, get_tensor_model_parallel_world_size())
    elif logits.device.type == "hpu":
        if envs.VLLM_HPU_DSV41_DSPARK_FULL_HCCL_MAIN:
            # The additive wire-view node keeps the exact FP32 bit transport
            # inside a recipe; aten.view.dtype cannot cross native recipes.
            wire_view = torch.ops.custom_op.custom_deepseek_v41_sampling_wire_view_gaudi2
            wire = wire_view(logits.float().contiguous(), False)
            if envs.VLLM_HPU_DSV41_DSPARK_LARGE_HCCL:
                # The DSpark-private bridge only widens stock AllGather's
                # capacity check. Its normal and replayed transport are HCCL.
                from vllm.distributed import get_tensor_model_parallel_world_size

                tp_size = get_tensor_model_parallel_world_size()
                gathered = torch.ops.vllm_gaudi.tp_peer_allgather(wire.reshape(1, -1).contiguous(), tp_size)
                full_wire = gathered.reshape(tp_size, *wire.shape).permute(1, 0, 2).contiguous()
                full = wire_view(full_wire.reshape(wire.shape[0], -1), True)
            else:
                full = wire_view(all_gather(wire, dim=-1).contiguous(), True)
        else:
            full = all_gather(logits.float().contiguous().view(torch.bfloat16), dim=-1)
            full = full.contiguous().view(torch.float32)
    else:
        full = all_gather(logits.float(), dim=-1)
    if weighted and not force_legacy:
        token, probability, certificate = weighted_sampling_distribution(full, controls)
        start = tp_rank * logits.shape[-1]
        values = (token, probability[:, start:start + logits.shape[-1]].contiguous())
        return (*values, certificate.reshape(-1)) if return_coverage else values
    if envs.VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING and not force_legacy:
        # The already-qualified global-logit transport is unchanged. Run only
        # the existing bounded CDF; serving repairs an uncertified transaction
        # asynchronously with its original RNG counter and exact full sampler.
        fused_local = envs.VLLM_HPU_DSV41_DSPARK_FUSED_BOUNDED_NUCLEUS
        token, probability, certificate = bounded_sampling_parts(
            full, controls, width=width, owned_shard=(tp_rank, logits.shape[-1]) if fused_local else None)
        start = tp_rank * logits.shape[-1]
        values = (token, probability if fused_local else probability[:, start:start + logits.shape[-1]].contiguous())
        return (*values, certificate.reshape(-1)) if return_coverage else values
    if logits.device.type == "hpu" and envs.VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS and not force_legacy:
        probability, token, certificate = torch.ops.custom_op.custom_deepseek_v41_nucleus_mass_sample_gaudi2(
            full.contiguous(), controls.contiguous())
        start = tp_rank * logits.shape[-1]
        values = (token, probability[:, start:start + logits.shape[-1]].contiguous())
        return (*values, certificate.bool()) if return_coverage else values
    if weighted and force_legacy and envs.VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF:
        # Repair shares the proposal's vocabulary-order draw. Changing only
        # the covered arm's draw map would bias retries selected by later
        # Markov-dependent coverage failures.
        token, probability = sample_full_distribution(
            full, controls, known_stochastic=known_stochastic, vocabulary_cdf=True)
        start = tp_rank * logits.shape[-1]
        values = token, probability[:, start:start + logits.shape[-1]].contiguous()
        return (*values, torch.ones_like(token, dtype=torch.bool)) if return_coverage else values
    token, probability = (sample_bounded_or_full_distribution(
        full, controls, width=width, radix=envs.VLLM_HPU_DSV41_DSPARK_RADIX_SAMPLING)
                          if envs.VLLM_HPU_DSV41_DSPARK_BOUNDED_SAMPLING and not force_legacy else
                          sample_full_distribution(full, controls, known_stochastic=known_stochastic))
    start = tp_rank * logits.shape[-1]
    values = (token, probability[:, start:start + logits.shape[-1]].contiguous())
    return (*values, torch.ones_like(token, dtype=torch.bool)) if return_coverage else values


def sample_target_packet_main(logits, controls, *, tp_rank, tp_size, width=128):
    """Reuse C1 partition packets for official C6 p; q and repair are unchanged.

    Only the covered main plan uses this function. Every uncertain score or
    probability boundary goes through the existing asynchronous exact repair
    with the original RNG counter and journal. No full sort is queued here.
    """
    from vllm_gaudi.ops.deepseek_v41_sampling import (
        local_nucleus_packet, nucleus_packet_fields, sample_nucleus_packet,
    )

    packet = local_nucleus_packet(logits, controls, tp_rank, width, lane_candidates=True)
    wire_view = torch.ops.custom_op.custom_deepseek_v41_sampling_wire_view_gaudi2
    wire = wire_view(packet.contiguous(), False)
    elements = wire.numel()
    flat = torch.nn.functional.pad(wire.reshape(1, -1), (0, -elements % 128))
    shards = torch.ops.vllm_gaudi.tp_peer_allgather(flat.contiguous(), tp_size).reshape(tp_size, flat.numel())
    ordered = shards[:, :elements].reshape(tp_size, *wire.shape).permute(1, 0, 2).contiguous()
    packet = wire_view(ordered.reshape(wire.shape[0], -1), True)
    token, covered = sample_nucleus_packet(packet, controls, tp_size=tp_size, width=width)
    probability, proposal_covered = bounded_proposal_distribution(
        packet, controls, tp_rank=tp_rank, tp_size=tp_size, width=width, local_vocab=logits.shape[-1])
    # The C1 packet certificate is augmented at the exact nucleus boundary:
    # a conservative margin prevents a partition-function rounding change
    # from moving the retained row across top_p without exact repair.
    shards, values, _, _ = nucleus_packet_fields(packet, tp_size, width)
    maximum = shards[..., 0].amax(-1, keepdim=True)
    total = (shards[..., 1] * (shards[..., 0]-maximum).exp()).sum(-1, keepdim=True)
    values = values.sort(-1, descending=True).values
    mass = (values-maximum).exp()/total
    cumulative = mass.cumsum(-1)
    keep = cumulative-mass < controls[:, 1:2]
    retained_mass = torch.where(keep, mass, 0.).sum(-1, keepdim=True)
    boundary = mass.gather(-1, keep.sum(-1, keepdim=True).clamp_min(1)-1)
    margin = 2e-6
    safe = ((retained_mass >= controls[:, 1:2]+margin)
            & (retained_mass-boundary <= controls[:, 1:2]-margin)
            & (cumulative[:, -1:] >= controls[:, 1:2]+margin)
            & (controls[:, :1] > 0) & (controls[:, 1:2] > 0) & (controls[:, 1:2] < 1)
            & (controls[:, 3:4] <= 0))
    return token.reshape(-1), probability, (covered & proposal_covered & safe).reshape(-1)


def sample_bounded_or_full_sharded(logits, controls, *, tp_rank, tp_size, all_gather, width=256, radix=False):
    """Reuse C1 packets; compute each partition once and retain exact p/q.

    This candidate needs a changing-coverage collective/backend gate before
    service qualification. Uncovered proposals use the original full q too.
    """
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet

    def gather(value):
        if value.device.type == "hpu":
            return all_gather(value.contiguous().view(torch.bfloat16), dim=-1).contiguous().view(torch.float32)
        return all_gather(value, dim=-1)

    packet = gather(local_nucleus_packet(logits, controls, tp_rank, width, radix=radix))
    token, covered = sample_nucleus_packet(packet, controls, tp_size=tp_size, width=width)
    probability, probability_covered = bounded_proposal_distribution(
        packet, controls, tp_rank=tp_rank, tp_size=tp_size, width=width, local_vocab=logits.shape[-1])
    certified = covered & probability_covered & (controls[:, :1] > 0)
    token = token.reshape(-1)

    def bounded(local, settings, selected, distribution):
        return selected.clone(), distribution.clone()

    def fallback(local, settings, selected, distribution):
        full = gather(local.float())
        selected, distribution = sample_full_distribution(full, settings)
        start = tp_rank * local.shape[-1]
        return selected, distribution[:, start:start + local.shape[-1]].contiguous()

    return torch.cond(certified.all(), bounded, fallback, (logits, controls, token, probability))


def weighted_sampling_distribution(logits, controls):
    """Select exact score boundaries by weighted radix, with a repair certificate.

    The two quantiles locate the nucleus and the ordinary score-order draw.
    This preserves the provided draw, rather than replacing it with another
    sampling order. A score tie at either boundary requires exact sort repair.
    All radix stages remain in the caller's native captured protocol.
    """
    scaled = (logits.float() / controls[:, :1].clamp_min(1e-5)).contiguous()
    from vllm_gaudi import envs

    probability = (torch.ops.custom_op.custom_deepseek_v41_vocab_softmax_f32_gaudi2(scaled)
                   if envs.VLLM_HPU_DSV41_DSPARK_VOCAB_SOFTMAX and scaled.device.type == "hpu"
                   else scaled.softmax(-1))
    ops = torch.ops.custom_op

    def cut(weights, target, active_mask=None):
        prefix = torch.zeros((scaled.shape[0], 1), dtype=torch.int32, device=scaled.device)
        remaining = target.contiguous()
        from vllm_gaudi import envs

        sparse = envs.VLLM_HPU_DSV41_DSPARK_WEIGHTED_SPARSE_BINS
        bounds = ops.custom_deepseek_v41_weighted_score_bounds_gaudi2(scaled, weights) if sparse else None
        for shift in range(28, -1, -4):
            partial = (
                ops.custom_deepseek_v41_weighted_sparse_bins_gaudi2(
                    scaled, weights, prefix, active_mask, shift)
                if active_mask is not None else
                ops.custom_deepseek_v41_weighted_sparse_bins_gaudi2(
                    scaled, weights, prefix,
                    ops.custom_deepseek_v41_weighted_active_blocks_gaudi2(bounds, prefix, shift), shift)
                if sparse and shift <= 16
                else ops.custom_deepseek_v41_weighted_mass_bins_gaudi2(scaled, weights, prefix, shift)
            )
            prefix, remaining = ops.custom_deepseek_v41_weighted_mass_advance_gaudi2(
                partial, prefix, remaining, shift)
        return prefix, remaining

    nucleus, _ = cut(probability, controls[:, 1:2])
    from vllm_gaudi import envs

    if envs.VLLM_HPU_DSV41_DSPARK_WEIGHTED_STATIC_MASK:
        kept, boundary_counts, boundary_ids, active_mask = ops.custom_deepseek_v41_weighted_finish_mask_gaudi2(
            scaled, probability, nucleus)
    else:
        kept, boundary_counts, boundary_ids = ops.custom_deepseek_v41_weighted_mass_finish_gaudi2(
            scaled, probability, nucleus)
        active_mask = None
    mass = kept.sum(-1, keepdim=True)
    if envs.VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF:
        normalized = kept / mass
        information = ops.custom_deepseek_v41_probability_draw_gaudi2(
            normalized.contiguous(), controls.contiguous())
        boundary_probability = probability.gather(1, boundary_ids.amax(-1, keepdim=True).clamp_min(0).long())
        margin = 4e-7
        certified = (boundary_counts.sum(-1, dtype=torch.int32) == 1) & information[:, 1].bool()
        certified = certified & torch.isfinite(mass.reshape(-1)) & (mass.reshape(-1) > 0)
        certified = certified & (controls[:, 0] > 0) & (controls[:, 1] > 0) & (controls[:, 1] < 1)
        certified = certified & (controls[:, 3] <= 0)
        safe_mass = (mass >= controls[:, 1:2] + margin) & (
            mass - boundary_probability <= controls[:, 1:2] - margin)
        # An uncertified transaction is already allowed to enqueue a draft
        # and a Target lookahead before host repair. Keep its placeholder
        # inside the embedding vocabulary even when the draw returns V.
        # The false certificate is retained; this word is never committed.
        token = information[:, 0].clamp(0, logits.shape[-1] - 1).contiguous()
        return token, normalized, certified & safe_mass.reshape(-1)
    selected, draw_remaining = cut(kept, controls[:, 2:3] * mass, active_mask)
    _, draw_counts, ids = ops.custom_deepseek_v41_weighted_mass_finish_gaudi2(scaled, kept, selected)
    token = ids.amax(-1)
    boundary_probability = probability.gather(1, boundary_ids.amax(-1, keepdim=True).clamp_min(0).long())
    draw_probability = kept.gather(1, token.reshape(-1, 1).clamp_min(0).long())
    certified = (boundary_counts.sum(-1, dtype=torch.int32) == 1) & (
        draw_counts.sum(-1, dtype=torch.int32) == 1)
    certified = certified & torch.isfinite(mass.reshape(-1)) & (mass.reshape(-1) > 0)
    certified = certified & (controls[:, 0] > 0) & (controls[:, 1] > 0) & (controls[:, 1] < 1)
    certified = certified & (controls[:, 3] <= 0)
    # Sorting and tiled mass reductions have different FP32 summation trees.
    # Repair near either boundary instead of silently accepting that rounding.
    margin = 4e-7
    safe_mass = (mass >= controls[:, 1:2] + margin) & (
        mass - boundary_probability <= controls[:, 1:2] - margin)
    safe_draw = (draw_remaining > margin) & (draw_remaining < draw_probability - margin)
    certified = certified & safe_mass.reshape(-1) & safe_draw.reshape(-1)
    return token, kept / mass, certified


def full_vocabulary_cdf_draw(probability, uniform):
    """Repair using the same prefix scan for both mass and selection."""
    cdf = probability.cumsum(-1)
    desired = uniform * cdf[:, -1:]
    # Zero uniform chooses the first positive word, not a leading zero.
    selected = ((cdf < desired) | (cdf == 0)).sum(-1)
    return selected.clamp_max(probability.shape[-1] - 1).to(torch.int32)


def sample_full_distribution(logits, controls, *, known_stochastic=False, vocabulary_cdf=False):
    """Return an ordinary sampling draw and its actual p/q with one sort.

    Select with the unnormalized retained CDF, exactly as the ordinary
    sampler does. Normalize only the vocabulary-order probabilities used
    by rejection. This avoids sorting the same logits a second time.
    """
    temperature, top_p, uniform, top_k = controls.unbind(-1)
    scaled = logits.float() / temperature.clamp_min(1e-5).unsqueeze(-1)
    sorted_logits, order = scaled.sort(dim=-1, descending=True)
    probabilities = sorted_logits.softmax(-1)
    cumulative = probabilities.cumsum(-1)
    lane = torch.arange(logits.shape[-1], dtype=torch.int32, device=logits.device)
    keep = ((cumulative - probabilities) < top_p.unsqueeze(-1)) & (
        (top_k.unsqueeze(-1) <= 0) | (lane < top_k.unsqueeze(-1)))
    retained = torch.where(keep, probabilities, 0.)
    if known_stochastic:
        # Both filters retain a prefix of the sorted probabilities. Reuse
        # its already-computed scan, clamping the suffix to the retained
        # boundary; do not scan 129280 entries a second time. The actual
        # p/q normalization below remains the original reduction.
        boundary = keep.sum(-1, keepdim=True).clamp_min(1) - 1
        cumulative = torch.minimum(cumulative, cumulative.gather(-1, boundary))
    else:
        cumulative = retained.cumsum(-1)
    threshold = uniform.unsqueeze(-1) * cumulative[..., -1:]
    selected = (cumulative < threshold).sum(-1, keepdim=True).clamp_max(logits.shape[-1] - 1)
    selected = order.gather(-1, selected)
    normalized = retained / retained.sum(-1, keepdim=True)
    probability = torch.zeros_like(normalized).scatter(-1, order, normalized)
    if vocabulary_cdf:
        if not known_stochastic:
            greedy = torch.zeros_like(probability).scatter(-1, logits.argmax(-1, keepdim=True), 1.)
            probability = torch.where(temperature.unsqueeze(-1) == 0, greedy, probability)
        # Only DSpark's proposal calls this alternative inverse-CDF map.
        # Target p and the non-speculative production sampler stay unchanged.
        if probability.device.type == "hpu":
            # This is the exact repair, not the certified main draw above.
            # The tiled draw can return V when its part reduction and local
            # prefix disagree at a boundary. Reusing it here discards its
            # certificate and may feed V into the next Markov embedding.
            # Use one consistent full CDF/total for repair; q and the saved
            # uniform are unchanged. The covered main path stays bounded.
            token = torch.ops.custom_op.custom_deepseek_v41_probability_full_draw_gaudi2(
                probability.contiguous(), controls.contiguous())
            return token, probability
        cdf = probability.double().cumsum(-1)
        desired = controls[:, 2:3].double() * cdf[:, -1:]
        selected = (cdf < desired).sum(-1).clamp_max(logits.shape[-1] - 1).to(torch.int32)
        return selected, probability
    if known_stochastic:
        # The sampled runner dispatches here only for temperature>0.
        # Retain the full sort/filter and RNG, omit the dead greedy scan
        # and its full-vocabulary one-hot scatter. Generic callers retain it.
        return selected.to(torch.int32).reshape(-1), probability
    greedy_id = logits.argmax(-1, keepdim=True)
    token = torch.where(temperature.unsqueeze(-1) == 0, greedy_id, selected).to(torch.int32)
    greedy = torch.zeros_like(probability).scatter(-1, greedy_id, 1.)
    probability = torch.where(temperature.unsqueeze(-1) == 0, greedy, probability)
    return token.reshape(-1), probability


def sample_speculative_prefix_sharded(target, proposal, proposed, acceptance_uniforms, correction_uniform,
                                      tp_rank, tp_size, all_gather, proposal_count=5):
    """Exact rejection over vocabulary shards with scalar-only exchanges.

    Each rank retains its actual normalized p/q fragment. The existing
    collective owns transport; this adds no new communication implementation.
    FP32 words use the same bit-preserving BF16 wire as the current argmax.
    """
    local_vocab = target.shape[-1]

    def gather(value):
        if value.device.type == "hpu":
            return all_gather(value.contiguous().view(torch.bfloat16), dim=1).contiguous().view(torch.float32)
        return all_gather(value, dim=1)

    offset = proposed.long() - tp_rank * local_vocab
    owned = (offset >= 0) & (offset < local_vocab)
    index = offset.clamp(0, local_vocab - 1).reshape(5, 1)
    p = torch.where(owned, target[:5].gather(-1, index).squeeze(-1), 0.)
    q = torch.where(owned, proposal.gather(-1, index).squeeze(-1), 0.)
    pairs = gather(torch.stack((p, q), -1)).reshape(5, tp_size, 2).sum(1)
    p, q = pairs.unbind(-1)
    ratio = (p / q.clamp_min(torch.finfo(q.dtype).tiny)).clamp_max(1)
    lane = torch.arange(5, dtype=torch.int64, device=target.device)
    accepted = torch.where(acceptance_uniforms < ratio, 5, lane).amin().clamp_max(proposal_count).reshape(1)
    row = accepted.reshape(1, 1).expand(1, local_vocab)
    correction = target.gather(0, row).squeeze(0)
    rejected_q = proposal.gather(0, row.clamp_max(4)).squeeze(0)
    residual = torch.where(accepted < proposal_count, (correction - rejected_q).clamp_min(0), correction)
    cumulative = residual.cumsum(-1)
    masses = gather(cumulative[-1:].reshape(1, 1)).reshape(tp_size)
    prefix = masses.cumsum(0)
    total = prefix[-1:]
    threshold = correction_uniform * total
    owner = (prefix < threshold).sum().clamp_max(tp_size - 1)
    before = prefix[tp_rank:tp_rank + 1] - masses[tp_rank:tp_rank + 1]
    local_token = (cumulative < threshold - before).sum().clamp_max(local_vocab - 1)
    token = torch.where(owner == tp_rank, local_token + tp_rank * local_vocab, -1)
    token = gather(token.reshape(1, 1).to(target.dtype)).amax().long()
    output_lane = torch.arange(6, dtype=torch.int64, device=target.device)
    candidates = torch.cat((proposed.long(), token.reshape(1)))
    output = torch.where(output_lane < accepted, candidates,
                         torch.where(output_lane == accepted, token, -1))
    valid = ((q > 0) | (lane >= proposal_count)).all() & (total > 0).all() & torch.isfinite(total).all()
    return output, accepted + 1, valid.reshape(1)


def target_coverage_for_commit(covered, committed):
    """Certify exactly the p rows that supplied acceptance and correction.

    After the first rejection, later speculative p rows cannot influence
    the published prefix. The bonus row is required only after all proposals
    were accepted. Earlier uncertified rows always force exact repair.
    """
    covered = covered.reshape(-1)
    count = committed.reshape(1)
    lane = torch.arange(covered.numel(), dtype=count.dtype, device=covered.device)
    return (covered | (lane >= count)).all() & (count >= 1).all() & (count <= covered.numel()).all()


def sample_speculative_prefix(target, proposal, proposed, acceptance_uniforms, correction_uniform):
    """Accept a longest proposal prefix, then sample the correction/bonus.

    Target has six rows; proposal has five. All rows share a vocabulary.
    The first rejection uses ``(p-q).clamp_min(0)``; accepting all proposals
    samples from the sixth target row. No data-dependent host branch exists.
    EOS and remaining output budget are handled by the ordinary commit owner.
    """
    selected = proposed.long().reshape(5, 1)
    p = target[:5].gather(-1, selected).squeeze(-1)
    q = proposal.gather(-1, selected).squeeze(-1)
    ratio = (p / q.clamp_min(torch.finfo(q.dtype).tiny)).clamp_max(1)
    lane = torch.arange(5, dtype=torch.int64, device=target.device)
    accepted = torch.where(acceptance_uniforms < ratio, 5, lane).amin().reshape(1)
    row = accepted.reshape(1, 1).expand(1, target.shape[-1])
    correction = target.gather(0, row).squeeze(0)
    rejected_q = proposal.gather(0, row.clamp_max(4)).squeeze(0)
    residual = torch.where(accepted < 5, (correction - rejected_q).clamp_min(0), correction)
    cumulative = residual.cumsum(-1)
    mass = cumulative[-1:]
    token = (cumulative < correction_uniform * mass).sum().clamp_max(target.shape[-1] - 1)
    output_lane = torch.arange(6, dtype=torch.int64, device=target.device)
    candidates = torch.cat((proposed.long(), token.reshape(1)))
    output = torch.where(output_lane < accepted, candidates,
                         torch.where(output_lane == accepted, token, -1))
    valid = (q > 0).all() & (mass > 0).all()
    return output, accepted + 1, valid.reshape(1)


def bounded_sampling_parts(logits, controls, width=256, *, radix=False, owned_shard=None):
    """Exact candidate scores plus a conservative nucleus coverage certificate."""
    scaled = logits.float() / controls[:, :1].clamp_min(1e-5)
    maximum = scaled.amax(-1, keepdim=True)
    total = (scaled - maximum).exp().sum(-1, keepdim=True)
    from vllm_gaudi import envs

    omitted = None
    if envs.VLLM_HPU_DSV41_DSPARK_LANE_CANDIDATES:
        from vllm_gaudi.ops.deepseek_v41_sampling import lane_sampling_candidates

        values, order, omitted = lane_sampling_candidates(scaled, width)
    elif envs.VLLM_HPU_DSV41_DSPARK_STREAM_FILTER_SAMPLING:
        from vllm_gaudi.ops.deepseek_v41_sampling import stream_filter_candidates

        values, order, omitted = stream_filter_candidates(scaled, controls, maximum, total, width)
    elif radix:
        values, order = torch.ops.custom_op.custom_deepseek_v41_vocab_radix_topk_gaudi2(
            scaled.contiguous(), min(width, logits.shape[-1]))
        values, permutation = values.sort(dim=-1, descending=True)
        order = order.gather(-1, permutation).long()
    else:
        values, order = scaled.topk(min(width, logits.shape[-1]), dim=-1, sorted=True)
    if envs.VLLM_HPU_DSV41_DSPARK_FUSED_BOUNDED_NUCLEUS:
        if width != 64:
            raise ValueError("Fused bounded nucleus requires its qualified K64 row")
        omitted = values[:, -1:] if omitted is None else omitted
        columns = logits.shape[-1] if owned_shard is None else owned_shard[1]
        offset = 0 if owned_shard is None else owned_shard[0] * columns
        information, probability = torch.ops.custom_op.custom_deepseek_v41_bounded_local_nucleus_gaudi2(
            values.contiguous(), order.to(torch.int32).contiguous(), maximum.contiguous(), total.contiguous(),
            controls.contiguous(), omitted.contiguous(), columns, offset)
        return information[:, 0].contiguous(), probability, information[:, 1:2].bool()
    probability = (values - maximum).exp() / total
    cumulative = probability.cumsum(-1)
    lane = torch.arange(values.shape[-1], dtype=torch.int32, device=logits.device)
    keep = ((cumulative - probability) < controls[:, 1:2]) & (
        (controls[:, 3:4] <= 0) | (lane < controls[:, 3:4]))
    retained = keep.sum(-1, keepdim=True).clamp_min(1)
    boundary = values.gather(-1, retained - 1)
    omitted = values[:, -1:] if omitted is None else omitted
    covered = (omitted < boundary) & (cumulative[:, -1:] >= controls[:, 1:2] + 1e-6)
    # The complete sort remains authoritative at ties and near the probability
    # boundary. Greedy/unfiltered controls retain their established path too.
    ties = ((values[:, 1:] == values[:, :-1]) & keep[:, 1:]).any(-1, keepdim=True)
    covered = covered & ~ties & torch.isfinite(total) & (total > 0)
    covered = covered & (controls[:, :1] > 0) & (controls[:, 1:2] < 1)
    retained_probability = torch.where(keep, probability, 0.)
    cdf = retained_probability.cumsum(-1)
    threshold = controls[:, 2:3] * cdf[:, -1:]
    selected = (cdf < threshold).sum(-1, keepdim=True).clamp_max(values.shape[-1] - 1)
    token = order.gather(-1, selected).to(torch.int32).reshape(-1)
    normalized = retained_probability / retained_probability.sum(-1, keepdim=True)
    probability = torch.zeros_like(logits, dtype=torch.float32).scatter(-1, order, normalized)
    return token, probability, covered


def sample_bounded_or_full_distribution(logits, controls, width=256, *, radix=False):
    """Retain the complete-sort fallback; native branch replay must be qualified."""
    token, probability, covered = bounded_sampling_parts(logits, controls, width, radix=radix)

    def bounded(x, settings, selected, distribution):
        return selected.clone(), distribution.clone()

    def fallback(x, settings, selected, distribution):
        return sample_full_distribution(x, settings)

    return torch.cond(covered.all(), bounded, fallback, (logits, controls, token, probability))
