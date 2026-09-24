# SPDX-License-Identifier: Apache-2.0
"""Bounded, expert-grouped prefill on the native decode runtime.

The compatibility path buckets routes on the host. The optional device path
uses occupancy-independent descriptors and never reads routes back to CPU.
Compiled BMM bodies reuse decoded weights across rows. Neither path drops
routes, including an entirely skewed router. Resident N256 weights and the
C1 implementation are unchanged.
"""

from __future__ import annotations

import functools
from types import FunctionType

import numpy as np
import torch
import torch.nn.functional as F

from vllm_gaudi import envs
from vllm_gaudi.ops.deepseek_v41_route_blocks import device_hybrid_route_blocks, device_route_blocks


def route_batches(ids: np.ndarray, experts: int, max_experts: int = 8, max_rows: int = 8192):
    """Return bounded expert IDs, padded route slots, and valid local slots."""
    if ids.ndim != 2 or ids.shape[1] != 6 or ids.size == 0:
        raise ValueError("V4.1 prefill routing must be nonempty [tokens,6]")
    flat = ids.reshape(-1).astype(np.int64, copy=False)
    if np.any(flat < 0) or np.any(flat >= experts):
        raise ValueError("V4.1 prefill contains an out-of-range expert ID")
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=experts)
    ends = np.cumsum(counts)
    buckets = {}
    for expert in np.flatnonzero(counts):
        slots = order[ends[expert] - counts[expert]:ends[expert]]
        # Repeated expert IDs are legal to this adapter. Even that worst case
        # is split without losing routes or allocating an oversized bucket.
        for start in range(0, slots.size, max_rows):
            part = slots[start:start + max_rows]
            capacity = max(16, 1 << (part.size - 1).bit_length())
            buckets.setdefault(capacity, []).append((expert, part))
    for capacity, rows in sorted(buckets.items()):
        width = min(max_experts, max_rows // capacity)
        for start in range(0, len(rows), width):
            batch = rows[start:start + width]
            slots = np.full((len(batch), capacity), -1, dtype=np.int64)
            expert_ids = np.empty((1, len(batch)), dtype=np.int32)
            for index, (expert, part) in enumerate(batch):
                expert_ids[0, index] = expert
                slots[index, :part.size] = part
            valid = np.flatnonzero(slots.reshape(-1) >= 0).astype(np.int64)
            yield expert_ids, slots, valid


def prefill_decode(expert_ids, q16, s16, lookup, normal_scales):
    if envs.VLLM_HPU_DSV41_PREFILL_FAST_DEQUANT:
        return torch.ops.custom_op.custom_deepseek_v41_prefill_weight_bf16_gaudi2(
            expert_ids, q16, s16, lookup, normal_scales, envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN
            and envs.VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY)
    return torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(expert_ids, q16, s16, lookup, normal_scales)


def grouped_body(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales: bool):
    """One compiled gather -> decode -> W13 -> weighted SwiGLU -> W2 chain."""
    groups, capacity = slots.shape
    safe_slots = slots.clamp(min=0)
    selected = value.index_select(0, (safe_slots // 6).reshape(-1))
    selected = selected.reshape(groups, capacity, value.shape[-1])
    weight13 = prefill_decode(expert_ids, q13, s13, lookup, normal_scales)
    projected = torch.bmm(selected, weight13)
    middle_width = projected.shape[-1] // 2
    gate = projected[..., :middle_width].float().clamp(max=10.0)
    up = projected[..., middle_width:].float().clamp(-10.0, 10.0)
    route = routing.reshape(-1).index_select(0, safe_slots.reshape(-1))
    route = route.reshape(groups, capacity).masked_fill(slots < 0, 0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    # Materialize the same BF16 boundary used by the native C1 compound op.
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, capacity, middle_width)
    weight2 = prefill_decode(expert_ids, q2, s2, lookup, normal_scales)
    return torch.bmm(middle, weight2).reshape(-1, value.shape[-1])


def grouped_project(selected, route, expert_ids, q13, q2, s13, s2, lookup, normal_scales: bool):
    """Fixed-shape decode/MME body after route gathering.

    Keeping the scheduler token count outside this graph gives serving a
    finite set of eight recipes (one through eight expert blocks).  A new
    prompt tail can therefore reuse a prepared body instead of compiling a
    large graph beside the resident model and 1M KV allocation.
    """
    groups, capacity, hidden = selected.shape
    weight13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(expert_ids, q13, s13, lookup,
                                                                               normal_scales)
    projected = torch.bmm(selected, weight13)
    middle_width = projected.shape[-1] // 2
    gate = projected[..., :middle_width].float().clamp(max=10.0)
    up = projected[..., middle_width:].float().clamp(-10.0, 10.0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, capacity, middle_width)
    weight2 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(expert_ids, q2, s2, lookup, normal_scales)
    return torch.bmm(middle, weight2).reshape(-1, hidden)


def grouped_fp8_linear(value, expert_ids, q, scales, channel, lookup):
    """Decode one FP8 weight per expert and feed a large-M FP8 MME."""
    groups, rows, inner = value.shape
    # Empty fixed-capacity descriptors carry -1. Their route weights are zero,
    # but the static recipe still needs an in-range immutable weight binding.
    safe_ids = expert_ids.clamp(min=0)
    weight = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2(
        safe_ids, q, scales, lookup, True)
    quantized, activation_scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(
        value.reshape(groups * rows, inner))
    weight_scale = channel.index_select(0, safe_ids.reshape(-1).long()).reshape(
        groups, 1, weight.shape[-1]).float()
    return torch.ops.hpu.fp8_gemm_v2(
        quantized.reshape(groups, rows, inner), False, weight, False, None,
        torch.bfloat16, activation_scale.reshape(groups, rows, 1), weight_scale,
        None, False)


def grouped_fp8_w13_dual(value, expert_ids, q, scales, channel, lookup):
    """Use two FP8 activation terms with one exact MXFP4->FP8 W13 decode.

    The prepared channel powers represent the source MXFP4 values exactly in
    Gaudi2 E4M3. The second term recovers activation bits lost by a single
    per-row FP8 quantizer; both products accumulate before the BF16 boundary.
    """
    groups, rows, inner = value.shape
    high, high_scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(
        value.reshape(groups * rows, inner))
    high = high.reshape(groups, rows, inner)
    high_scale = high_scale.reshape(groups, rows, 1)
    recovered = torch.ops.hpu.cast_from_fp8(
        high, torch.ones((), device=value.device, dtype=torch.float32), torch.bfloat16)
    residual = (value.float() - recovered.float() * high_scale).to(torch.bfloat16)
    low, low_scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(
        residual.reshape(groups * rows, inner))
    low = low.reshape(groups, rows, inner)
    low_scale = low_scale.reshape(groups, rows, 1)
    return grouped_fp8_w13_terms(high, high_scale, low, low_scale, expert_ids, q, scales, channel, lookup)


def grouped_fp8_w13_terms(high, high_scale, low, low_scale, expert_ids, q, scales, channel, lookup,
                          packed=False):
    groups, rows = high.shape[:2]
    safe_ids = expert_ids.clamp(min=0)
    weight = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2(
        safe_ids, q, scales, lookup, True)
    weight_scale = channel.index_select(0, safe_ids.reshape(-1).long()).reshape(
        groups, 1, weight.shape[-1]).float()
    if packed:
        joined = torch.cat((high, low), dim=1)
        joined_scale = torch.cat((high_scale, low_scale), dim=1)
        projected = torch.ops.hpu.fp8_gemm_v2(joined, False, weight, False, None,
                                               torch.float32, joined_scale, weight_scale, None, False)
        return (projected[:, :rows] + projected[:, rows:]).to(torch.bfloat16)
    first = torch.ops.hpu.fp8_gemm_v2(high, False, weight, False, None,
                                      torch.float32, high_scale, weight_scale, None, False)
    second = torch.ops.hpu.fp8_gemm_v2(low, False, weight, False, None,
                                       torch.float32, low_scale, weight_scale, None, False)
    return (first + second).to(torch.bfloat16)


def dual_prequant(value):
    """Quantize each input token once, before its six expert routes diverge."""
    rows, inner = value.shape
    high, high_scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(value)
    recovered = torch.ops.hpu.cast_from_fp8(
        high, torch.ones((), device=value.device, dtype=torch.float32), torch.bfloat16)
    residual = (value.float() - recovered.float() * high_scale).to(torch.bfloat16)
    low, low_scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(residual)
    return high.reshape(rows, inner), high_scale.reshape(rows, 1), low.reshape(rows, inner), low_scale.reshape(rows, 1)


def single_prequant(value):
    """Quantize each input token once before its six expert routes diverge."""
    rows, inner = value.shape
    high, high_scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(value)
    return high.reshape(rows, inner), high_scale.reshape(rows, 1)


@functools.lru_cache(maxsize=32)
def compiled_single_prequant(signature):
    entry = FunctionType(single_prequant.__code__.replace(co_name=f"prefill_single_prequant_{signature}"),
                         single_prequant.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_dual_prequant(signature):
    entry = FunctionType(dual_prequant.__code__.replace(co_name=f"prefill_dual_prequant_{signature}"),
                         dual_prequant.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def grouped_project_fp8(selected, route, expert_ids, q13, q2, s13, s2, channel13, channel2, lookup,
                        normal_scales: bool, mode: str):
    """Grouped mixed-precision body used only after component qualification."""
    groups, capacity, hidden = selected.shape
    if mode == "w13_dual":
        projected = grouped_fp8_w13_dual(selected, expert_ids, q13, s13, channel13, lookup)
    elif mode in ("both", "w13"):
        projected = grouped_fp8_linear(selected, expert_ids, q13, s13, channel13, lookup)
    else:
        weight13 = prefill_decode(expert_ids, q13, s13, lookup, normal_scales)
        projected = torch.bmm(selected, weight13)
    middle_width = projected.shape[-1] // 2
    gate = projected[..., :middle_width].float().clamp(max=10.0)
    up = projected[..., middle_width:].float().clamp(-10.0, 10.0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, capacity, middle_width)
    if mode in ("both", "w2"):
        output = grouped_fp8_linear(middle, expert_ids, q2, s2, channel2, lookup)
    else:
        weight2 = prefill_decode(expert_ids, q2, s2, lookup, normal_scales)
        output = torch.bmm(middle, weight2)
    return output.reshape(-1, hidden)


def grouped_body_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales, routed, indices):
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    output = grouped_body(value, routing, local_slots, local_experts, q13, q2, s13, s2, lookup, normal_scales)
    # Native sparse writes ignore -1 padding and emit a completion token.
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(routed, output,
                                                                              local_slots.flatten().int())


def grouped_body_fp8_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                           channel13, channel2, mode, indices):
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    selected, route = grouped_gather(value, routing, local_slots)
    output = grouped_project_fp8(selected, route, local_experts, q13, q2, s13, s2, channel13, channel2, lookup,
                                 normal_scales, mode)
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(routed, output,
                                                                              local_slots.flatten().int())


def grouped_body_dual_prequant_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales,
                                     routed, channel13, high, high_scale, low, low_scale, indices):
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    groups, rows = local_slots.shape
    safe_slots = local_slots.clamp(min=0)
    token_rows = (safe_slots.reshape(-1) // 6).long()
    selected_high = high.index_select(0, token_rows).reshape(groups, rows, value.shape[-1])
    selected_low = low.index_select(0, token_rows).reshape(groups, rows, value.shape[-1])
    selected_high_scale = high_scale.index_select(0, token_rows).reshape(groups, rows, 1)
    selected_low_scale = low_scale.index_select(0, token_rows).reshape(groups, rows, 1)
    projected = grouped_fp8_w13_terms(selected_high, selected_high_scale, selected_low, selected_low_scale,
                                       local_experts, q13, s13, channel13, lookup, packed=True)
    width = projected.shape[-1] // 2
    gate = projected[..., :width].float().clamp(max=10.0)
    up = projected[..., width:].float().clamp(-10.0, 10.0)
    route = routing.reshape(-1).index_select(0, safe_slots.reshape(-1))
    route = route.reshape(groups, rows).masked_fill(local_slots < 0, 0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, rows, width)
    weight2 = prefill_decode(local_experts, q2, s2, lookup, normal_scales)
    output = torch.bmm(middle, weight2).reshape(-1, value.shape[-1])
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(routed, output,
                                                                              local_slots.flatten().int())


def grouped_body_single_prequant_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales,
                                       routed, channel13, high, high_scale, indices):
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    groups, rows = local_slots.shape
    safe_slots = local_slots.clamp(min=0)
    token_rows = (safe_slots.reshape(-1) // 6).long()
    selected_high = high.index_select(0, token_rows).reshape(groups, rows, value.shape[-1])
    selected_scale = high_scale.index_select(0, token_rows).reshape(groups, rows, 1)
    safe_experts = local_experts.clamp(min=0)
    weight13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2(
        safe_experts, q13, s13, lookup, True)
    weight_scale = channel13.index_select(0, safe_experts.reshape(-1).long()).reshape(
        groups, 1, weight13.shape[-1]).float()
    projected = torch.ops.hpu.fp8_gemm_v2(selected_high, False, weight13, False, None,
                                          torch.bfloat16, selected_scale, weight_scale, None, False)
    width = projected.shape[-1] // 2
    gate = projected[..., :width].float().clamp(max=10.0)
    up = projected[..., width:].float().clamp(-10.0, 10.0)
    route = routing.reshape(-1).index_select(0, safe_slots.reshape(-1))
    route = route.reshape(groups, rows).masked_fill(local_slots < 0, 0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, rows, width)
    weight2 = prefill_decode(local_experts, q2, s2, lookup, normal_scales)
    output = torch.bmm(middle, weight2).reshape(-1, value.shape[-1])
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(routed, output,
                                                                              local_slots.flatten().int())


@functools.lru_cache(maxsize=32)
def compiled_write_body_single_prequant(signature):
    entry = FunctionType(
        grouped_body_single_prequant_write.__code__.replace(co_name=f"prefill_write_single_{signature}"),
        grouped_body_single_prequant_write.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_write_body_dual_prequant(signature):
    entry = FunctionType(grouped_body_dual_prequant_write.__code__.replace(co_name=f"prefill_write_dual_{signature}"),
                         grouped_body_dual_prequant_write.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_write_body(signature):
    entry = FunctionType(grouped_body_write.__code__.replace(co_name=f"prefill_write_{signature}"),
                         grouped_body_write.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_write_body_fp8(signature):
    entry = FunctionType(grouped_body_fp8_write.__code__.replace(co_name=f"prefill_write_fp8_{signature}"),
                         grouped_body_fp8_write.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=128)
def compiled_body(signature):
    # Distinct occupancy buckets own distinct code objects, as do the model's
    # existing static layer groups. Routing values remain device inputs.
    entry = FunctionType(grouped_body.__code__.replace(co_name=f"grouped_prefill_{signature}"),
                         grouped_body.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=16)
def compiled_project(signature):
    entry = FunctionType(grouped_project.__code__.replace(co_name=f"grouped_project_{signature}"),
                         grouped_project.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=16)
def compiled_project_fp8(signature):
    entry = FunctionType(grouped_project_fp8.__code__.replace(co_name=f"grouped_project_fp8_{signature}"),
                         grouped_project_fp8.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def grouped_gather(value, routing, slots):
    """Build one canonical activation/route block without decoding weights."""
    groups, rows = slots.shape
    safe_slots = slots.clamp(min=0)
    selected = value.index_select(0, (safe_slots // 6).reshape(-1)).reshape(groups, rows, value.shape[-1])
    route = routing.reshape(-1).index_select(0, safe_slots.reshape(-1)).reshape(groups, rows)
    return selected, route.masked_fill(slots < 0, 0)


@functools.lru_cache(maxsize=64)
def compiled_gather(signature):
    entry = FunctionType(grouped_gather.__code__.replace(co_name=f"grouped_gather_{signature}"),
                         grouped_gather.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def prepare_device_grouped_prefill_recipes(normal_scales: bool = True):
    """Compile the finite expert bodies before model and KV residency.

    This runs once per serving rank from ``load_model``.  Temporary tensors
    match the runtime N256-v3 layout and are released before checkpoint
    weights are allocated.  Disk cache then makes subsequent starts hits.
    """
    if not (envs.VLLM_HPU_DSV41_PREFILL_GROUPED and envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES):
        return
    # ``load_model`` deliberately calls this before model construction.  The
    # entrypoint has selected and fingerprinted the library by then, but no
    # model module has necessarily registered its torch custom-op schemas.
    if not hasattr(torch.ops.custom_op, "custom_deepseek_v41_expert_n256_bf16_gaudi2"):
        import os
        torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    rows, experts, hidden = envs.VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS, 384, 5120
    if rows != 128:
        return
    device = torch.device("hpu")
    q13 = torch.empty((experts, 9, 327680), dtype=torch.int16, device=device)
    q2 = torch.empty((experts, 20, 73728), dtype=torch.int16, device=device)
    s13 = torch.empty((experts, 9, 40960), dtype=torch.int16, device=device)
    s2 = torch.empty((experts, 20, 9216), dtype=torch.int16, device=device)
    channel13 = torch.empty((experts, 9, 256), dtype=torch.bfloat16, device=device)
    channel2 = torch.empty((experts, 20, 256), dtype=torch.bfloat16, device=device)
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    lookup = mxfp4_bf16_lut(device)
    # Serving calls the body below ``execute_model``'s inference-mode
    # boundary.  Dynamo guards that global mode, and it also distinguishes
    # tensors allocated inside inference mode from ordinary tensors.  The
    # resident weights and lookup are loaded/created outside that boundary,
    # while selected activations, routes and expert descriptors are produced
    # inside it.  Reproduce that mixed contract here; otherwise the first
    # live prompt retraces the bodies instead of reusing startup preparation.
    with torch.inference_mode():
        for groups in range(1, 9):
            selected = torch.empty((groups, rows, hidden), dtype=torch.bfloat16, device=device)
            route = torch.empty((groups, rows), dtype=torch.float32, device=device)
            expert_ids = torch.zeros((1, groups), dtype=torch.int32, device=device)
            mode = envs.VLLM_HPU_DSV41_PREFILL_GROUPED_FP8
            if mode:
                if mode not in ("w13", "w13_dual", "w13_dual_prequant", "w13_single_prequant", "w2", "both"):
                    raise ValueError("Grouped Prefill FP8 mode must be w13, w13_dual, w13_dual_prequant, "
                                     "w13_single_prequant, w2 or both")
                if mode in ("w13_dual_prequant", "w13_single_prequant"):
                    # This mode consumes the pre-quantized per-token operands
                    # through its own route-write recipe below.
                    continue
                output = compiled_project_fp8((groups, rows, experts, bool(normal_scales), mode))(
                    selected, route, expert_ids, q13, q2, s13, s2, channel13, channel2, lookup,
                    bool(normal_scales), mode)
            else:
                output = compiled_project((groups, rows, experts, bool(normal_scales)))(selected, route, expert_ids,
                                                                                        q13, q2, s13, s2, lookup,
                                                                                        bool(normal_scales))
            torch.hpu.synchronize()
            del selected, route, expert_ids, output
        # Normal serving admits C8192 but executes it as a finite set of exact
        # model buckets.  Compile every descriptor/gather/scatter/reduction
        # shape before model loading so the live request reuses prepared
        # recipes. Operator-contract failures must still be diagnosed from
        # the compiler log, not inferred from resident-memory size.
        for tokens in (2048, 1024, 512, 256, 128):
            ids = (torch.arange(tokens * 6, dtype=torch.int32, device=device).remainder(experts).reshape(tokens, 6))
            routing = torch.full((tokens, 6), 1.0 / 6.0, dtype=torch.float32, device=device)
            value = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
            expert_ids, slots, _, _ = compiled_routes((tokens, experts, rows, False))(ids, experts, rows, False)
            torch.hpu.synchronize()
            for start in range(0, slots.shape[0], 8):
                groups = min(8, slots.shape[0] - start)
                group_slots = slots[start:start + groups].clone()
                selected, route = compiled_gather((tokens, groups, rows, hidden))(value, routing, group_slots)
                destinations = compiled_destinations((groups, rows, tokens * 6))(group_slots, tokens * 6)
                padded = torch.empty((tokens * 6 + 1, hidden), dtype=torch.bfloat16, device=device)
                projected = torch.empty((groups * rows, hidden), dtype=torch.bfloat16, device=device)
                padded = compiled_scatter((tokens, groups, rows, hidden))(padded, destinations, projected)
                torch.hpu.synchronize()
                del group_slots, selected, route, destinations, padded, projected
            ordered = torch.empty((tokens, 6, hidden), dtype=torch.bfloat16, device=device)
            reduced = compiled_reduce((tokens, hidden))(ordered)
            torch.hpu.synchronize()
            del ids, routing, value, expert_ids, slots, ordered, reduced
    del q13, q2, s13, s2, channel13, channel2, lookup
    import gc
    gc.collect()


def ordered_reduce(value):
    total = value[:, 0].float()
    for index in range(1, 6):
        total = total + value[:, index].float()
    return total.to(torch.bfloat16)


@functools.lru_cache(maxsize=128)
def compiled_reduce(signature):
    # Prompt lengths must not compete for Dynamo's per-code recompile limit.
    # As with grouped_body, each static shape owns its compiled code object.
    entry = FunctionType(ordered_reduce.__code__.replace(co_name=f"grouped_reduce_{signature}"),
                         ordered_reduce.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_routes(signature):
    entry = FunctionType(device_route_blocks.__code__.replace(co_name=f"prefill_routes_{signature}"),
                         device_route_blocks.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_hybrid_routes(signature):
    entry = FunctionType(device_hybrid_route_blocks.__code__.replace(co_name=f"prefill_hybrid_routes_{signature}"),
                         device_hybrid_route_blocks.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_destinations(signature):

    def destination_slots(slots, routes):
        # Every valid route has exactly one writer. Padding writes only an
        # unused sentinel row, never a live route or a neighbouring request.
        return torch.where(slots >= 0, slots, routes).flatten()

    entry = FunctionType(destination_slots.__code__.replace(co_name=f"route_output_{signature}"),
                         destination_slots.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def grouped_scatter_(padded, destinations, output):
    """Write one expert group into the canonical route buffer in place.

    Habana eager lowers ``index_copy_`` through a compiled graph. Leaving it
    outside the finite prefill recipe bundle makes the first real C2048 prompt
    compile beside the resident model. Keep the write as a bounded recipe so
    the large route buffer is neither cloned nor retained by the expert MME
    graph.
    """
    padded.index_copy_(0, destinations, output)
    return padded


@functools.lru_cache(maxsize=64)
def compiled_scatter(signature):
    entry = FunctionType(grouped_scatter_.__code__.replace(co_name=f"route_scatter_{signature}"),
                         grouped_scatter_.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def run_device_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales,
                               channel13=None, channel2=None):
    if torch.compiler.is_compiling():
        raise RuntimeError("Prefill group submission must remain outside a C1 graph")
    if value.ndim != 2 or ids.shape != (value.shape[0], 6) or routing.shape != ids.shape:
        raise ValueError("V4.1 device prefill requires matching [tokens,6] routing")
    rows = envs.VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS
    if rows not in (32, 64, 128, 512):
        raise ValueError("Prefill expert rows must be 32, 64, 128 or 512")
    mark_empty = envs.VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY
    fp8_mode = envs.VLLM_HPU_DSV41_PREFILL_GROUPED_FP8
    if fp8_mode not in ("", "w13", "w13_dual", "w13_dual_prequant", "w13_single_prequant", "w2", "both"):
        raise ValueError("Grouped Prefill FP8 mode must be empty, w13, w13_dual, w13_dual_prequant, "
                         "w13_single_prequant, w2 or both")
    if fp8_mode and (not isinstance(channel13, torch.Tensor) or not isinstance(channel2, torch.Tensor)):
        raise ValueError("Grouped Prefill FP8 requires both prepared channel-scale tensors")
    if fp8_mode in ("w13_dual_prequant", "w13_single_prequant") and not envs.VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS:
        raise ValueError("Pre-quantized W13 requires hybrid route plans")
    if envs.VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS:
        if (rows != 128 or fp8_mode not in ("", "w13_dual", "w13_dual_prequant", "w13_single_prequant")
                or not envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN
                or not envs.VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT
                or not envs.VLLM_HPU_DSV41_PREFILL_FAST_DEQUANT
                or not envs.VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY):
            raise ValueError("Hybrid prefill rows require BF16 or W13 FP8 128-row native route-output plans")
        if not fp8_mode:
            from vllm_gaudi.ops.deepseek_v41_prefill_buckets import run_bucketed_prefill
            return run_bucketed_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales)
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import execute_prefill_experts, routed_workspace
        base_ids, base_slots, tail_ids, tail_slots, occupied = compiled_hybrid_routes(
            (ids.shape[0], q13.shape[0], rows, 64))(ids, q13.shape[0], rows, 64)
        base_active, tail_active = (int(value) for value in occupied.cpu().tolist())
        routed = routed_workspace(value, ids.numel())
        if fp8_mode == "w13_dual_prequant":
            high, high_scale, low, low_scale = compiled_dual_prequant((value.shape[0], value.shape[-1]))(value)
            base_body = compiled_write_body_dual_prequant((value.shape[0], rows, q13.shape[0]))
            base_args = (value, routing, base_slots, base_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                         channel13, high, high_scale, low, low_scale)
        elif fp8_mode == "w13_single_prequant":
            high, high_scale = compiled_single_prequant((value.shape[0], value.shape[-1]))(value)
            base_body = compiled_write_body_single_prequant((value.shape[0], rows, q13.shape[0]))
            base_args = (value, routing, base_slots, base_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                         channel13, high, high_scale)
        elif fp8_mode:
            base_body = compiled_write_body_fp8((value.shape[0], rows, q13.shape[0], fp8_mode))
            base_args = (value, routing, base_slots, base_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                         channel13, channel2, fp8_mode)
        else:
            base_body = compiled_write_body((value.shape[0], rows, q13.shape[0], bool(normal_scales)))
            base_args = (value, routing, base_slots, base_ids, q13, q2, s13, s2, lookup, normal_scales, routed)
        execute_prefill_experts(base_body, base_args, base_slots.shape[0], base_active)
        if tail_active:
            if fp8_mode == "w13_dual_prequant":
                tail_body = compiled_write_body_dual_prequant((value.shape[0], 64, q13.shape[0]))
                tail_args = (value, routing, tail_slots, tail_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                             channel13, high, high_scale, low, low_scale)
            elif fp8_mode == "w13_single_prequant":
                tail_body = compiled_write_body_single_prequant((value.shape[0], 64, q13.shape[0]))
                tail_args = (value, routing, tail_slots, tail_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                             channel13, high, high_scale)
            elif fp8_mode:
                tail_body = compiled_write_body_fp8((value.shape[0], 64, q13.shape[0], fp8_mode))
                tail_args = (value, routing, tail_slots, tail_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                             channel13, channel2, fp8_mode)
            else:
                tail_body = compiled_write_body((value.shape[0], 64, q13.shape[0], bool(normal_scales)))
                tail_args = (value, routing, tail_slots, tail_ids, q13, q2, s13, s2, lookup, normal_scales, routed)
            execute_prefill_experts(tail_body, tail_args, tail_slots.shape[0], tail_active)
        return compiled_reduce(
            (value.shape[0], value.shape[-1]))(routed[:-1].reshape(value.shape[0], 6, value.shape[-1]))
    # Descriptor shapes depend only on the finite scheduler compute bucket,
    # never on occupancy.  This exact graph is prepared before model loading.
    experts, slots, inverse, counts = compiled_routes((ids.shape[0], q13.shape[0], rows, mark_empty))(
        ids, q13.shape[0], rows, mark_empty)
    if envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN:
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import execute_prefill_experts, routed_workspace
        active_blocks = None
        if envs.VLLM_HPU_DSV41_PREFILL_ACTIVE_PLAN:
            # The sorted descriptors place every real route before empty
            # tail blocks. Read only the 384 occupancy counters; the native
            # plan can then omit the empty GEMMs, not merely mask their output.
            host_counts = counts.cpu().tolist()
            active_blocks = sum((int(count) + rows - 1) // rows for count in host_counts)
        routed = routed_workspace(value, ids.numel())
        if fp8_mode:
            body = compiled_write_body_fp8((value.shape[0], rows, q13.shape[0], fp8_mode))
            arguments = (value, routing, slots, experts, q13, q2, s13, s2, lookup, normal_scales, routed,
                         channel13, channel2, fp8_mode)
        else:
            body = compiled_write_body((value.shape[0], rows, q13.shape[0], bool(normal_scales)))
            arguments = (value, routing, slots, experts, q13, q2, s13, s2, lookup, normal_scales, routed)
        execute_prefill_experts(body, arguments, slots.shape[0], active_blocks)
        return compiled_reduce(
            (value.shape[0], value.shape[-1]))(routed[:-1].reshape(value.shape[0], 6, value.shape[-1]))
    # The bound includes padding for every expert, even under total skew.
    # One bounded body is active at a time; no list retains every body's
    # decoded weight blocks or a second complete copy of the routed outputs.
    compact = envs.VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT
    padded = value.new_empty((ids.numel() + 1 if compact else slots.numel(), value.shape[-1]))
    # Each body is bounded to at most eight experts x ``rows`` routes.  The
    # production recipe bundle prepares these shapes before model loading.
    # Reuse the compiled entry, including its input layout, so serving does
    # not create a separate eager recipe for the same expert computation.
    for start in range(0, slots.shape[0], 8):
        stop = min(start + 8, slots.shape[0])
        groups = stop - start
        # A contiguous view may still have a nonzero storage offset. Bridge
        # specializes those offsets into distinct recipes; materialize only
        # these small descriptors so every group reuses the same body recipe.
        group_slots = slots[start:stop].clone()
        # ``experts`` is a [1, capacity] view whose leading stride inherits
        # the runtime route-block capacity.  A size-one dimension is still
        # considered contiguous with that arbitrary stride, and ``clone``
        # preserves it.  Bridge keys the static recipe on strides, so the
        # inherited value (for example 477 at C2052) would miss the prepared
        # [1, 8] recipe whose canonical stride is (8, 1).  Flatten before the
        # copy and reshape afterwards to make every prompt reuse the finite
        # one-through-eight recipe bundle.
        group_experts = experts[:, start:stop].reshape(groups).clone().reshape(1, groups)
        selected, route = compiled_gather((value.shape[0], groups, rows, value.shape[-1]))(value, routing, group_slots)
        if fp8_mode:
            body = compiled_project_fp8((groups, rows, q13.shape[0], bool(normal_scales), fp8_mode))
            output = body(selected, route, group_experts, q13, q2, s13, s2, channel13, channel2, lookup,
                          normal_scales, fp8_mode)
        else:
            body = compiled_project((groups, rows, q13.shape[0], bool(normal_scales)))
            output = body(selected, route, group_experts, q13, q2, s13, s2, lookup, normal_scales)
        if compact:
            # Destination and scatter shapes are prepared for every finite
            # scheduler bucket; route values remain runtime device inputs.
            destinations = compiled_destinations((groups, rows, ids.numel()))(group_slots, ids.numel())
            padded = compiled_scatter((value.shape[0], groups, rows, value.shape[-1]))(padded, destinations, output)
        else:
            padded[start * rows:stop * rows].copy_(output)
    ordered = (padded[:-1] if compact else padded.index_select(0, inverse)).reshape(value.shape[0], 6, value.shape[-1])
    # Reuse the bucket's ordered reduction and its existing BF16 boundary.
    return compiled_reduce((value.shape[0], value.shape[-1]))(ordered)


@torch.compiler.disable
def run_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales,
                        channel13=None, channel2=None):
    """Execute the bounded expert plan at an intentional regional boundary.

    Large prompt layers are now regionally compiled.  The prepared plan owns
    multiple precompiled recipes and performs runtime binding in Python/C++;
    tracing that orchestration into the surrounding FX graph would either
    duplicate its recipes or retain every expert workspace for the full
    layer.  A Dynamo boundary here keeps the plan native while allowing the
    producer and consumer arithmetic on both sides to remain compiled.
    """
    if envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES:
        return run_device_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales,
                                          channel13, channel2)
    if envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN or envs.VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT:
        raise RuntimeError("Native prefill plans and route output require device route grouping")
    if torch.compiler.is_compiling():
        raise RuntimeError("Grouped prefill routing must be prepared outside the C1 replay graph")
    if value.ndim != 2 or ids.shape != (value.shape[0], 6) or routing.shape != ids.shape:
        raise ValueError("V4.1 grouped prefill requires matching [tokens,6] routing")
    host_ids = ids.to(device="cpu", dtype=torch.int32).numpy()
    # Maximum C8192 output is 480 MiB. Bounded bodies own at most eight
    # experts and 8192 padded activation rows, keeping workspace under 2 GiB.
    routed = torch.empty((ids.numel(), value.shape[-1]), dtype=value.dtype, device=value.device)
    for expert_ids, slots, valid in route_batches(host_ids, q13.shape[0]):
        body = compiled_body((value.shape[0], *slots.shape, q13.shape[0], bool(normal_scales)))
        device_experts = torch.from_numpy(expert_ids).to(value.device)
        device_slots = torch.from_numpy(slots).to(value.device)
        output = body(value, routing, device_slots, device_experts, q13, q2, s13, s2, lookup, normal_scales)
        local = torch.from_numpy(valid).to(value.device)
        destination = torch.from_numpy(slots.reshape(-1)[valid].copy()).to(value.device)
        routed.index_copy_(0, destination, output.index_select(0, local))
    return compiled_reduce((value.shape[0], value.shape[-1]))(routed.reshape(value.shape[0], 6, value.shape[-1]))
