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
from vllm_gaudi.ops.deepseek_v41_route_blocks import device_route_blocks


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


def grouped_body(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales: bool):
    """One compiled gather -> decode -> W13 -> weighted SwiGLU -> W2 chain."""
    groups, capacity = slots.shape
    safe_slots = slots.clamp(min=0)
    selected = value.index_select(0, (safe_slots // 6).reshape(-1))
    selected = selected.reshape(groups, capacity, value.shape[-1])
    weight13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(expert_ids, q13, s13, lookup,
                                                                               normal_scales)
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
    weight2 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(expert_ids, q2, s2, lookup, normal_scales)
    return torch.bmm(middle, weight2).reshape(-1, value.shape[-1])


def grouped_project(selected, route, expert_ids, q13, q2, s13, s2, lookup, normal_scales: bool):
    """Fixed-shape decode/MME body after route gathering.

    Keeping the scheduler token count outside this graph gives serving a
    finite set of eight recipes (one through eight expert blocks).  A new
    prompt tail can therefore reuse a prepared body instead of compiling a
    large graph beside the resident model and 1M KV allocation.
    """
    groups, capacity, hidden = selected.shape
    weight13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
        expert_ids, q13, s13, lookup, normal_scales)
    projected = torch.bmm(selected, weight13)
    middle_width = projected.shape[-1] // 2
    gate = projected[..., :middle_width].float().clamp(max=10.0)
    up = projected[..., middle_width:].float().clamp(-10.0, 10.0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(
        middle.reshape(1, -1)).reshape(groups, capacity, middle_width)
    weight2 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
        expert_ids, q2, s2, lookup, normal_scales)
    return torch.bmm(middle, weight2).reshape(-1, hidden)


def grouped_body_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales, routed, indices):
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    output = grouped_body(value, routing, local_slots, local_experts, q13, q2, s13, s2, lookup, normal_scales)
    # Native sparse writes ignore -1 padding and emit a completion token.
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(
        routed, output, local_slots.flatten().int())


@functools.lru_cache(maxsize=32)
def compiled_write_body(signature):
    entry = FunctionType(grouped_body_write.__code__.replace(co_name=f"prefill_write_{signature}"),
                         grouped_body_write.__globals__)
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


def grouped_gather(value, routing, slots):
    """Build one canonical activation/route block without decoding weights."""
    groups, rows = slots.shape
    safe_slots = slots.clamp(min=0)
    selected = value.index_select(0, (safe_slots // 6).reshape(-1)).reshape(
        groups, rows, value.shape[-1])
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
    if not (envs.VLLM_HPU_DSV41_PREFILL_GROUPED
            and envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES):
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
            output = compiled_project((groups, rows, experts, bool(normal_scales)))(
                selected, route, expert_ids, q13, q2, s13, s2, lookup, bool(normal_scales))
            torch.hpu.synchronize()
            del selected, route, expert_ids, output
        # Normal serving admits C8192 but executes it as a finite set of exact
        # model buckets.  Compile every descriptor/gather/scatter/reduction
        # shape before model loading so the live request reuses prepared
        # recipes. Operator-contract failures must still be diagnosed from
        # the compiler log, not inferred from resident-memory size.
        for tokens in (2048, 1024, 512, 256, 128):
            ids = (torch.arange(tokens * 6, dtype=torch.int32, device=device)
                   .remainder(experts).reshape(tokens, 6))
            routing = torch.full((tokens, 6), 1.0 / 6.0,
                                 dtype=torch.float32, device=device)
            value = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
            expert_ids, slots, _, _ = compiled_routes(
                (tokens, experts, rows, False))(ids, experts, rows, False)
            torch.hpu.synchronize()
            for start in range(0, slots.shape[0], 8):
                groups = min(8, slots.shape[0] - start)
                group_slots = slots[start:start + groups].clone()
                selected, route = compiled_gather((tokens, groups, rows, hidden))(
                    value, routing, group_slots)
                destinations = compiled_destinations((groups, rows, tokens * 6))(
                    group_slots, tokens * 6)
                padded = torch.empty((tokens * 6 + 1, hidden), dtype=torch.bfloat16,
                                     device=device)
                projected = torch.empty((groups * rows, hidden), dtype=torch.bfloat16,
                                        device=device)
                padded = compiled_scatter((tokens, groups, rows, hidden))(
                    padded, destinations, projected)
                torch.hpu.synchronize()
                del group_slots, selected, route, destinations, padded, projected
            ordered = torch.empty((tokens, 6, hidden), dtype=torch.bfloat16, device=device)
            reduced = compiled_reduce((tokens, hidden))(ordered)
            torch.hpu.synchronize()
            del ids, routing, value, expert_ids, slots, ordered, reduced
    del q13, q2, s13, s2, lookup
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


def run_device_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales):
    if torch.compiler.is_compiling():
        raise RuntimeError("Prefill group submission must remain outside a C1 graph")
    if value.ndim != 2 or ids.shape != (value.shape[0], 6) or routing.shape != ids.shape:
        raise ValueError("V4.1 device prefill requires matching [tokens,6] routing")
    rows = envs.VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS
    if rows not in (32, 64, 128):
        raise ValueError("Prefill expert rows must be 32, 64 or 128")
    mark_empty = envs.VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY
    # Descriptor shapes depend only on the finite scheduler compute bucket,
    # never on occupancy.  This exact graph is prepared before model loading.
    experts, slots, inverse, _ = compiled_routes(
        (ids.shape[0], q13.shape[0], rows, mark_empty))(
            ids, q13.shape[0], rows, mark_empty)
    if envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN:
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import execute_prefill_experts, routed_workspace
        routed = routed_workspace(value, ids.numel())
        body = compiled_write_body((value.shape[0], rows, q13.shape[0], bool(normal_scales)))
        arguments = (value, routing, slots, experts, q13, q2, s13, s2, lookup, normal_scales, routed)
        execute_prefill_experts(body, arguments, slots.shape[0])
        return compiled_reduce((value.shape[0], value.shape[-1]))(
            routed[:-1].reshape(value.shape[0], 6, value.shape[-1]))
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
        selected, route = compiled_gather((value.shape[0], groups, rows, value.shape[-1]))(
            value, routing, group_slots)
        body = compiled_project((groups, rows, q13.shape[0], bool(normal_scales)))
        output = body(selected, route, group_experts, q13, q2, s13, s2, lookup, normal_scales)
        if compact:
            # Destination and scatter shapes are prepared for every finite
            # scheduler bucket; route values remain runtime device inputs.
            destinations = compiled_destinations((groups, rows, ids.numel()))(
                group_slots, ids.numel())
            padded = compiled_scatter((value.shape[0], groups, rows, value.shape[-1]))(
                padded, destinations, output)
        else:
            padded[start * rows:stop * rows].copy_(output)
    ordered = (padded[:-1] if compact else padded.index_select(0, inverse)).reshape(
        value.shape[0], 6, value.shape[-1])
    # Reuse the bucket's ordered reduction and its existing BF16 boundary.
    return compiled_reduce((value.shape[0], value.shape[-1]))(ordered)


@torch.compiler.disable
def run_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales):
    """Execute the bounded expert plan at an intentional regional boundary.

    Large prompt layers are now regionally compiled.  The prepared plan owns
    multiple precompiled recipes and performs runtime binding in Python/C++;
    tracing that orchestration into the surrounding FX graph would either
    duplicate its recipes or retain every expert workspace for the full
    layer.  A Dynamo boundary here keeps the plan native while allowing the
    producer and consumer arithmetic on both sides to remain compiled.
    """
    if envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES:
        return run_device_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales)
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
