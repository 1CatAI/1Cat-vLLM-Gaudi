# SPDX-License-Identifier: Apache-2.0
"""Bounded, expert-grouped prefill on the native decode runtime.

Prefill owns one routing-table readback per layer. It buckets actual routes
by expert occupancy; compiled BMM bodies reuse each decoded weight over all
rows for that expert. No routes are dropped, including an entirely skewed
router. Resident N256 weights and the C1 implementation are unchanged.
"""

from __future__ import annotations

import functools
from types import FunctionType

import numpy as np
import torch
import torch.nn.functional as F


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


@functools.lru_cache(maxsize=128)
def compiled_body(signature):
    # Distinct occupancy buckets own distinct code objects, as do the model's
    # existing static layer groups. Routing values remain device inputs.
    entry = FunctionType(grouped_body.__code__.replace(co_name=f"grouped_prefill_{signature}"),
                         grouped_body.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


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


def run_grouped_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales):
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
