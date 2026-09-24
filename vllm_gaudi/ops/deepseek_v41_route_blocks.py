# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity, device-only prefill route descriptors.

Capacity depends on the scheduled token tile, never expert occupancy. A unique
integer sort key preserves token/top-k order without a stable-sort requirement.
All real routes survive arbitrarily skewed distributions; padding is explicit.
"""
import torch


def route_block_capacity(routes: int, experts: int, rows: int) -> int:
    if routes <= 0 or experts <= 0 or rows not in (32, 64, 128, 512):
        raise ValueError("Route blocks require positive dimensions and 32/64/128/512 rows")
    return min(routes, (routes + experts * (rows - 1)) // rows)


def device_route_blocks(ids, experts=384, rows=64, mark_empty=False):
    """Return expert IDs, route slots and inverse mapping; no device readback.

    The producer must supply valid expert IDs. ``inverse`` maps every original
    route to exactly one padded block row. Its extra sentinel slot is discarded,
    so duplicate stores from padding never affect a real route's destination.
    """
    if ids.ndim != 2 or ids.shape[1] != 6 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Prefill routes must be integer [tokens,6]")
    routes = ids.numel()
    capacity = route_block_capacity(routes, experts, rows)
    if routes * experts >= 2**31:
        raise ValueError("Route sort key exceeds I32 capacity")
    flat = ids.reshape(-1).int()
    expert_range = torch.arange(experts, dtype=torch.int32, device=ids.device)
    # This fixed reduction is deliberately separate from the descriptor
    # contract, so a native histogram can replace it without changing recipes'
    # data-dependent shapes or the consumer's ownership rules.
    counts = (flat[None] == expert_range[:, None]).sum(-1, dtype=torch.int32)
    route_ends = counts.cumsum(0, dtype=torch.int32)
    block_counts = (counts + rows - 1) // rows
    block_ends = block_counts.cumsum(0, dtype=torch.int32)
    keys = flat * routes + torch.arange(routes, dtype=torch.int32, device=ids.device)
    order = keys.sort().values.remainder(routes).long()
    block = torch.arange(capacity, dtype=torch.int32, device=ids.device)
    owners = (block[:, None] >= block_ends[None]).sum(-1, dtype=torch.int32)
    safe_owners = owners.clamp(max=experts - 1).long()
    relative_block = block - (block_ends - block_counts)[safe_owners]
    local = relative_block[:, None] * rows + torch.arange(rows, dtype=torch.int32, device=ids.device)[None]
    valid = (owners[:, None] < experts) & (local < counts[safe_owners, None])
    sorted_slots = (route_ends - counts)[safe_owners, None] + local
    slots = order[sorted_slots.clamp(0, routes - 1).long()]
    slots = torch.where(valid, slots, -1)
    inverse = torch.zeros(routes + 1, dtype=torch.int64, device=ids.device)
    inverse.scatter_(0, torch.where(slots >= 0, slots, routes).flatten(),
                     torch.arange(capacity * rows, dtype=torch.int64, device=ids.device))
    decoded = torch.where(owners < experts, owners, -1) if mark_empty else safe_owners.int()
    return decoded.reshape(1, -1), slots, inverse[:-1], counts


def device_hybrid_route_blocks(ids, experts=384, base_rows=128, tail_rows=64):
    """Describe occupied experts with one large block and smaller overflow blocks.

    Both capacities depend only on the input shape. The two occupied prefixes
    depend on the current route IDs; no route is lost under arbitrary skew.
    Each expert's original token/top-k order is retained across its blocks.
    """
    if ids.ndim != 2 or ids.shape[1] != 6 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Prefill routes must be integer [tokens,6]")
    if ids.numel() == 0 or experts <= 0 or (base_rows, tail_rows) != (128, 64):
        raise ValueError("Hybrid route blocks require positive routes/experts and 128+64 rows")
    routes = ids.numel()
    if routes * experts >= 2**31:
        raise ValueError("Route sort key exceeds I32 capacity")
    flat = ids.reshape(-1).int()
    expert_range = torch.arange(experts, dtype=torch.int32, device=ids.device)
    counts = (flat[None] == expert_range[:, None]).sum(-1, dtype=torch.int32)
    route_ends = counts.cumsum(0, dtype=torch.int32)
    keys = flat * routes + torch.arange(routes, dtype=torch.int32, device=ids.device)
    order = keys.sort().values.remainder(routes).long()

    def describe(block_counts, capacity, rows, offset):
        block_ends = block_counts.cumsum(0, dtype=torch.int32)
        block = torch.arange(capacity, dtype=torch.int32, device=ids.device)
        owners = (block[:, None] >= block_ends[None]).sum(-1, dtype=torch.int32)
        safe_owners = owners.clamp(max=experts - 1).long()
        relative = block - (block_ends - block_counts)[safe_owners]
        local = relative[:, None] * rows + torch.arange(rows, dtype=torch.int32, device=ids.device)[None]
        source = local + offset
        valid = (owners[:, None] < experts) & (source < counts[safe_owners, None])
        sorted_slots = (route_ends - counts)[safe_owners, None] + source
        slots = torch.where(valid, order[sorted_slots.clamp(0, routes - 1).long()], -1)
        decoded = torch.where(owners < experts, owners, -1).reshape(1, -1)
        return decoded, slots

    base_counts = (counts > 0).int()
    tail_counts = ((counts - base_rows).clamp(min=0) + tail_rows - 1) // tail_rows
    base_ids, base_slots = describe(base_counts, min(experts, routes), base_rows, 0)
    # A single expert with every route maximizes the overflow-block count.
    # ceil(routes / tail_rows) is a fixed safe bound for every distribution.
    tail_ids, tail_slots = describe(tail_counts, (routes + tail_rows - 1) // tail_rows, tail_rows, base_rows)
    occupied = torch.stack((base_counts.sum(), tail_counts.sum()))
    return base_ids, base_slots, tail_ids, tail_slots, occupied
