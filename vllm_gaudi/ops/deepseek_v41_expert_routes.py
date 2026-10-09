# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity device descriptors for unique-expert execution.

Descriptors preserve original route order for the final BF16 reduction. This
does not select a GEMM recipe: a consumer must honour device counts, including
the zero-count branch, before it can replace the ordinary routed operator.
"""
import torch


def pack_expert_routes(ids):
    """Return unique IDs, row counts, route slots and the inverse route map.

    The router selects distinct experts within each token, so an expert has
    at most ``tokens`` rows. Capacity is the original number of routes. Every
    tensor retains a static shape; no device scalar is read by the host.
    """
    tokens, top_k = ids.shape
    capacity = tokens * top_k
    lane = torch.arange(capacity, dtype=torch.int64, device=ids.device)
    flattened = ids.reshape(-1).to(torch.int64)
    # Original slot breaks ties deterministically, independently of sort's
    # stability. All IDs are nonnegative validated router outputs.
    order = (flattened * capacity + lane).argsort()
    sorted_ids = flattened.gather(0, order)
    starts = torch.cat((torch.ones(1, dtype=torch.bool, device=ids.device),
                        sorted_ids[1:] != sorted_ids[:-1]))
    group = starts.to(torch.int64).cumsum(0) - 1
    first = torch.full_like(lane, capacity).scatter_reduce(0, group, lane, reduce="amin", include_self=True)
    counts = torch.zeros_like(lane).scatter_add(0, group, torch.ones_like(lane))
    experts = sorted_ids.gather(0, first.clamp_max(capacity - 1))
    experts = torch.where(counts > 0, experts, -1)
    row = lane - first.gather(0, group)
    slots = torch.zeros(capacity * tokens, dtype=torch.int64, device=ids.device)
    slots = slots.scatter(0, group * tokens + row, order + 1).reshape(capacity, tokens) - 1
    inverse = torch.zeros_like(lane).scatter(0, order, group)
    inverse_row = torch.zeros_like(lane).scatter(0, order, row)
    return experts.to(torch.int32), counts.to(torch.int32), slots.to(torch.int32), inverse, inverse_row
