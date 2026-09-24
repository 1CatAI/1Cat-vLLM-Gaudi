# SPDX-License-Identifier: Apache-2.0
"""Bounded BF16 expert execution grouped by current route occupancy."""

import functools
from types import FunctionType

import torch

_ROW_BUCKETS = (64, 128, 192, 256)
_executors = {}


def expert_bucket_width(rows):
    if rows not in _ROW_BUCKETS:
        raise ValueError("Unsupported BF16 expert row bucket")
    # The route-write operator accepts at most 4096 rows. These batch widths
    # also retain the qualified compiler's W13 N256 and W2 batch-two tiles.
    return 24 if rows <= 128 else 16


def device_bucketed_route_blocks(ids, experts=384):
    """Describe full 256-row slabs and one rounded remainder per expert.

    Capacities depend on input shape; occupied prefixes depend on current IDs.
    Stable token/top-k positions are retained for the ordered route reduction.
    """
    if ids.ndim != 2 or ids.shape[1] != 6 or ids.numel() == 0:
        raise ValueError("Prefill routes must be nonempty [tokens,6]")
    if ids.dtype not in (torch.int32, torch.int64) or experts <= 0:
        raise ValueError("Prefill requires integer routes and positive expert count")
    routes = ids.numel()
    if routes * experts >= 2**31:
        raise ValueError("Route sort key exceeds I32 capacity")
    flat = ids.reshape(-1).int()
    expert_range = torch.arange(experts, dtype=torch.int32, device=ids.device)
    counts = (flat[None] == expert_range[:, None]).sum(-1, dtype=torch.int32)
    route_ends = counts.cumsum(0, dtype=torch.int32)
    keys = flat * routes + torch.arange(routes, dtype=torch.int32, device=ids.device)
    order = keys.sort().values.remainder(routes).long()
    full = counts // 256
    rounded = ((counts.remainder(256) + 63) // 64) * 64
    descriptors, occupied = [], []
    for rows in _ROW_BUCKETS:
        width = expert_bucket_width(rows)
        block_counts = (rounded == rows).int()
        if rows == 256:
            block_counts = block_counts + full
        # Every remainder has at least rows-63 routes; full slabs do as well.
        capacity = routes // (rows - 63)
        if rows < 256:
            capacity = min(experts, capacity)
        capacity = max(width, ((capacity + width - 1) // width) * width)
        block_ends = block_counts.cumsum(0, dtype=torch.int32)
        block = torch.arange(capacity, dtype=torch.int32, device=ids.device)
        owners = (block[:, None] >= block_ends[None]).sum(-1, dtype=torch.int32)
        safe = owners.clamp(max=experts - 1).long()
        if rows == 256:
            relative = block - (block_ends - block_counts)[safe]
            offset = relative * 256
        else:
            offset = full[safe] * 256
        local = offset[:, None] + torch.arange(rows, dtype=torch.int32, device=ids.device)[None]
        valid = (owners[:, None] < experts) & (local < counts[safe, None])
        positions = (route_ends - counts)[safe, None] + local
        slots = torch.where(valid, order[positions.clamp(0, routes - 1).long()], -1)
        expert_ids = torch.where(owners < experts, owners, -1).reshape(1, -1)
        descriptors.extend((expert_ids, slots))
        occupied.append(block_counts.sum())
    return (*descriptors, torch.stack(occupied))


@functools.lru_cache(maxsize=32)
def _compiled_routes(signature):
    entry = FunctionType(device_bucketed_route_blocks.__code__.replace(co_name=f"expert_buckets_{signature}"),
                         device_bucketed_route_blocks.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


class _BucketedPrefillPlans:

    def __init__(self):
        self.plans = {}
        self.workspace = None

    def close(self):
        if self.plans:
            torch.hpu.synchronize()
            for plan in self.plans.values():
                plan.plan.invalidate()
        self.plans.clear()
        self.workspace = None

    def __call__(self, value, ids, routing, q13, q2, s13, s2, lookup, normal_scales):
        import vllm_gaudi.envs as envs
        from vllm_gaudi.ops.deepseek_v41_grouped_prefill import compiled_reduce, compiled_write_body
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import PrefillExpertPlan, _expert_audit

        interleaved = envs.VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE and bool(normal_scales)
        if interleaved:
            from vllm_gaudi.ops.deepseek_v41_prefill_columns import (compiled_permuted_reduce,
                                                                  compiled_permuted_write_body)
            compile_body, compile_reduce = compiled_permuted_write_body, compiled_permuted_reduce
        else:
            compile_body, compile_reduce = compiled_write_body, compiled_reduce
        if torch.hpu.current_stream().hpu_stream != torch.hpu.default_stream().hpu_stream:
            raise RuntimeError("Bucketed prefill scratch belongs to the default model stream")
        if value.dtype != torch.bfloat16:
            raise ValueError("Bucketed expert prefill requires BF16 activations")
        workspace_shape = (ids.numel() + 1, value.shape[-1])
        if self.workspace is None or tuple(self.workspace.shape) != workspace_shape:
            self.close()
            self.workspace = value.new_empty(workspace_shape)
        desc = _compiled_routes((ids.shape[0], q13.shape[0]))(ids, q13.shape[0])
        active = [int(count) for count in desc[-1].cpu().tolist()]
        weight_signature = tuple((tuple(v.shape), v.stride(), v.dtype, v.device)
                                 for v in (q13, q2, s13, s2, lookup))
        for index, rows in enumerate(_ROW_BUCKETS):
            if not active[index]:
                continue
            width = expert_bucket_width(rows)
            experts, slots = desc[index * 2:index * 2 + 2]
            arguments = (value, routing, slots, experts, q13, q2, s13, s2, lookup, normal_scales, self.workspace)
            signature = (rows, bool(normal_scales), interleaved, value.stride(), weight_signature)
            plan = self.plans.get(signature)
            if plan is None:
                body = compile_body(("bucketed", value.shape[0], rows, q13.shape[0], bool(normal_scales)))
                groups = [
                    torch.arange(start, start + width, dtype=torch.int32, device=value.device)
                    for start in range(0, slots.shape[0], width)
                ]
                plan = PrefillExpertPlan(body, arguments, groups, require_prefix=True)
                self.plans[signature] = plan
                executed = plan.recipes
            else:
                executed = plan.replay(arguments, (active[index] + width - 1) // width)
            _expert_audit["calls"] += 1
            _expert_audit["tokens"] += value.shape[0]
            _expert_audit["recipe_executions"] += executed
            _expert_audit["largest_token_bucket"] = max(_expert_audit["largest_token_bucket"], value.shape[0])
        return compile_reduce((value.shape[0], value.shape[-1]))(
            self.workspace[:-1].reshape(value.shape[0], 6, value.shape[-1]))


def run_bucketed_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales):
    executor = _executors.get(value.device)
    if executor is None:
        executor = _BucketedPrefillPlans()
        _executors[value.device] = executor
    return executor(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales)


def invalidate_bucketed_prefill_plans():
    for executor in _executors.values():
        executor.close()
    _executors.clear()


def bucketed_prefill_plan_stats():
    plans = [plan for executor in _executors.values() for plan in executor.plans.values()]
    return dict(plans=len(plans), recipes=sum(p.recipes for p in plans), replays=sum(p.replays for p in plans),
                workspace_bytes=sum(e.workspace.numel() * e.workspace.element_size()
                                    for e in _executors.values() if e.workspace is not None))
