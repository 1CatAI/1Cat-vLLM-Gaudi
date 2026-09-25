# SPDX-License-Identifier: Apache-2.0
"""Bounded BF16 expert execution grouped by current route occupancy."""

import functools
from collections import OrderedDict
from types import FunctionType

import torch

_ROW_BUCKETS = (64, 128, 192, 256)
_FP8_ROW_BUCKETS = tuple(range(32, 257, 32))
_executors = {}


def expert_bucket_width(rows, quantum=64):
    buckets = _FP8_ROW_BUCKETS if quantum == 32 else _ROW_BUCKETS
    if rows not in buckets:
        raise ValueError("Unsupported expert row bucket")
    # The route-write operator accepts at most 4096 rows. These batch widths
    # also retain the qualified compiler's W13 N256 and W2 batch-two tiles.
    return 24 if rows <= 128 else 16


def device_bucketed_route_blocks(ids, experts=384, quantum=64):
    """Describe full 256-row slabs and one rounded remainder per expert.

    Capacities depend on input shape; occupied prefixes depend on current IDs.
    Stable token/top-k positions are retained for the ordered route reduction.
    """
    if ids.ndim != 2 or ids.shape[1] != 6 or ids.numel() == 0:
        raise ValueError("Prefill routes must be nonempty [tokens,6]")
    if ids.dtype not in (torch.int32, torch.int64) or experts <= 0 or quantum not in (32, 64):
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
    rounded = ((counts.remainder(256) + quantum - 1) // quantum) * quantum
    descriptors, occupied = [], []
    buckets = _FP8_ROW_BUCKETS if quantum == 32 else _ROW_BUCKETS
    for rows in buckets:
        width = expert_bucket_width(rows, quantum)
        block_counts = (rounded == rows).int()
        if rows == 256:
            block_counts = block_counts + full
        # Every remainder has at least rows-quantum+1 routes; full slabs do as well.
        capacity = routes // (rows - quantum + 1)
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
        # Retain a full tile and a tail without keeping every request shape's
        # captured intermediates alive. The limit is independent of length.
        self.plans = OrderedDict()
        self.workspace = None
        self.indices = {}

    def close(self):
        if self.plans:
            torch.hpu.synchronize()
            for family in self.plans.values():
                for plan in family.values():
                    plan.plan.invalidate()
        self.plans.clear()
        self.workspace = None
        self.indices.clear()

    def group_indices(self, start, width, device):
        key = (start, width, device)
        value = self.indices.get(key)
        if value is None:
            value = torch.arange(start, start + width, dtype=torch.int32, device=device)
            self.indices[key] = value
        return value

    def routed_workspace(self, value, routes):
        if (self.workspace is None or self.workspace.shape[0] < routes + 1 or self.workspace.shape[1] != value.shape[-1]
                or self.workspace.dtype != value.dtype or self.workspace.device != value.device):
            # Prepared plans retain the old allocation. Drain their consumers
            # before replacing it; smaller shapes share its capacity instead.
            self.close()
            self.workspace = value.new_empty((routes + 1, value.shape[-1]))
        # The last row is the current invocation's invalid-route sentinel.
        return self.workspace[:routes + 1]

    def shape_plans(self, signature):
        family = self.plans.get(signature)
        if family is None:
            if len(self.plans) == 2:
                torch.hpu.synchronize()
                _, expired = self.plans.popitem(last=False)
                for plan in expired.values():
                    plan.plan.invalidate()
            family = self.plans[signature] = {}
        self.plans.move_to_end(signature)
        return family

    def __call__(self, value, ids, routing, q13, q2, s13, s2, lookup, normal_scales, channel13=None):
        import vllm_gaudi.envs as envs
        from vllm_gaudi.ops.deepseek_v41_grouped_prefill import (compiled_reduce, compiled_single_prequant,
                                                                 compiled_write_body)
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import PrefillExpertPlan, _expert_audit

        interleaved = envs.VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE and bool(normal_scales)
        single_fp8 = channel13 is not None
        if single_fp8 and not interleaved:
            raise ValueError("Bucketed W13 FP8 requires interleaved columns and normal scale codes")
        if single_fp8:
            from vllm_gaudi.ops.deepseek_v41_prefill_columns import (compiled_permuted_reduce,
                                                                     compiled_permuted_single_fp8_write_body)
            compile_body, compile_reduce = compiled_permuted_single_fp8_write_body, compiled_permuted_reduce
        elif interleaved:
            from vllm_gaudi.ops.deepseek_v41_prefill_columns import (compiled_permuted_reduce,
                                                                     compiled_permuted_write_body)
            compile_body, compile_reduce = compiled_permuted_write_body, compiled_permuted_reduce
        else:
            compile_body, compile_reduce = compiled_write_body, compiled_reduce
        if torch.hpu.current_stream().hpu_stream != torch.hpu.default_stream().hpu_stream:
            raise RuntimeError("Bucketed prefill scratch belongs to the default model stream")
        if value.dtype != torch.bfloat16:
            raise ValueError("Bucketed expert prefill requires BF16 activations")
        workspace = self.routed_workspace(value, ids.numel())
        if single_fp8:
            high, high_scale = compiled_single_prequant((value.shape[0], value.shape[-1]))(value)
        quantum = 32 if single_fp8 else 64
        buckets = _FP8_ROW_BUCKETS if single_fp8 else _ROW_BUCKETS
        desc = _compiled_routes((ids.shape[0], q13.shape[0], quantum))(ids, q13.shape[0], quantum)
        active = [int(count) for count in desc[-1].cpu().tolist()]
        weights = (q13, q2, s13, s2, lookup, channel13) if single_fp8 else (q13, q2, s13, s2, lookup)

        def layout(v):
            return tuple(v.shape), v.stride(), v.dtype, v.device

        signature = (bool(normal_scales), interleaved, single_fp8,
                     tuple(layout(v) for v in (value, ids, routing, *weights)))
        family = self.shape_plans(signature)
        for index, rows in enumerate(buckets):
            if not active[index]:
                continue
            width = expert_bucket_width(rows, quantum)
            experts, slots = desc[index * 2:index * 2 + 2]
            arguments = (value, routing, slots, experts, q13, q2, s13, s2, lookup, normal_scales, workspace)
            if single_fp8:
                arguments += (channel13, high, high_scale)
            complete, remainder = divmod(active[index], width)
            # Match the compiler's two-expert MME tiles for the final group.
            # Full groups keep their existing recipe and weight reuse.
            tail_width = ((remainder + 1) // 2) * 2 if single_fp8 else width
            compact_tail = 0 < tail_width < width
            submissions = []
            if compact_tail:
                if complete:
                    submissions.append((arguments, complete, None))
                tail_indices = self.group_indices(complete * width, tail_width, value.device)
                submissions.append(((*arguments, tail_indices), 1, tail_width))
            else:
                submissions.append((arguments, (active[index] + width - 1) // width, None))
            executed = 0
            for bindings, group_count, compact_width in submissions:
                binding_signature = (rows, tuple(layout(v) if isinstance(v, torch.Tensor) else v for v in bindings))
                plan = family.get(binding_signature)
                if plan is None:
                    body = compile_body(
                        ("bucketed_fp8" if single_fp8 else "bucketed", value.shape[0], rows, q13.shape[0],
                         bool(normal_scales), compact_width))
                    groups = None if compact_width else [
                        self.group_indices(start, width, value.device) for start in range(0, slots.shape[0], width)
                    ]
                    plan = PrefillExpertPlan(body, bindings, groups, require_prefix=True)
                    family[binding_signature] = plan
                    executed += plan.recipes
                else:
                    executed += plan.replay(bindings, group_count)
            _expert_audit["calls"] += 1
            _expert_audit["tokens"] += value.shape[0]
            _expert_audit["recipe_executions"] += executed
            _expert_audit["largest_token_bucket"] = max(_expert_audit["largest_token_bucket"], value.shape[0])
        return compile_reduce(
            (value.shape[0], value.shape[-1]))(workspace[:-1].reshape(value.shape[0], 6, value.shape[-1]))


def run_bucketed_prefill(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales, channel13=None):
    executor = _executors.get(value.device)
    if executor is None:
        executor = _BucketedPrefillPlans()
        _executors[value.device] = executor
    return executor(value, ids, routing, q13, q2, s13, s2, lookup, normal_scales, channel13)


def invalidate_bucketed_prefill_plans():
    for executor in _executors.values():
        executor.close()
    _executors.clear()


def bucketed_prefill_plan_stats():
    plans = [plan for executor in _executors.values() for family in executor.plans.values() for plan in family.values()]
    return dict(plans=len(plans),
                recipes=sum(p.recipes for p in plans),
                replays=sum(p.replays for p in plans),
                workspace_bytes=sum(e.workspace.numel() * e.workspace.element_size() for e in _executors.values()
                                    if e.workspace is not None))
