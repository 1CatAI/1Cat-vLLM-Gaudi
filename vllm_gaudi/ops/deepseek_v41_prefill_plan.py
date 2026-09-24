# SPDX-License-Identifier: Apache-2.0
"""Bounded prompt expert execution through the maintained prepared-plan ABI.

Only the first ordinary invocation discovers the compiler's recipe boundaries.
Subsequent invocations bind current device route descriptors and submit the
whole bounded expert sequence in C++. No graph or numerical code is injected
into a running model, and expert occupancy never becomes a host scalar.
"""
import threading
from collections import OrderedDict
from pathlib import Path
import os

import torch

_local = threading.local()
_registered = False
_plans = OrderedDict()
_workspaces = {}
_attention_plans = OrderedDict()
_attention_workspaces = {}
_expert_audit = dict(calls=0, tokens=0, recipe_executions=0, largest_token_bucket=0)


def validate_prefill_plan_config(*, n256):
    from vllm_gaudi import envs
    if (envs.VLLM_HPU_DSV41_PREFILL_GROUPED_FP8 in ("w13_dual_prequant", "w13_single_prequant")
            and not envs.VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS):
        raise ValueError("Pre-quantized W13 requires hybrid route plans")
    if envs.VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS and not (
            envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES and envs.VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT
            and envs.VLLM_HPU_DSV41_PREFILL_GROUPED and envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN
            and envs.VLLM_HPU_DSV41_PREFILL_FAST_DEQUANT and envs.VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY
            and envs.VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS == 128
            and envs.VLLM_HPU_DSV41_PREFILL_GROUPED_FP8 in ("", "w13_dual", "w13_dual_prequant",
                                                                "w13_single_prequant")):
        raise ValueError("Hybrid prefill rows require BF16 or W13 FP8 128-row native plans with device routes, "
                         "route output, fast dequant and empty-block masking")
    if not envs.VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN:
        return
    if envs.VLLM_HPU_DSV41_PREFILL_ACTIVE_PLAN and not envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES:
        raise ValueError("Active prefill plans require device route grouping")
    requirements = {
        "N256 prepared weights": n256,
        "grouped expert dispatch": envs.VLLM_HPU_DSV41_PREFILL_GROUPED,
        "device route grouping": envs.VLLM_HPU_DSV41_PREFILL_DEVICE_ROUTES,
        "route-ordered output": envs.VLLM_HPU_DSV41_PREFILL_ROUTE_OUTPUT,
    }
    missing = [name for name, enabled in requirements.items() if not enabled]
    if missing:
        raise ValueError("Native prefill plan cannot execute without: " + ", ".join(missing))


def register_prefill_plan_pass():
    global _registered
    if _registered:
        return
    from habana_frameworks.torch.dynamo.compile_backend.passes import (OptimizationPassPlacement,
                                                                       register_pass_at_optimization_pass)

    class Observe(torch.nn.Module):

        def __init__(self, recipe):
            super().__init__()
            self.recipe = recipe

        def forward(self, *inputs):
            result = self.recipe(*inputs)
            calls = getattr(_local, "calls", None)
            if calls is not None:
                outputs = list(result) if isinstance(result, (tuple, list)) else [result]
                aliases = set((self.recipe._in_to_out_dups or {}).values())
                calls.append(
                    (self.recipe._recipe_id, list(inputs), [v for i, v in enumerate(outputs) if i not in aliases]))
            return result

    def wrap(context):
        if getattr(_local, "calls", None) is None:
            return False
        changed = False
        for name, child in list(context.graph_module.named_children()):
            if hasattr(child, "_recipe_id"):
                if child.is_dynamic or child._has_randoms:
                    raise RuntimeError("Prefill plans require static deterministic recipes")
                context.graph_module.add_module(name, Observe(child))
                changed = True
        return changed

    register_pass_at_optimization_pass(wrap, OptimizationPassPlacement.POST_PARTITIONER)
    _registered = True


def _runtime():
    from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _verify_prepared_runtime, _resolve_runtime
    # A standalone complete-chain test uses one HCCL rank. Serving always
    # resolves the actual TP communicator, never the four-rank default group.
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() == 1:
        path = Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"])
        bridge = _load_bridge(path)
        _verify_prepared_runtime(path)
        backend = torch.distributed.distributed_c10d._get_default_group()._get_backend(torch.device("hpu"))
        return bridge, backend
    bridge, backend, _ = _resolve_runtime()
    return bridge, backend


def _key(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
    return type(value), value


class PrefillExpertPlan:

    def __init__(self, body, arguments, groups, require_prefix=False):
        self.bridge, backend = _runtime()
        if not hasattr(self.bridge, "PreparedGroupPlan"):
            raise RuntimeError("Prefill requires the version-locked prepared recipe API")
        if require_prefix and not hasattr(self.bridge, "replay_prepared_groups_prefix"):
            raise RuntimeError("Active expert plan requires the matching prepared runtime ABI")
        register_prefill_plan_pass()
        if getattr(_local, "calls", None) is not None:
            raise RuntimeError("Nested prefill plan preparation")
        _local.calls = []
        self.group_node_ends = []
        try:
            for indices in groups:
                body(*arguments, indices)
                self.group_node_ends.append(len(_local.calls))
            torch.hpu.synchronize()
            calls = _local.calls
        finally:
            _local.calls = None
        if not calls:
            raise RuntimeError("Prefill compiler produced no recordable recipe; cannot use eager fallback")
        plan = self.bridge.PreparedGroupPlan()
        slots, tensors, external = {}, {}, []
        root = {_key(value): index for index, value in enumerate(arguments)}
        self.bindings = []

        def slot(value, is_input):
            identity = _key(value)
            if identity not in slots:
                if isinstance(value, torch.Tensor) and value.is_contiguous():
                    for known, tensor in tensors.items():
                        if (tensor.data_ptr() == value.data_ptr() and tensor.dtype == value.dtype
                                and tensor.numel() == value.numel() and tensor.is_contiguous()):
                            slots[identity] = plan.add_reshape_view(slots[known], list(value.shape))
                            return slots[identity]
                slots[identity] = plan.add_slot(value, is_input)
                if isinstance(value, torch.Tensor):
                    tensors[identity] = value
                if is_input:
                    external.append(value)
                    self.bindings.append(root.get(identity))
            return slots[identity]

        for recipe, inputs, outputs in calls:
            plan.add_compute(recipe, [slot(v, True) for v in inputs], [slot(v, False) for v in outputs])
        # Completion tensors retain the writes; routed storage is an explicit
        # input mutation and is consumed by the following ordered reduction.
        plan.prepare(backend, [])
        self.plan, self.external, self.groups = plan, external, groups
        self.recipes, self.replays = len(calls), 0

    def replay(self, arguments, active_groups=None):
        inputs = [
            value if index is None else arguments[index]
            for index, value in zip(self.bindings, self.external, strict=True)
        ]
        if not self.plan.matches(inputs):
            raise RuntimeError("Prefill recipe binding changed; invalidate before modifying request state")
        if active_groups is None:
            self.bridge.replay_prepared_groups([self.plan], [inputs])
            executed = self.recipes
        else:
            if not 1 <= active_groups <= len(self.group_node_ends):
                raise ValueError("Active expert group count exceeds the prepared plan")
            if not hasattr(self.bridge, "replay_prepared_groups_prefix"):
                raise RuntimeError("Active expert plan requires the matching prepared runtime ABI")
            executed = self.group_node_ends[active_groups - 1]
            self.bridge.replay_prepared_groups_prefix([self.plan], [inputs], [executed])
        self.replays += 1
        return executed


def execute_prefill_experts(body, arguments, blocks, active_blocks=None):
    if torch.hpu.current_stream().hpu_stream != torch.hpu.default_stream().hpu_stream:
        raise RuntimeError("Prefill scratch is owned by the default model compute stream")
    # Weights are explicit runtime bindings, just like activation/route inputs.
    # Reuse a single sequence across equal-shaped layers rather than retaining
    # one prompt activation and workspace for every layer.
    from vllm_gaudi import envs
    experts_per_plan = envs.VLLM_HPU_DSV41_PREFILL_EXPERTS_PER_PLAN
    if experts_per_plan not in (8, 16, 24, 32):
        raise ValueError("Prefill expert plans support 8, 16, 24 or 32 experts per submission")
    if active_blocks is not None and not 1 <= active_blocks <= blocks:
        raise ValueError("Active expert blocks must be a nonempty descriptor prefix")
    signature = tuple((tuple(v.shape), v.stride(), v.dtype, v.device) for v in arguments[4:9]) + (tuple(
        arguments[0].shape), tuple(arguments[2].shape), bool(arguments[9]), experts_per_plan, body)
    plan = _plans.get(signature)
    if plan is None:
        groups = [
            torch.arange(start, min(start + experts_per_plan, blocks), device=arguments[0].device, dtype=torch.int32)
            for start in range(0, blocks, experts_per_plan)
        ]
        plan = PrefillExpertPlan(body, arguments, groups, require_prefix=active_blocks is not None)
        _plans[signature] = plan
        if len(_plans) > 4:
            torch.hpu.synchronize()
            _, expired = _plans.popitem(last=False)
            expired.plan.invalidate()
        executed = plan.recipes
    else:
        _plans.move_to_end(signature)
        active_groups = None if active_blocks is None else (active_blocks + experts_per_plan - 1) // experts_per_plan
        executed = plan.replay(arguments, active_groups)
    _expert_audit["calls"] += 1
    _expert_audit["tokens"] += arguments[0].shape[0]
    _expert_audit["recipe_executions"] += executed
    _expert_audit["largest_token_bucket"] = max(_expert_audit["largest_token_bucket"], arguments[0].shape[0])


def routed_workspace(value, routes):
    # All layer calls on the model compute stream reuse this allocation.
    # Stream ordering holds ownership through the following ordered reduction.
    key = (value.device, value.dtype, value.shape[-1])
    capacity = _workspaces.get(key)
    if capacity is None or capacity.shape[0] < routes + 1:
        # Old plans retain the old allocation. Reprepare only at a legitimate
        # prompt bucket change, never for route occupancy or token values.
        _invalidate(_plans)
        _workspaces.clear()
        capacity = value.new_empty((routes + 1, value.shape[-1]))
        _workspaces[key] = capacity
    return capacity[:routes + 1]


def _invalidate(collection):
    if collection:
        torch.hpu.synchronize()
        for plan in collection.values():
            plan.plan.invalidate()
    collection.clear()


def invalidate_prefill_plans():
    from vllm_gaudi.ops.deepseek_v41_prefill_buckets import invalidate_bucketed_prefill_plans
    invalidate_bucketed_prefill_plans()
    _invalidate(_plans)
    _invalidate(_attention_plans)
    _workspaces.clear()
    _attention_workspaces.clear()


def prefill_plan_stats():
    from vllm_gaudi.ops.deepseek_v41_prefill_buckets import bucketed_prefill_plan_stats
    from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import prefill_index_stats
    bucketed = bucketed_prefill_plan_stats()
    workspace_bytes = bucketed["workspace_bytes"] + sum(
        t.numel() * t.element_size() for t in (*_workspaces.values(), *_attention_workspaces.values()))
    return dict(executed=dict(_expert_audit),
                index_query_tp=prefill_index_stats(),
                plans=len(_plans) + bucketed["plans"],
                recipes=sum(p.recipes for p in _plans.values()) + bucketed["recipes"],
                replays=sum(p.replays for p in _plans.values()) + bucketed["replays"],
                attention_plans=len(_attention_plans),
                attention_recipes=sum(p.recipes for p in _attention_plans.values()),
                attention_replays=sum(p.replays for p in _attention_plans.values()),
                workspace_bytes=workspace_bytes)
