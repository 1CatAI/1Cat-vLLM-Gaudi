# SPDX-License-Identifier: Apache-2.0
"""Prompt plan ownership and runtime input binding contracts."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_prefill_plan as plans


def test_hybrid_rows_require_the_exact_native_bf16_or_fp8_contract(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS", "1")
    with pytest.raises(ValueError, match="Hybrid prefill rows require"):
        plans.validate_prefill_plan_config(n256=True)
    for key in ("GROUPED", "DEVICE_ROUTES", "ROUTE_OUTPUT", "NATIVE_PLAN", "FAST_DEQUANT", "SKIP_EMPTY"):
        monkeypatch.setenv(f"VLLM_HPU_DSV41_PREFILL_{key}", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS", "128")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "")
    plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13_dual")
    plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13_dual_prequant")
    plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13_single_prequant")
    plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13_single_bucket")
    plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS", "0")
    with pytest.raises(ValueError, match="Pre-quantized W13 requires"):
        plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13")
    with pytest.raises(ValueError, match="Hybrid prefill rows require"):
        plans.validate_prefill_plan_config(n256=True)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_GROUPED_FP8", "w13_dual")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EXPERT_ROWS", "64")
    with pytest.raises(ValueError, match="128-row"):
        plans.validate_prefill_plan_config(n256=True)


def test_active_plan_checks_prefix_abi_before_recipe_preparation(monkeypatch):
    bridge = SimpleNamespace(PreparedGroupPlan=object)
    monkeypatch.setattr(plans, "_runtime", lambda: (bridge, object()))
    with pytest.raises(RuntimeError, match="matching prepared runtime ABI"):
        plans.PrefillExpertPlan(None, (), (), require_prefix=True)


def test_workspace_growth_drains_consumers_and_invalidates_old_addresses(monkeypatch):
    calls = []
    monkeypatch.setattr(plans, "_plans", OrderedDict())
    monkeypatch.setattr(plans, "_workspaces", {})
    attention = OrderedDict(kept=SimpleNamespace(plan=SimpleNamespace(invalidate=lambda: calls.append("wrong"))))
    monkeypatch.setattr(plans, "_attention_plans", attention)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: calls.append("complete"))
    value = torch.zeros(2, 16, dtype=torch.bfloat16)
    first = plans.routed_workspace(value, 12)
    smaller = plans.routed_workspace(value, 6)
    assert first.data_ptr() == smaller.data_ptr()
    plans._plans["prepared"] = SimpleNamespace(plan=SimpleNamespace(invalidate=lambda: calls.append("invalidate")))
    second = plans.routed_workspace(value, 24)
    assert calls == ["complete", "invalidate"]
    assert not plans._plans
    assert second.data_ptr() != first.data_ptr()
    assert len(plans._workspaces) == 1
    assert list(plans._attention_plans) == ["kept"]


def test_replay_consumes_rebound_activation_weight_and_route_inputs():
    observed = []
    plan = object.__new__(plans.PrefillExpertPlan)
    fixed_index = torch.tensor([1, 2], dtype=torch.int32)
    old = [torch.zeros(2, 4), torch.zeros(2, 6), torch.zeros(4, 4)]
    plan.external = [old[2], fixed_index, old[0], old[1]]
    plan.bindings = [2, None, 0, 1]
    plan.plan = SimpleNamespace(matches=lambda inputs: True)
    plan.bridge = SimpleNamespace(replay_prepared_groups=lambda p, values: observed.append(values[0]))
    plan.recipes = 3
    plan.replays = 0
    current = [torch.ones(2, 4), torch.full((2, 6), 5), torch.eye(4)]
    plan.replay(current)
    assert all(a is b for a, b in zip(observed[-1], [current[2], fixed_index, current[0], current[1]], strict=True))
    plan.plan.matches = lambda inputs: False
    with pytest.raises(RuntimeError, match="invalidate before modifying request state"):
        plan.replay(current)
    assert len(observed) == 1 and plan.replays == 1


def test_active_prefix_replay_keeps_current_bindings_and_bounded_node_count():
    observed = []
    plan = object.__new__(plans.PrefillExpertPlan)
    plan.external = [torch.zeros(2, 4)]
    plan.bindings = [0]
    plan.plan = SimpleNamespace(matches=lambda inputs: inputs[0].shape == (2, 4))
    plan.bridge = SimpleNamespace(
        replay_prepared_groups_prefix=lambda p, values, limits: observed.append((values[0][0], limits[0])))
    plan.group_node_ends = [2, 5, 7]
    plan.recipes = 7
    plan.replays = 0
    current = torch.ones(2, 4)
    assert plan.replay([current], 2) == 5
    assert len(observed) == 1 and observed[0][0] is current and observed[0][1] == 5 and plan.replays == 1
    with pytest.raises(ValueError, match="exceeds the prepared plan"):
        plan.replay([current], 4)
    assert len(observed) == 1 and plan.replays == 1


def test_model_invalidation_retires_prefill_before_releasing_weight_bindings(monkeypatch):
    from vllm_gaudi.models.deepseek_v41_program import PreparedStage
    from vllm_gaudi.ops import deepseek_v41_prefill_regions as regions

    events = []
    monkeypatch.setattr(plans, "invalidate_prefill_plans", lambda: events.append("prefill"))
    monkeypatch.setattr(regions, "_function_regions", OrderedDict(old=object()))
    attention = SimpleNamespace(
        _prefill_tensor_regions={"old": object()},
        invalidate_qkv_input_weight=lambda: events.append("weight"),
        invalidate_compressor_input_weight=lambda: None,
    )
    layer = SimpleNamespace(
        attention=attention,
        _prefill_tensor_regions={"old": object()},
        release_mhc_control_weights=lambda: None,
    )
    stage = SimpleNamespace(
        layers=[layer],
        loaded=True,
        generation=3,
        replay_owner=SimpleNamespace(close=lambda: events.append("decode")),
    )
    stage._invalidate_prefill_regions = lambda: PreparedStage._invalidate_prefill_regions(stage)
    PreparedStage.invalidate(stage)
    assert events == ["decode", "prefill", "weight"]
    assert not regions._function_regions
    assert "_prefill_tensor_regions" not in vars(layer)
    assert "_prefill_tensor_regions" not in vars(attention)
    assert stage.generation == 4 and not stage.loaded
