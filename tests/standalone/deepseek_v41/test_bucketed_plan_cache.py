# SPDX-License-Identifier: Apache-2.0
"""Bounded shape reuse, route sentinels and prepared-plan lifetime."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_prefill_buckets as buckets


def test_capacity_reuse_preserves_the_current_sentinel_and_drains_before_growth(monkeypatch):
    events = []
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: events.append("drain"))
    owner = buckets._BucketedPrefillPlans()
    value = torch.zeros(7, 16, dtype=torch.bfloat16)
    full = owner.routed_workspace(value, 42)
    full.fill_(9)
    owner.shape_plans("full")[64] = SimpleNamespace(plan=SimpleNamespace(
        invalidate=lambda: events.append("invalidate")))
    tail = owner.routed_workspace(value, 18)
    assert full.data_ptr() == tail.data_ptr() and tail.shape == (19, 16)
    tail[-1].fill_(-1)
    assert bool((full[18] == -1).all()) and bool((full[19:] == 9).all())
    assert not events and list(owner.plans) == ["full"]
    grown = owner.routed_workspace(value, 84)
    assert events == ["drain", "invalidate"] and not owner.plans
    assert grown.data_ptr() != full.data_ptr()


def test_shape_lru_retires_all_buckets_after_their_consumers(monkeypatch):
    events = []
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: events.append("drain"))
    owner = buckets._BucketedPrefillPlans()
    for shape in (7, 3):
        for row in (32, 64):
            owner.shape_plans(shape)[row] = SimpleNamespace(plan=SimpleNamespace(
                invalidate=lambda s=shape, r=row: events.append((s, r))))
    owner.shape_plans(7)
    owner.shape_plans(5)
    assert list(owner.plans) == [7, 5]
    assert events == ["drain", (3, 32), (3, 64)]
    owner.close()
    assert events[3:] == ["drain", (7, 32), (7, 64)] and not owner.plans


@pytest.mark.parametrize("interleaved", [False, True])
def test_alternating_shapes_rebind_inputs_and_reduce_only_live_routes(monkeypatch, interleaved):
    from vllm_gaudi import envs
    from vllm_gaudi.ops import deepseek_v41_grouped_prefill as grouped
    from vllm_gaudi.ops import deepseek_v41_prefill_columns as columns
    from vllm_gaudi.ops import deepseek_v41_prefill_plan as plans

    prepared, replayed = [], []

    class Plan:

        def __init__(self, body, arguments, groups, require_prefix):
            self.recipes = len(groups)
            self.plan = SimpleNamespace(invalidate=lambda: None)
            self.replays = 0
            prepared.append(self)
            self.write(arguments)

        def write(self, args):
            value, route, slots, experts, weight = args[:5]
            workspace = args[10]
            for block, expert in enumerate(experts.flatten()):
                for slot in slots[block]:
                    if slot >= 0:
                        workspace[slot] = value[slot // 6] * route.flatten()[slot] * weight[expert]
                    else:
                        workspace[-1].fill_(123)

        def replay(self, args, groups):
            replayed.append(self)
            self.replays += 1
            self.write(args)
            return groups

    monkeypatch.setattr(plans, "PrefillExpertPlan", Plan)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: SimpleNamespace(hpu_stream=0))
    monkeypatch.setattr(torch.hpu, "default_stream", lambda: SimpleNamespace(hpu_stream=0))
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE", interleaved)
    monkeypatch.setattr(buckets, "_compiled_routes", lambda key: buckets.device_bucketed_route_blocks)
    for module, names in ((grouped, ("compiled_write_body", "compiled_reduce")),
                          (columns, ("compiled_permuted_write_body", "compiled_permuted_reduce"))):
        monkeypatch.setattr(module, names[0], lambda key: None)
        monkeypatch.setattr(module, names[1], lambda key: lambda rows: rows.sum(1))
    owner = buckets._BucketedPrefillPlans()
    for generation, tokens in enumerate((7, 3, 7, 3)):
        value = torch.full((tokens, 16), generation + 1, dtype=torch.bfloat16)
        ids = (torch.arange(tokens * 6, dtype=torch.int32).reshape(tokens, 6) + generation) % 4
        route = torch.full((tokens, 6), 0.5)
        weight = torch.arange(1 + generation, 5 + generation, dtype=torch.bfloat16)
        other = torch.zeros(4, 4)
        output = owner(value, ids, route, weight, other, other, other, other, True)
        expected = value * (route * weight[ids.long()]).sum(1, keepdim=True)
        assert torch.equal(output, expected)
    assert len(prepared) == 2 and replayed == prepared
    assert owner.workspace.shape == (43, 16)
