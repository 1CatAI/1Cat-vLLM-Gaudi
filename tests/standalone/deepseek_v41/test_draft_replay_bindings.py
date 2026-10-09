# SPDX-License-Identifier: Apache-2.0
"""Input and retirement contracts for the shared C6 control-plan owner."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_draft_replay as replay


class Draft(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tensor_parallel_size = tp
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
        for layer in self.layers:
            layer.attention = torch.nn.Module()
            layer.attention.register_buffer("swa", torch.zeros(6, 8, dtype=torch.bfloat16))
        self.register_buffer("weight", torch.ones(8))

    def verify_and_propose(self, hidden, proposed, control, auxiliary, positions):
        for layer in self.layers:
            layer.attention.swa.copy_(auxiliary)
        record = torch.cat((control, proposed))
        wire = record.bfloat16()
        target = (hidden.float() * self.weight).sum(-1) + positions
        confidence = auxiliary.float().sum(-1)[:5]
        return record, wire, target, confidence


@pytest.mark.parametrize("tp", (2, 4))
@torch.inference_mode()
def test_hot_inputs_and_generation_use_one_plan_without_rebinding_weights(monkeypatch, tp):
    captures, plans, compilations, context = [], {}, [], {}

    @contextmanager
    def collect(**roots):
        context.update(roots)
        captures.append(roots)
        yield roots
        plans[roots["owner"]] = roots
        context.clear()

    def record(*outputs):
        context["outputs"] = outputs

    def native(owner, **roots):
        if owner not in plans:
            return None
        values = owner.draft.verify_and_propose(roots["hidden_states"], roots["input_ids"],
                                               roots["metadata"].control, roots["metadata"].auxiliary,
                                               roots["positions"])
        return values[0], None, *values[1:]

    def compile_entry(function, **kwargs):
        compilations.append(function)
        return function

    invalidations = []

    def invalidate(*, owner, reason):
        invalidations.append(reason)
        plans.pop(owner, None)

    monkeypatch.setattr(replay, "collect_prepared_group_replays", collect)
    monkeypatch.setattr(replay, "record_native_decoder_outputs", record)
    monkeypatch.setattr(replay, "replay_native_decoder", native)
    monkeypatch.setattr(replay, "invalidate_prepared_group_plans", invalidate)
    monkeypatch.setattr(torch, "compile", compile_entry)
    draft = Draft(tp)
    generation = SimpleNamespace(value=1)
    owner = replay.NativeDraftProtocol(draft, generation=lambda: generation.value)
    assert owner.adapter.collectives == 18 and owner.draft.weight is draft.weight
    for epoch in range(2):
        generation.value = epoch + 1
        for accepted in range(1, 7):
            hidden = torch.full((6, 8), float(accepted + epoch)).bfloat16()
            auxiliary = -hidden
            control = torch.tensor([epoch + 1, 6, 5, 100, 16384, 20000, accepted])
            proposed = torch.arange(5) + accepted
            positions = torch.arange(6, dtype=torch.int32) + 16384 + epoch * 6
            expected = tuple(value.clone() for value in draft.verify_and_propose(
                hidden, proposed, control, auxiliary, positions))
            actual = owner(hidden, proposed, control, auxiliary, positions)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert len(captures) == len(compilations) == 2
    assert invalidations == ["draft_protocol_generation"]
    assert set(captures[-1]["metadata"].__dict__) == {"control", "auxiliary", "native_completion"}
    assert captures[-1]["state_tensors"] == tuple(layer.attention.swa for layer in draft.layers)
    owner.close()
    with pytest.raises(RuntimeError, match="retired"):
        owner(hidden, proposed, control, auxiliary, positions)


def test_c6_plan_rejects_a_different_protocol_before_capture():
    owner = replay.NativeDraftProtocol(Draft(4), generation=1)
    with pytest.raises(ValueError, match="C6 hidden"):
        owner(torch.zeros(2, 8), torch.zeros(5), torch.zeros(7), torch.zeros(6, 8), torch.zeros(6))


@pytest.mark.parametrize("enabled,sampled", ((False, True), (True, False), (True, True)))
def test_batched_input_capability_is_scoped_to_sampled_owner(monkeypatch, enabled, sampled):
    from vllm_gaudi.ops import tp2_prepared_plan as prepared
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK_BATCH_INPUT_STAGING", str(int(enabled)))
    calls = []
    graph = SimpleNamespace(enable_batched_input_staging=lambda: calls.append("batch"))
    owner = replay.NativeDraftProtocol(Draft(4), generation=1)
    owner.sampled = sampled
    monkeypatch.setattr(prepared, "_native_entries", {owner: (graph, None, None, None)})
    owner.require_ready()
    assert calls == (["batch"] if enabled and sampled else [])


@pytest.mark.parametrize("complete_capture", (False, True))
@torch.inference_mode()
def test_prepare_preserves_prefix_state_and_requires_a_complete_second_capture(monkeypatch, complete_capture):
    from vllm_gaudi.ops import tp2_prepared_plan as prepared

    entries, calls = {}, []
    monkeypatch.setattr(prepared, "_native_entries", entries)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(replay, "replay_native_decoder", lambda *args, **kwargs: None)
    context = {}

    @contextmanager
    def collect(**roots):
        calls.append(tuple(t.clone() for t in roots["state_tensors"]))
        context.update(roots)
        yield roots
        if complete_capture and len(calls) == 2:
            entries[roots["owner"]] = object()
        context.clear()

    monkeypatch.setattr(replay, "collect_prepared_group_replays", collect)
    monkeypatch.setattr(replay, "record_native_decoder_outputs", lambda *values: context.update(outputs=values))
    draft = Draft(4)
    owner = replay.NativeDraftProtocol(draft, generation=1)
    inputs = (torch.ones(6, 8).bfloat16(), torch.arange(5), torch.arange(7),
              torch.full((6, 8), -3.).bfloat16(), torch.arange(6, dtype=torch.int32))
    if complete_capture:
        owner.prepare(*inputs)
    else:
        with pytest.raises(RuntimeError, match="complete native plan"):
            owner.prepare(*inputs)
    assert len(calls) == 2
    assert all(torch.count_nonzero(value) == 0 for saved in calls for value in saved)
    assert all(torch.count_nonzero(layer.attention.swa) == 0 for layer in draft.layers)
