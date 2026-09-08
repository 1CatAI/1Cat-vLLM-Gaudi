# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from collections import namedtuple
from dataclasses import make_dataclass

import torch
import pytest

from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs


def roots():
    return dict(positions=torch.tensor([3]),
                hidden_states=torch.ones(1, 4),
                residual=torch.zeros(1, 4),
                metadata=SimpleNamespace(is_prompt=False,
                                         direct_gdn_state=True,
                                         block_size=16,
                                         lengths=torch.tensor([7])))


def test_updates_metadata_views_without_copying_weights_or_stable_inputs():
    model = torch.nn.Linear(4, 4)
    first = roots()
    captured = [[
        model.weight, first["positions"], first["hidden_states"], first["residual"],
        first["metadata"].lengths.view(1, 1)
    ]]
    bindings = FixedDecodeInputs(model, first, captured)
    next_roots = dict(first, metadata=SimpleNamespace(**vars(first["metadata"])))
    next_roots["metadata"].lengths = torch.tensor([8])
    updates = bindings.updates(next_roots)
    assert len(updates) == 1
    bindings.apply(updates)
    assert captured[0][-1].item() == 8
    assert all(binding.destination is not model.weight for binding in bindings.bindings)


@pytest.mark.parametrize("kind", ["namedtuple", "slots_dataclass"])
def test_processed_metadata_without_dict_is_bound(kind):
    names = ("is_prompt", "direct_gdn_state", "block_size", "lengths")
    cls = (namedtuple("ProcessedMetadata", names)
           if kind == "namedtuple" else make_dataclass("ProcessedMetadata", names, slots=True))
    first = roots()
    first["metadata"] = cls(False, True, 16, torch.tensor([7]))
    captured = first["metadata"].lengths.view(1, 1)
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[captured]])
    updated = dict(first, metadata=cls(False, True, 16, torch.tensor([13])))
    bindings.apply(bindings.updates(updated))
    assert captured.item() == 13


def test_shape_change_is_rejected_before_any_input_mutation():
    first = roots()
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["positions"], first["hidden_states"]]])
    changed = dict(first, positions=torch.tensor([9]), hidden_states=torch.ones(2, 4))
    assert bindings.updates(changed) is None
    assert first["positions"].item() == 3


def test_different_views_in_one_rotary_storage_are_both_updated():
    first = roots()
    model = torch.nn.Identity()
    cache = torch.arange(8).float()
    model.register_buffer("cos", cache[:4])
    model.register_buffer("sin", cache[4:])
    bindings = FixedDecodeInputs(model, first, [[model.cos, model.sin]])
    model.cos, model.sin = torch.full((4, ), 11.0), torch.full((4, ), 22.0)
    bindings.apply(bindings.updates(first))
    assert torch.equal(cache, torch.tensor([11.0] * 4 + [22.0] * 4))


def test_prefill_transition_invalidates_decode_bindings():
    first = roots()
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["positions"]]])
    changed = dict(first, metadata=SimpleNamespace(**{**vars(first["metadata"]), "is_prompt": True}))
    assert bindings.updates(changed) is None


def test_pool_pointer_facade_cannot_skip_changed_offset(monkeypatch):
    first = roots()
    pool = torch.arange(8).float()
    first["hidden_states"] = pool[:4].view(1, 4)
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["hidden_states"]]])
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda value: value.untyped_storage().data_ptr())
    changed = dict(first, hidden_states=pool[4:].view(1, 4))
    assert bindings.updates(changed) is None
    assert torch.equal(pool, torch.arange(8).float())


def test_pointer_facade_copies_from_new_storage(monkeypatch):
    first = roots()
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["hidden_states"]]])
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda value: 0)
    updated = dict(first, hidden_states=torch.full((1, 4), 8.0))
    bindings.apply(bindings.updates(updated))
    assert torch.equal(first["hidden_states"], updated["hidden_states"])


def test_direct_entry_updates_inputs_and_returns_graph_outputs(monkeypatch):
    from vllm_gaudi.ops import tp2_prepared_plan as replay

    first = roots()
    model = torch.nn.Identity()
    bindings = FixedDecodeInputs(model, first, [[first["hidden_states"], first["metadata"].lengths]])
    output = torch.zeros(1, 4)
    calls = []

    class Graph:

        def replay_fixed(self):
            calls.append("native")
            output.copy_(first["hidden_states"] + first["metadata"].lengths)

    communicator = object()
    monkeypatch.setattr(replay, "_runtime", lambda: (None, communicator))
    replay._native_entries[model] = Graph(), bindings, (output, first["residual"]), communicator
    updated = roots()
    updated["hidden_states"].fill_(5)
    updated["metadata"].lengths.fill_(9)
    assert torch.equal(replay.replay_native_decoder(model, **updated)[0], torch.full((1, 4), 14.0))
    assert calls == ["native"]
    replay._native_entries.pop(model)


def test_communicator_change_invalidates_before_copy_or_replay(monkeypatch):
    from vllm_gaudi.ops import tp2_prepared_plan as replay

    first = roots()
    model = torch.nn.Identity()
    bindings = FixedDecodeInputs(model, first, [[first["hidden_states"]]])
    invalidated = []
    replay._native_entries[model] = None, bindings, (), object()
    monkeypatch.setattr(replay, "_runtime", lambda: (None, object()))
    monkeypatch.setattr(replay, "invalidate_prepared_group_plans", lambda: invalidated.append(True))
    updated = dict(first, hidden_states=torch.full((1, 4), 9.0))
    assert replay.replay_native_decoder(model, **updated) is None
    assert invalidated == [True] and torch.equal(first["hidden_states"], torch.ones(1, 4))
    replay._native_entries.pop(model)


def test_native_failure_is_propagated_without_a_second_execution(monkeypatch):
    from vllm_gaudi.ops import tp2_prepared_plan as replay

    first = roots()
    model = torch.nn.Identity()
    bindings = FixedDecodeInputs(model, first, [[first["hidden_states"]]])
    calls = []

    class Graph:

        def replay_fixed(self):
            calls.append("mutation")
            raise RuntimeError("failed after mutation")

    communicator = object()
    monkeypatch.setattr(replay, "_runtime", lambda: (None, communicator))
    replay._native_entries[model] = Graph(), bindings, (), communicator
    with pytest.raises(RuntimeError, match="failed after mutation"):
        replay.replay_native_decoder(model, **first)
    assert calls == ["mutation"] and model not in replay._native_entries
