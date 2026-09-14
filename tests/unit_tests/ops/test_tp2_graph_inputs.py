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


def test_state_allocation_is_held_without_copy_and_replacement_invalidates_replay():
    first = roots()
    state = torch.arange(32, dtype=torch.uint8)
    first["state_tensors"] = (state, )
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["positions"], state]])
    assert bindings.state_tensors[0] is state
    state[4] = 19
    assert bindings.updates(first) == []
    assert bindings.updates(dict(first, state_tensors=(state.clone(), ))) is None
    assert bindings.updates(dict(first, state_tensors=())) is None
    assert state[4].item() == 19


def test_pool_pointer_facade_cannot_skip_changed_offset(monkeypatch):
    first = roots()
    pool = torch.arange(8).float()
    first["hidden_states"] = pool[:4].view(1, 4)
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["hidden_states"]]])
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda value: value.untyped_storage().data_ptr())
    changed = dict(first, hidden_states=pool[4:].view(1, 4))
    assert bindings.updates(changed) is None
    assert torch.equal(pool, torch.arange(8).float())


def test_v4_pack_copies_once_and_updates_all_captured_metadata_views():
    first = roots()
    first.update(metadata={},
                 input_ids=torch.tensor([1]),
                 state_generation=3,
                 metadata_pack=torch.arange(12, dtype=torch.int32))
    first["attention_inputs"] = (first["metadata_pack"][:4], first["metadata_pack"][4:])
    captured = [[first["input_ids"], *first["attention_inputs"]]]
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, captured)
    updated = dict(first, input_ids=torch.tensor([9]), metadata_pack=torch.arange(12, dtype=torch.int32) + 20)
    updates = bindings.updates(updated)
    assert len(updates) == 2
    bindings.apply(updates)
    assert captured[0][0].item() == 9
    assert torch.equal(captured[0][2], torch.arange(4, 12, dtype=torch.int32) + 20)
    assert bindings.updates(dict(updated, state_generation=4)) is None


def test_fixed_metadata_destination_is_independent_of_h2d_ring_reuse():
    first = roots()
    ring = [torch.arange(12, dtype=torch.int32), torch.arange(12, dtype=torch.int32) + 20]
    destination = ring[0].clone()
    first.update(metadata={},
                 metadata_pack=ring[0],
                 metadata_destination=destination,
                 attention_inputs=(destination[:4], destination[4:]))
    captured = [[*first["attention_inputs"]]]
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, captured)
    updates = bindings.updates(dict(first, metadata_pack=ring[1]))
    assert len(updates) == 1
    bindings.apply(updates)
    # Reusing a retired transfer slot cannot overwrite the later token's
    # currently consumed capture destination.
    ring[0].fill_(99)
    assert torch.equal(captured[0][1], torch.arange(4, 12, dtype=torch.int32) + 20)
    bindings.apply(bindings.updates(dict(first, metadata_pack=ring[0])))
    assert torch.equal(destination, torch.full((12, ), 99, dtype=torch.int32))


def test_nested_v4_metadata_fields_are_dynamic():
    first = roots()
    first["metadata"] = {"layer0": SimpleNamespace(lengths=torch.tensor([5]))}
    # Dataclass metadata is the model runner's normal nested representation.
    cls = make_dataclass("LayerMetadata", [("lengths", torch.Tensor)])
    first["metadata"]["layer0"] = cls(torch.tensor([5]))
    captured = first["metadata"]["layer0"].lengths
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[captured]])
    updated = dict(first, metadata={"layer0": cls(torch.tensor([11]))})
    bindings.apply(bindings.updates(updated))
    assert captured.item() == 11


def test_pointer_facade_copies_from_new_storage(monkeypatch):
    first = roots()
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, [[first["hidden_states"]]])
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda value: 0)
    updated = dict(first, hidden_states=torch.full((1, 4), 8.0))
    bindings.apply(bindings.updates(updated))
    assert torch.equal(first["hidden_states"], updated["hidden_states"])


def test_batched_engram_views_and_position_sources_keep_capture_destinations_fixed():
    first = roots()
    first["metadata"] = {}
    first["input_ids"] = torch.zeros(6, dtype=torch.int64)
    first["positions"] = torch.zeros(6, dtype=torch.int32)
    first["attention_inputs"] = tuple(torch.zeros(6, 2, 33, dtype=torch.uint8) for _ in range(2))
    captured = [[first["positions"], first["input_ids"], *first["attention_inputs"]]]
    bindings = FixedDecodeInputs(torch.nn.Identity(), first, captured)
    for generation in range(4):
        pair = torch.full((2, 6, 2, 33), generation, dtype=torch.uint8)
        pair[1].add_(10)
        control = torch.arange(7, dtype=torch.int64) + generation
        updated = dict(first,
                       input_ids=control[:6],
                       positions=torch.arange(6, dtype=torch.int32) + generation,
                       attention_inputs=(pair[0], pair[1]))
        changes = bindings.updates(updated)
        assert len(changes) == 4
        bindings.apply(changes)
        assert torch.equal(first["positions"], updated["positions"])
        assert torch.equal(first["input_ids"], control[:6])
        for index in range(2):
            assert torch.equal(first["attention_inputs"][index], pair[index])
        pair.fill_(255)
        assert not (first["attention_inputs"][0] == 255).any()


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
    monkeypatch.setattr(replay, "invalidate_prepared_group_plans", lambda **kwargs: invalidated.append(True))
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


def test_segmented_native_replay_stages_root_then_attention_inputs(monkeypatch):
    from vllm_gaudi.ops import tp2_prepared_plan as replay

    metadata = SimpleNamespace(is_prompt=False,
                               direct_gdn_state=True,
                               block_size=16,
                               native_completion=None)
    first = dict(positions=torch.tensor([3]),
                 input_ids=torch.tensor([7]),
                 attention_inputs=(torch.tensor([11]), torch.tensor([12])),
                 metadata=metadata,
                 state_generation=5,
                 state_tensors=(),
                 adapter=SimpleNamespace(name="deepseek_v41_pp0_input"))
    owner = torch.nn.Identity()
    captured = [[first["positions"], first["input_ids"], *first["attention_inputs"]]]
    bindings = FixedDecodeInputs(owner, first, captured)
    calls = []

    class Graph:

        def stage_fixed_inputs(self, sources, destinations):
            calls.append(("stage", tuple(value.item() for value in sources), len(destinations)))

        def replay_fixed_prefix(self):
            calls.append(("prefix", ))

        def replay_fixed_finish_with_completion(self):
            calls.append(("finish", ))
            return "completion"

    communicator = object()
    graph = Graph()
    replay._native_entries[owner] = graph, bindings, ("output", ), communicator
    monkeypatch.setattr(replay, "_runtime", lambda: (None, communicator))
    updated = dict(first,
                   positions=torch.tensor([4]),
                   input_ids=torch.tensor([8]),
                   attention_inputs=(torch.tensor([21]), torch.tensor([22])))
    assert replay.begin_segmented_native_decoder(owner, **updated) == ("output", )
    assert replay.finish_segmented_native_decoder(owner, **updated) == ("output", )
    assert calls == [("stage", (4, 8), 2), ("prefix", ),
                     ("stage", (21, 22), 2), ("finish", )]
    assert metadata.native_completion == "completion"
    assert owner not in replay._segmented_native_entries
    replay._native_entries.pop(owner)
