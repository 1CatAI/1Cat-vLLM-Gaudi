# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import (
    CompiledStage,
    PreparedInput,
    PreparedLayerGroup,
)
from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, capture_engram_inputs
from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V41_PP0, DEEPSEEK_V41_PP0_INPUT, DEEPSEEK_V41_PP1


class EchoGroup(torch.nn.Module):

    def __init__(self, native_input):
        super().__init__()
        self.native_input = native_input

    def forward(self, hidden, pre, positions, ids, engram):
        return hidden, pre, positions, ids, engram


def program(pp_rank=0, dspark=False, fp8=False, expert_n256=False):
    value = torch.nn.Module()
    value.pp_rank, value.dspark, value.fp8_decode = pp_rank, dspark, fp8
    value.expert_n256 = expert_n256
    return value


def test_native_group_replaces_only_input_arguments():
    embedding = torch.nn.Embedding(16, 5120, dtype=torch.bfloat16)
    incoming = PreparedInput(embedding, 0, lambda value: value)
    ids, positions, rows = torch.tensor([7]), torch.tensor([3], dtype=torch.int32), (object(), object())
    expected = incoming(ids)
    actual = PreparedLayerGroup.native_forward(EchoGroup(incoming), None, None, positions, ids, rows)
    assert all(torch.equal(value, reference) for value, reference in zip(actual[:2], expected))
    assert actual[2] is positions and actual[3] is ids and actual[4] is rows
    assert actual[0].shape == (1, 4, 5120) and actual[1].dtype == torch.float32


def test_ordinary_native_group_still_consumes_supplied_hidden():
    hidden, pre, positions, ids, rows = (object() for _ in range(5))
    actual = PreparedLayerGroup.native_forward(EchoGroup(None), hidden, pre, positions, ids, rows)
    assert actual == (hidden, pre, positions, ids, rows)


def test_embedding_special_token_mapping_is_preserved():
    incoming = PreparedInput(torch.nn.Embedding(64640, 4, dtype=torch.bfloat16), 1, lambda value: value)
    left, right = incoming(torch.tensor([129264])), incoming(torch.tensor([129265]))
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_topology_adds_embedding_without_dropping_other_reductions():
    assert DEEPSEEK_V41_PP0.collectives == 42
    assert DEEPSEEK_V41_PP0_INPUT.collectives == 43
    assert DEEPSEEK_V41_PP1.collectives == 40
    assert DEEPSEEK_V41_PP0_INPUT.reductions == 40
    assert not DEEPSEEK_V41_PP0_INPUT.external_prefix


@pytest.mark.parametrize("indexers", [0, 1, 2, 3])
def test_segmented_input_covers_long_index_exchanges(indexers):
    from dataclasses import replace
    value = replace(DEEPSEEK_V41_PP0_INPUT, extra_collectives=3 + 2 * indexers)
    assert value.supports_segmented_input
    assert value.reductions == 40 and value.collectives == 43 + 2 * indexers
    assert not replace(value, extra_collectives=value.extra_collectives + 1).supports_segmented_input
    assert not replace(value, external_prefix=True).supports_segmented_input
    assert not replace(value, name=DEEPSEEK_V41_PP1.name).supports_segmented_input


@pytest.mark.parametrize("search", [512, 1024, 16384, 1048576])
def test_paged_variants_bind_only_their_active_state_allocations(search):
    from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors
    owner = torch.nn.Module()
    owner.length, owner.search_length = 1048576, search
    names = ("swa", "main", "index", "indices", "candidate_pool", "kv_history", "score_history", "block_table",
             "decoded_swa", "decoded_main")
    for name in names:
        owner.register_buffer(name, torch.zeros(1))
    bound = {id(value) for value in stage_state_tensors(owner)}
    for name in names:
        active = search <= 512 or not name.startswith("decoded_")
        assert (id(getattr(owner, name)) in bound) == active


@pytest.mark.parametrize("kwargs", ({"pp_rank": 1}, {"dspark": True}, {"fp8": True}))
def test_incompatible_compile_contract_is_rejected(kwargs):
    with pytest.raises(ValueError, match="ordinary BF16 PP0"):
        CompiledStage(program(**kwargs), native=True, native_input=True)


def test_seeds_are_persistent_and_external_call_stays_separate(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1")

    class CaptureReplay(StageReplay):

        def __call__(self, *args, **kwargs):
            return args, kwargs

    owner = program()
    replay = CaptureReplay(owner)
    ids, positions = torch.tensor([1]), torch.tensor([0], dtype=torch.int32)
    first, mode = replay.from_input_ids(positions, ids, ())
    second, _ = replay.from_input_ids(positions, ids + 1, ())
    assert first[0] is second[0] and first[1] is second[1]
    assert mode == {"native_input": True}
    with pytest.raises(ValueError, match="C1 PP0"):
        replay.from_input_ids(positions.expand(6), ids.expand(6), ())


def test_segmented_input_replay_keeps_the_complete_variant(monkeypatch):
    from vllm_gaudi.ops import tp2_prepared_plan

    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    calls = []

    class CaptureSegmented(StageReplay):

        def begin_segmented_from_input_ids(self, positions, ids):
            calls.append(("prefix", positions, ids))

        def finish_segmented(self, positions, ids, engram):
            calls.append(("suffix", positions, ids, engram))
            return "output"

    owner = program()
    replay = CaptureSegmented(owner)

    class Variant:
        pass

    variant = Variant()
    replay.variants[(1, "input")] = variant
    tp2_prepared_plan._native_entries[variant] = object()
    ids, positions, engram = torch.tensor([7]), torch.tensor([3], dtype=torch.int32), (object(), object())
    try:
        assert replay.from_input_ids(positions, ids, engram) == "output"
    finally:
        tp2_prepared_plan._native_entries.pop(variant, None)
    assert calls == [("prefix", positions, ids), ("suffix", positions, ids, engram)]


def test_direct_engram_retains_exact_packed_views():
    packed = torch.arange(2 * 12 * 264, dtype=torch.int64).to(torch.uint8)
    views = packed[:12 * 264].view(1, 12, 264), packed[12 * 264:].view(1, 12, 264)
    captured = capture_engram_inputs(views, direct=True)
    assert all(a is b for a, b in zip(captured, views))
    copies = capture_engram_inputs(views)
    assert all(torch.equal(a, b) and a.data_ptr() != b.data_ptr() for a, b in zip(copies, views))
    for invalid in ((views[0], views[0]), tuple(reversed(views)), copies, views[:1], (views[0].transpose(1,
                                                                                                         2), views[1])):
        with pytest.raises(ValueError, match="Engram"):
            capture_engram_inputs(invalid, direct=True)


def test_device_engram_retains_decoded_and_late_inputs():
    layer1 = torch.empty((1, 12, 256), dtype=torch.bfloat16)
    layer14 = torch.empty((1, 12, 264), dtype=torch.uint8)
    captured = capture_engram_inputs((layer1, layer14), direct=True, device_layer1=True)
    assert captured[0] is layer1 and captured[1] is layer14
    invalid = ((layer1.to(torch.float32), layer14), (layer1[:, :, :255], layer14), (layer1, layer14.to(torch.int8)),
               (layer1, layer14[:, :, :263]), (layer1.transpose(1, 2), layer14))
    for values in invalid:
        with pytest.raises(ValueError, match="Device Engram"):
            capture_engram_inputs(values, direct=True, device_layer1=True)


def test_native_input_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", raising=False)
    owner = program()
    replay = StageReplay(owner)
    assert not replay.native_input_enabled
    with pytest.raises(ValueError, match="enabled"):
        replay.from_input_ids(torch.tensor([0]), torch.tensor([1]), ())


def test_direct_engram_does_not_require_a_pp0_input_graph_on_pp1(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT", "1")
    replay = StageReplay(program(pp_rank=1))
    assert not replay.native_input_enabled


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_direct_engram_accepts_n256_fp8_with_bf16_stage_boundaries(monkeypatch, pp_rank):
    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT", "1")
    replay = StageReplay(program(pp_rank=pp_rank, fp8=True, expert_n256=True))
    assert replay.native_input_enabled == (pp_rank == 0)


def test_input_modes_have_separate_cached_variants(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_replay

    class Variant:

        def __init__(self, *args, native_input=False):
            self.native_input = native_input

        def __call__(self, *args):
            return self

    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1")
    monkeypatch.setattr(deepseek_v41_replay, "StageVariant", Variant)
    owner = program()
    replay = StageReplay(owner)
    ids, positions = torch.tensor([1]), torch.tensor([0], dtype=torch.int32)
    external = replay(None, None, positions, ids, ())
    ids6, positions6 = torch.arange(6), torch.arange(6, dtype=torch.int32)
    with pytest.raises(ValueError, match="C1 when DSpark is disabled"):
        replay(None, None, positions6, ids6, ())
    internal = replay.from_input_ids(positions, ids, ())
    assert external is replay(None, None, positions, ids, ())
    assert internal is replay.from_input_ids(positions, ids + 1, ())
    assert set(replay.variants) == {1, (1, "input")}
    assert not external.native_input and internal.native_input


def test_native_input_variants_are_separate_across_paged_buckets(monkeypatch):
    from vllm_gaudi.ops import tp2_prepared_plan

    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1")
    owner = program()
    owner.search_length = 512
    replay = StageReplay(owner)
    assert replay._input_key() == (1, "input")
    owner.search_length = 1024
    assert replay._input_key() == (1, "input", 1024)

    class Variant:
        pass

    variant = Variant()
    replay.variants[(1, "input", 1024)] = variant
    assert not replay.input_variant_ready(512)
    assert not replay.input_variant_ready(1024)
    tp2_prepared_plan._native_entries[variant] = object()
    try:
        assert replay.input_variant_ready(1024)
        assert not replay.input_variant_ready(2048)
        assert owner.search_length == 1024
    finally:
        tp2_prepared_plan._native_entries.pop(variant, None)


@pytest.mark.parametrize("pp_rank,native_input", ((0, False), (0, True), (1, True)))
def test_warmup_checks_the_selected_mode(monkeypatch, pp_rank, native_input):
    from vllm_gaudi.ops import tp2_prepared_plan

    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH", "1" if native_input else "0")
    entries = {}
    monkeypatch.setattr(tp2_prepared_plan, "_native_entries", entries)
    owner = program(pp_rank=pp_rank)
    replay = StageReplay(owner)
    external, internal = object(), object()
    replay.variants.update({1: external, (1, "input"): internal})
    selected = internal if native_input and pp_rank == 0 else external
    entries[external if selected is internal else internal] = object()
    with pytest.raises(RuntimeError, match="warmup"):
        replay.require_ready(1)
    entries[selected] = object()
    replay.require_ready(1)


def test_input_capture_requires_native_compilation():
    with pytest.raises(ValueError, match="ordinary BF16 PP0"):
        CompiledStage(program(), native=False, native_input=True)
