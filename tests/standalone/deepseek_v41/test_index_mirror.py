# SPDX-License-Identifier: Apache-2.0
"""Derived keys must follow canonical page ownership and exact FP4 values."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_index_mirror import index_mirror_execution_mode, initialize_index_mirror
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors
from vllm_gaudi.ops.deepseek_v41_state import PagedStageState


def fixture(monkeypatch, ratio=2, length=1024, tp=4, runtime=False):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.ops.custom_op, "deepseek_v41_decoded_index_capacity", lambda: 32768, raising=False)
    program = torch.nn.Module()
    program.pp_rank, program.generation, program.replay_owner = 0, 0, None
    program.length = program.search_length = length
    program.register_buffer("swa", torch.zeros(4, dtype=torch.uint8))
    shared = program.shared = torch.nn.Module()
    shared.length, shared.runtime_indexer = length, runtime
    shared.register_buffer("block_table", torch.zeros(length // 128, dtype=torch.int32))
    cache = torch.nn.Module()
    cache.ratio = ratio
    cache.register_buffer("main", torch.empty(0, 288, dtype=torch.uint8))
    cache.register_buffer("index", torch.empty(0, 68, dtype=torch.uint8))
    shared.sources = torch.nn.ModuleDict({"2": cache})
    for name in ("prepare_index_mirror", "invalidate_index_mirror"):
        setattr(shared, name, MethodType(getattr(PagedCSA2SharedState, name), shared))
    initialize_index_mirror(shared, tp, "cpu")
    state = PagedStageState(program)
    state.allocate(length // 128 + 4, "cpu")
    generator = torch.Generator().manual_seed(815)
    cache.index.copy_(pack_fp4(torch.randn(cache.index.shape[0], 128, generator=generator).bfloat16(), 32))
    return program, shared, cache, state


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("visible", [0, 255, 256, 259, 1024])
def test_restore_nonidentity_pages_and_zero_future_rows(monkeypatch, ratio, visible):
    _, shared, cache, state = fixture(monkeypatch, ratio)
    pages = [4, 2, 5, 1, 8, 6, 7, 3]
    state.activate("a", pages)
    cache.index_mirror.fill_(99)
    shared.prepare_index_mirror(visible)
    rows = torch.arange(shared.length // ratio)
    width = 128 // ratio
    physical = torch.tensor(pages)[rows // width] * width + rows % width
    expected = unpack_fp4(cache.index[physical], 128, 32)
    expected[rows >= visible // ratio] = 0
    assert torch.equal(cache.index_mirror, expected)
    assert shared.index_mirror_rebuilds == 1
    # Repeated normal decode must preserve newly maintained values.
    cache.index_mirror[0].fill_(17)
    shared.prepare_index_mirror(visible)
    assert shared.index_mirror_rebuilds == 1 and (cache.index_mirror[0] == 17).all()


@pytest.mark.parametrize("bad_page", [-1, 2147483647])
def test_restore_rejects_invalid_physical_pages_without_integer_wrap(monkeypatch, bad_page):
    _, shared, cache, state = fixture(monkeypatch)
    state.activate("a", [1, 2, 3])
    shared.block_table[1] = bad_page
    shared.prepare_index_mirror(384)
    assert not cache.index_mirror[64:128].any()


def test_ownership_switch_rebuilds_without_snapshotting_derived_cache(monkeypatch):
    program, shared, cache, state = fixture(monkeypatch)
    state.activate("a", [1, 2, 3])
    shared.prepare_index_mirror(256)
    expected_a = cache.index_mirror.clone()
    program.swa.fill_(7)
    state.activate("b", [4, 5, 6])
    assert not shared.index_mirror_valid
    shared.prepare_index_mirror(256)
    assert not torch.equal(cache.index_mirror, expected_a)
    program.swa.fill_(9)
    state.activate("a", [1, 2, 3])
    assert not shared.index_mirror_valid and (program.swa == 7).all()
    shared.prepare_index_mirror(256)
    assert torch.equal(cache.index_mirror, expected_a)
    assert all("index_mirror" not in name for name in state.working)
    assert all("index_mirror" not in name for saved in state.saved.values() for name in saved)


@pytest.mark.parametrize("transition", ["reset", "remap", "release", "pool", "clear"])
def test_state_transitions_invalidate_before_next_consumer(monkeypatch, transition):
    _, shared, _, state = fixture(monkeypatch)
    state.activate("a", [1, 2])
    shared.prepare_index_mirror(128)
    if transition == "reset":
        state.activate("a", [1, 2], reset=True)
    elif transition == "remap":
        state.activate("a", [3, 4])
    elif transition == "release":
        state.release("a")
    elif transition == "pool":
        state.allocate(12, "cpu")
    else:
        state.clear()
    assert not shared.index_mirror_valid


def test_unchanged_or_preinstalled_identity_pages_keep_live_derived_rows(monkeypatch):
    _, shared, _, state = fixture(monkeypatch)
    state.activate("a", [1, 2])
    shared.prepare_index_mirror(128)
    state.activate("a", [1, 2])
    state.activate("a", [1, 2, 3])
    state.release("unrelated")
    assert shared.index_mirror_valid and shared.index_mirror_rebuilds == 1


def test_replay_snapshot_contains_mirror_only_when_active(monkeypatch):
    program, shared, cache, state = fixture(monkeypatch)
    state.activate("a", [1])
    shared.prepare_index_mirror(128)
    assert any(value is cache.index_mirror for value in stage_state_tensors(program))
    program.search_length = 65536
    assert all(value is not cache.index_mirror for value in stage_state_tensors(program))
    program.search_length = 1024
    shared.invalidate_index_mirror()
    assert all(value is not cache.index_mirror for value in stage_state_tensors(program))


@pytest.mark.parametrize("tp,runtime", [(2, False), (4, True)])
def test_shared_runtime_contracts_allocate_mirror(monkeypatch, tp, runtime):
    _, shared, cache, _ = fixture(monkeypatch, tp=tp, runtime=runtime)
    assert shared.index_mirror_tokens == 1024 and hasattr(cache, "index_mirror")


def test_capacity_bound_retains_long_context_fallback(monkeypatch):
    _, shared, cache, _ = fixture(monkeypatch, length=65536)
    assert shared.index_mirror_tokens == 32768
    assert cache.index_mirror.shape == (16384, 128)
    with pytest.raises(ValueError, match="bounded capacity"):
        shared.prepare_index_mirror(32769)


def test_segmented_prefix_and_native_replay_distinguish_index_state():
    from vllm_gaudi.models.deepseek_v41_program import CompiledStage
    from vllm_gaudi.ops.deepseek_v41_replay import StageReplay
    owner = SimpleNamespace(generation=1, search_length=32768, decode_token_bound=20480,
                            shared=SimpleNamespace(index_mirror_tokens=32768, index_mirror_valid=False))
    prefix = SimpleNamespace(owner=owner, prefix_groups=1)
    prefix._prefix_key = MethodType(CompiledStage._prefix_key, prefix)
    prefix.prefix_generation = prefix._prefix_key(32768)
    replay = object.__new__(StageReplay)
    replay.program = lambda: owner
    packed_key = replay._input_key()
    assert index_mirror_execution_mode(owner) is False
    owner.shared.index_mirror_valid = True
    assert not CompiledStage.prefix_ready(prefix, 32768)
    mirror_key = replay._input_key()
    assert mirror_key != packed_key
    assert index_mirror_execution_mode(owner, 65536) is False
    owner.shared.index_mirror_valid = False
    assert replay._input_key() == packed_key
    assert CompiledStage.prefix_ready(prefix, 32768)


def test_v2_prefix_defers_until_page_restore_is_ready(monkeypatch):
    from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import V41V2ModelRunner
    from vllm_gaudi.v1.worker.deepseek_v41_runner import decode_source_prefix_bound, target_search_length
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner = object.__new__(V41V2ModelRunner)
    shared = SimpleNamespace(index_mirror_tokens=32768, index_mirror_valid=False)
    search = target_search_length(16385, 1, 1048576)
    bound = decode_source_prefix_bound(16386, search, 4, runtime_indexer=False)
    runner.model = SimpleNamespace(tensor_parallel_size=4, decode_prefix_ready=lambda length: True,
                                  program=SimpleNamespace(length=1048576, search_length=search,
                                                          decode_token_bound=bound, shared=shared))
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True))
    runner.audit = {}
    record = SimpleNamespace(request_id="a", start=16384)
    scheduled = SimpleNamespace(finished_req_ids=set(), num_scheduled_tokens={"a": 1},
                                scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], resumed_req_ids=set(),
                                                                    num_computed_tokens=[16385], new_block_ids=[None]))
    assert not runner._prefix_authorized(record, scheduled)
    assert runner.audit["v2_prefix_index_restores"] == 1
    shared.index_mirror_valid = True
    assert runner._prefix_authorized(record, scheduled)


def test_shared_native_graph_does_not_reuse_a_shorter_visible_prefix():
    from vllm_gaudi.ops.deepseek_v41_replay import StageReplay
    from vllm_gaudi.ops import tp2_prepared_plan
    owner = SimpleNamespace(generation=1, search_length=32768, decode_token_bound=20480,
                            shared=SimpleNamespace(index_mirror_tokens=32768, index_mirror_valid=True))
    replay = object.__new__(StageReplay)
    replay.program = lambda: owner
    key = replay._input_key()
    variant = torch.nn.Module()
    replay.variants = {key: variant}
    tp2_prepared_plan._native_entries[variant] = object()
    try:
        assert replay.input_variant_ready(32768)
        owner.decode_token_bound = 24576
        assert replay._input_key() != key
        assert not replay.input_variant_ready(32768)
        owner.decode_token_bound = 20480
        assert replay.input_variant_ready(32768)
    finally:
        tp2_prepared_plan._native_entries.pop(variant, None)
