# SPDX-License-Identifier: Apache-2.0
"""The B1 mirror and B2+ slots must share one canonical request history."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa
from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors


def program(device="cpu"):
    result = torch.nn.Module()
    result.shared = shared = torch.nn.Module()
    shared.register_buffer("block_table", torch.zeros(16, dtype=torch.int32, device=device))
    shared.register_buffer("decoded_swa", torch.zeros(512, 512, dtype=torch.bfloat16, device=device))
    shared.register_buffer("candidate_pool", torch.full((1, 2048), -1, dtype=torch.int32, device=device))
    cache = torch.nn.Module()
    cache.ratio = 2
    for name, shape, dtype in (("main", (32 * 64, 288), torch.uint8), ("index", (32 * 64, 68), torch.uint8),
                               ("decoded_main", (512, 512), torch.bfloat16), ("decoded_index_hot", (512, 128),
                                                                              torch.bfloat16)):
        cache.register_buffer(name, torch.zeros(shape, dtype=dtype, device=device))
    shared.sources = torch.nn.ModuleDict({"2": cache})
    selection = torch.nn.Module()
    selection.register_buffer("indices", torch.full((1, 512), -1, dtype=torch.int32, device=device))
    shared.topk = torch.nn.ModuleDict({"2": selection})
    attention = torch.nn.Module()
    attention.ratio, attention.owns_kv, attention.decoded_swa_offset = 2, True, 0
    attention.register_buffer("swa", torch.zeros(256, 528, dtype=torch.uint8, device=device))
    for name in ("kv_history", "score_history"):
        attention.register_buffer(name, torch.zeros(8, 512, dtype=torch.float32, device=device))
    layer = torch.nn.Module()
    layer.layer, layer.attention = 2, attention
    result.layers = torch.nn.ModuleList([layer])
    return result


@pytest.fixture
def bank(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *_a, **_k: self)
    return BatchStageState(program(), 2)


def populate(bank, owner, blocks, seed):
    generator = torch.Generator().manual_seed(seed)
    bank.publish_pages(owner, blocks, 32)
    state = bank.layers[2]
    state.swa[owner.index * 256:(owner.index + 1) * 256].copy_(
        pack_swa(torch.randn(256, 512, generator=generator).bfloat16()))
    state.kv_history[owner.index * 8:(owner.index + 1) * 8].fill_(seed)
    state.score_history[owner.index * 8:(owner.index + 1) * 8].fill_(-seed)
    cache = bank.program.shared.sources["2"]
    for name, width, group in (("main", 512, 16), ("index", 128, 32)):
        values = pack_fp4(torch.randn(len(blocks) * 64, width, generator=generator).bfloat16(), group)
        rows = torch.tensor([block * 64 + offset for block in blocks for offset in range(64)])
        getattr(cache, name).index_copy_(0, rows, values)


def check_mirror(bank, owner, position):
    shared = bank.program.shared
    attention = bank.program.layers[0].attention
    state = bank.layers[2]
    assert torch.equal(attention.swa, state.swa[owner.index * 256:(owner.index + 1) * 256])
    assert torch.equal(shared.decoded_swa[:256], unpack_swa(attention.swa))
    assert torch.equal(shared.block_table, bank.pages[owner.index])
    blocks = bank.page_versions[owner.index][1]
    rows = torch.tensor([blocks[row // 64] * 64 + row % 64 for row in range(position // 2)])
    cache = shared.sources["2"]
    for name, packed, width, group in (("decoded_main", cache.main, 512, 16), ("decoded_index_hot", cache.index, 128,
                                                                               32)):
        expected = unpack_fp4(packed.index_select(0, rows), width, group)
        assert torch.equal(getattr(cache, name)[:len(rows)], expected)
        assert not torch.count_nonzero(getattr(cache, name)[len(rows):])


def test_prefill_b1_b2_survivor_and_reused_slot(bank):
    a, b = bank.acquire("a"), bank.acquire("b")
    populate(bank, a, [5, 3, 8, 2, 9], 11)
    populate(bank, b, [6, 4, 10, 7, 11], 17)
    fixed_swa = bank.single_bindings[1][2]
    bank.bind_prefill(a)
    assert bank.program.layers[0].attention.swa is not fixed_swa
    bank.bind_single(a, 513)
    assert bank.program.layers[0].attention.swa is fixed_swa
    check_mirror(bank, a, 513)
    # Real B1 writers mutate their fixed replay allocation. B2 must see it.
    fixed_swa[3].fill_(13)
    bank.program.layers[0].attention.kv_history[4].fill_(31)
    generation = bank.begin([b, a])
    assert bank.single_owner is None
    assert torch.equal(bank.layers[2].swa[3], fixed_swa[3])
    assert torch.all(bank.layers[2].kv_history[4] == 31)
    bank.finish(generation, SimpleNamespace(synchronize=lambda: None, query=lambda: True))
    # A request in a nonzero slot survives the batch. Reconstruct its mirrors.
    bank.bind_single(b, 511)
    check_mirror(bank, b, 511)
    assert torch.all(bank.program.layers[0].attention.kv_history == 17)
    assert torch.all(bank.layers[2].kv_history[4] == 31)
    bank.release("a")
    c = bank.acquire("c")
    assert c.index == a.index and c.generation > a.generation
    assert not torch.count_nonzero(bank.layers[2].swa[:256])
    with pytest.raises(RuntimeError, match="Stale"):
        bank.bind_single(a, 513)
    populate(bank, c, [12, 15, 13], 23)
    bank.bind_single(c, 255)
    check_mirror(bank, c, 255)


def test_continuous_b1_never_recopies_working_state(bank, monkeypatch):
    a = bank.acquire("a")
    populate(bank, a, [1, 2, 3, 4, 5], 7)
    bank.bind_single(a, 513)
    bank.program.layers[0].attention.swa[1].fill_(99)
    bank.program.shared.sources["2"].decoded_main[0].fill_(19)
    monkeypatch.setattr(bank, "_restore_decoded", lambda *_a: pytest.fail("continuous B1 rebuilt mirrors"))
    for position in range(514, 520):
        bank.bind_single(a, position)
    assert torch.all(bank.program.layers[0].attention.swa[1] == 99)
    assert torch.all(bank.program.shared.sources["2"].decoded_main[0] == 19)
    bank.publish_pages(a, [1, 2, 3, 4, 5, 8], 32)
    bank.bind_single(a, 520)
    assert bank.program.shared.block_table[5] == 8


def test_handoff_audit_preserves_state_and_owns_cpu_snapshot(bank, tmp_path):
    import json
    from vllm_gaudi.ops.deepseek_v41_state_audit import save_single_handoff
    owner = bank.acquire("request/with/path")
    populate(bank, owner, [5, 3, 8, 2, 9], 11)
    bank.program.pp_rank, bank.program.tp_rank = 0, 1
    bank.bind_single(owner, 513)
    path = save_single_handoff(bank, owner, 513, "request/with/path", tmp_path, engram_history=[1, 2])
    metadata = json.loads(path.with_suffix(".json").read_text())
    saved = torch.load(path.with_suffix(".pt"), weights_only=True)
    assert path.parent == tmp_path
    assert metadata["mirrors_complete"] and metadata["blocks"] == [5, 3, 8, 2, 9]
    assert metadata["engram_history"] == [1, 2]
    assert saved["source2.decoded_main"].shape == (256, 512)
    check_mirror(bank, owner, 513)
    bank.program.layers[0].attention.swa.zero_()
    assert torch.count_nonzero(saved["layer2.swa"])
    bank.leave_single()
    with pytest.raises(RuntimeError, match="completed native B1 handoff"):
        save_single_handoff(bank, owner, 513, "invalid", tmp_path)


def test_native_b1_capture_excludes_unread_batch_banks(bank):
    tensors = stage_state_tensors(bank.program)
    assert any(value is bank.program.layers[0].attention.swa for value in tensors)
    assert all(value is not bank.layers[2].swa for value in tensors)


@pytest.mark.parametrize("search,active", [(2560, True), (4096, False)])
def test_native_hot_capture_declares_decoded_handoff_buffers(bank, search, active):
    bank.program.length = 1048576
    bank.program.search_length = search
    shared = bank.program.shared
    shared.decoded_kv_state = True
    cache = shared.sources["2"]
    cache.decoded_main = torch.zeros(1280, 512, dtype=torch.bfloat16)
    cache.decoded_index_hot = torch.zeros(1280, 128, dtype=torch.bfloat16)
    tensors = stage_state_tensors(bank.program)
    assert any(value is shared.decoded_swa for value in tensors) == active
    assert any(value is cache.decoded_main for value in tensors) == active
    assert any(value is cache.decoded_index_hot for value in tensors) == active
    assert all(value is not bank.layers[2].swa for value in tensors)


def test_prefill_and_release_publish_b1_before_alias_change(bank):
    a, b = bank.acquire("a"), bank.acquire("b")
    populate(bank, a, [1, 2, 3], 7)
    populate(bank, b, [4, 5, 6], 9)
    bank.bind_single(a, 255)
    bank.program.layers[0].attention.swa[7].fill_(31)
    bank.bind_prefill(b)
    assert torch.all(bank.layers[2].swa[7] == 31)
    bank.bind_single(a, 255)
    bank.release("a")
    assert bank.single_owner is None
    assert bank.acquire("c").index == a.index


@pytest.mark.parametrize("names", [("a", ), ("a", "b")])
def test_normal_scheduler_selects_native_b1_with_batch_capacity(names):
    from contextlib import nullcontext
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, V41ModelRunner
    runner = object.__new__(V41ModelRunner)
    runner.pending = runner.prefix_checkpoints = runner.draft_token_ids = None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True))
    runner.requests = {name: RequestState(name, [1], [], None, ([1], ), 1, [10]) for name in names}
    runner._update = lambda _: None
    calls = []

    def batch(requests):
        calls.append("batch")
        return ModelRunnerOutput(req_ids=list(names),
                                 req_id_to_index={
                                     n: i
                                     for i, n in enumerate(names)
                                 },
                                 sampled_token_ids=[[11]] * len(names))

    runner.request_batches = SimpleNamespace(execute=batch, trace_single=lambda _: nullcontext(), capacity=32)
    runner._execute_request = lambda *_: calls.append("native")
    runner._finish_request = lambda: ModelRunnerOutput(
        req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[11]])
    runner.execute_model(SimpleNamespace(num_scheduled_tokens=dict.fromkeys(names, 1)))
    assert calls == (["native"] if len(names) == 1 else ["batch"])
    assert runner.batch_result.sampled_token_ids == [[11]] * len(names)
