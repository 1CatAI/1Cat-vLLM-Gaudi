# SPDX-License-Identifier: Apache-2.0
"""CPU ownership contracts used by the normal V4.1 PP runner."""

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_state import PagedStageState, StageStateBlocks, V41StateSpec
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, greedy_verify


def test_control_cpu_sets_reject_worker_and_smt_overlap(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.ops import deepseek_v41_cpu as cpu
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_CPUS", "0,1")
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_HELPER_CPUS", "2;3")
    monkeypatch.setattr(cpu.envs, "VLLM_HPU_DSV41_ENGINE_CPUS", "4,5")
    monkeypatch.setattr(cpu.envs, "VLLM_HPU_DSV41_API_CPUS", "6")

    def topology(path):
        core = int(str(path).split("/cpu")[-1].split("/")[0]) % 100
        return SimpleNamespace(read_text=lambda: f"{core},{core + 100}")

    monkeypatch.setattr(cpu, "Path", topology)
    assert cpu.control_cpu_set("engine", set(range(8))) == {4, 5}
    assert cpu.control_cpu_set("api", set(range(8))) == {6}
    with pytest.raises(ValueError, match="launch affinity"):
        cpu.control_cpu_set("api", {4, 5})
    monkeypatch.setattr(cpu.envs, "VLLM_HPU_DSV41_API_CPUS", "102")
    with pytest.raises(ValueError, match="share a core"):
        cpu.control_cpu_set("engine", set(range(104)))


def test_packed_pp_preserves_raw_bits_and_rejects_early_consumer_reuse():
    from vllm_gaudi.ops.deepseek_v41_pp import PackedC1Buffers
    buffers = PackedC1Buffers("cpu")
    pointers = {value.data_ptr() for value in buffers.packets}
    for generation in range(7):
        bits = (torch.arange(20480, dtype=torch.int32) + generation * 20480).to(torch.int16)
        hidden = bits.view(torch.bfloat16).reshape(1, 5120, 4).transpose(1, 2)
        pre = torch.tensor([0, -2147483648, 2139095040, 2143289410], dtype=torch.int32)
        pre = pre.view(torch.float32).reshape(1, 4)
        packet, values = buffers.acquire()
        buffers.pack({"hidden_states": hidden, "pre_mix": pre})
        assert packet.data_ptr() in pointers
        assert torch.equal(values["hidden_states"].view(torch.int16), hidden.view(torch.int16))
        assert torch.equal(values["pre_mix"].view(torch.int32), pre.view(torch.int32))
        with pytest.raises(RuntimeError, match="consumer has not completed"):
            buffers.acquire()
        buffers.complete()
    assert buffers.completed == [7, 6]


def test_single_token_prefill_keeps_disjoint_transport_buffers(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.v1.worker import deepseek_v41_runner as runner
    sent = []
    monkeypatch.setattr(runner.envs, "VLLM_HPU_DSV41_PACKED_PP", True)
    monkeypatch.setattr(runner, "get_pp_group",
                        lambda: SimpleNamespace(is_first_rank=True, ranks=[0, 2], device_group=None))
    monkeypatch.setattr(runner.dist, "isend", lambda value, **kwargs:
                        (sent.append(value), SimpleNamespace(wait=lambda: None))[1])
    buffers = runner.PPBuffers("cpu", dspark=False)
    values = {
        "hidden_states": torch.zeros(1, 4, 5120, dtype=torch.bfloat16),
        "pre_mix": torch.zeros(1, 4, dtype=torch.float32)
    }
    buffers.exchange(values, 1, decode=False)
    assert len(sent) == 2 and buffers.packed.generation == 0
    assert sent[0].untyped_storage()._cdata != sent[1].untyped_storage()._cdata
    buffers.exchange(values, 1, decode=True)
    assert len(sent) == 3 and sent[-1].dtype == torch.int32
    assert buffers.packed.generation == 1
    buffers.drain()
    buffers.complete_packet()


def test_native_c1_binding_consumes_prompt_updates_and_rejects_foreign_request_prefix():
    from types import SimpleNamespace
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
    consumed = []
    prepared = []

    class Model:
        native = True

        def prepare_step(self, *args, **kwargs):
            prepared.append(kwargs)

        def __call__(self, ids, positions, **kwargs):
            consumed.append((ids.clone(), ids.dtype))
            return None

    runner = object.__new__(V41ModelRunner)
    runner.model, runner.use_dspark, runner.direct_token_ids = Model(), False, True
    runner.decode_ids = torch.empty(1, dtype=torch.int32)
    runner.input_views = {1: torch.empty(1, dtype=torch.int64)}
    runner.position_views = {1: torch.empty(1, dtype=torch.int32)}
    runner.position_bank = None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=True), exchange=lambda *a, **k: None)
    runner.audit = {"target_steps": 0, "target_tokens": 0}
    commit = torch.tensor([1, 1, 1, 42], dtype=torch.int32)
    token = commit[3:4]
    for start, value in ((1, 42), (2, 43)):
        token.fill_(value)
        runner._next_input = ("a", start, token)
        runner._forward("a", [99], start, decode=True)
    runner._forward("b", [7], 0, decode=False)
    runner._forward("a", [55], 8, decode=True)
    assert [item[0].item() for item in consumed] == [42, 43, 7, 55]
    assert [item[1] for item in consumed] == [torch.int32] * 4
    assert prepared[2]["is_decode"] and prepared[2]["use_replay"]


def test_position_bank_reuses_one_allocation_across_request_resets_and_prefill_tails():
    from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
    bank = PositionBank(512, 6, "cpu")
    result = torch.empty(6, dtype=torch.int32)
    pointer = result.data_ptr()
    for start, count in ((0, 6), (6, 1), (128, 3), (511, 1), (0, 1), (506, 6)):
        bank.copy_into(result[:count], start)
        assert result[:count].tolist() == list(range(start, start + count))
        assert result.data_ptr() == pointer
    for start, count in ((-1, 1), (512, 1), (510, 3)):
        with pytest.raises(ValueError, match="range/device"):
            bank.copy_into(result[:count], start)


def test_position_bank_uses_bounded_views_for_one_million_token_context():
    from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
    bank = PositionBank(1_048_576, 6, "cpu")
    assert bank._views is None
    result = torch.empty(6, dtype=torch.int32)
    for start, count in ((0, 6), (1024, 1), (1_048_570, 6), (1_048_575, 1)):
        bank.copy_into(result[:count], start)
        assert result[:count].tolist() == list(range(start, start + count))
    with pytest.raises(ValueError, match="range/device"):
        bank.copy_into(result, 1_048_571)


def test_engine_registers_opaque_state_without_worker_model_initialization(monkeypatch):
    from vllm.v1 import kv_cache_spec_registry as registry
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    monkeypatch.setenv("VLLM_HPU_DSV41_PREPARED_SHARDS", "1")
    monkeypatch.setattr(registry, "_REGISTRY_KVCACHESPEC_LIST", {})
    spec = V41StateSpec(block_size=512, state_shape=(512, 528), state_dtype=torch.uint8)
    registry.KVCacheSpecRegistry.check_kv_cache_spec_registry({"state": spec})
    assert registry.KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager


def test_pp_scheduler_reconciliation_does_not_append_broadcast_tokens_twice():
    state = RequestState("a", [10, 11], [], None, ([1], ), output=[12, 13])
    state.reconcile(2, [12, 13])
    assert state.tokens == [10, 11, 12, 13]
    state.reconcile(1, [12])
    assert state.tokens == [10, 11, 12]
    state.reconcile(3, [14, 15])
    assert state.tokens == [10, 11, 12, 14, 15]
    with pytest.raises(RuntimeError, match="missing committed"):
        state.reconcile(5, [16])


@pytest.mark.parametrize("target,draft,expected,accepted", [
    ([7, 8, 9], [7, 8], [7, 8, 9], 2),
    ([7, 9, 10], [7, 8], [7, 9], 1),
    ([6, 8, 9], [7, 8], [6], 0),
])
def test_verify_returns_only_accepted_prefix_and_target_correction(target, draft, expected, accepted):
    assert greedy_verify(target, draft) == (expected, accepted)


def test_scheduler_state_binding_preserves_shared_owner_and_invalidates_addresses():
    program = torch.nn.Module()
    program.pp_rank, program.generation, program.replay_owner = 0, 0, None
    program.cache = torch.nn.Module()
    program.cache.register_buffer("swa", torch.zeros(512, 528, dtype=torch.uint8))
    program.cache.register_buffer("indices", torch.zeros(512, 512, dtype=torch.int32))
    program.reader = program.cache
    state = StageStateBlocks(program)
    state.allocate(2, "cpu")
    state.bind(1)
    state.clear()
    assert program.reader.swa.data_ptr() == program.cache.swa.data_ptr()
    program.cache.swa[0, 0] = 41
    assert (program.cache.indices == -1).all()
    generation = program.generation
    state.bind(0)
    assert program.cache.swa[0, 0] == 0 and program.generation == generation + 1
    state.bind(1)
    assert program.cache.swa[0, 0] == 41
    assert state.allocated_bytes == 2 * (512 * 528 + 512 * 512 * 4)
    with pytest.raises(ValueError, match="block ID"):
        state.bind(2)
    spec = V41StateSpec(block_size=512, state_shape=(512, 528), state_dtype=torch.uint8)
    with pytest.raises(ValueError, match="cannot be split"):
        spec.copy_with_new_block_size(128)


def test_paged_state_reuses_pinned_block_table_and_only_publishes_changes(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)

    class BlockTable:

        def __init__(self):
            self.shape = (8, )
            self.device = torch.device("cpu")
            self.values = torch.empty(self.shape, dtype=torch.int32)
            self.copies = 0

        def numel(self):
            return self.values.numel()

        def zero_(self):
            self.values.zero_()

        def __setitem__(self, index, value):
            self.values[index] = value

        def copy_(self, source, non_blocking=False):
            assert non_blocking
            self.values.copy_(source)
            self.copies += 1

    program = torch.nn.Module()
    program.pp_rank, program.generation, program.replay_owner = 0, 0, None
    program.register_buffer("swa", torch.ones(2, dtype=torch.uint8))
    cache = SimpleNamespace(ratio=32)
    program.shared = SimpleNamespace(sources={0: cache}, block_table=BlockTable())
    state = PagedStageState(program)
    state.allocate(5, "cpu")

    state.activate("request", [1, 2], reset=True)
    assert program.shared.block_table.copies == 1
    assert program.shared.block_table.values.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    state.activate("request", [1, 2])
    assert program.shared.block_table.copies == 1
    state.activate("request", [2, 3, 4])
    assert program.shared.block_table.copies == 2
    assert program.shared.block_table.values.tolist() == [2, 3, 4, 0, 0, 0, 0, 0]
    state.activate("request", [1, 2, 3])
    assert program.shared.block_table.copies == 3
    assert program.shared.block_table.values.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_paged_state_saves_and_clears_decoded_working_set(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    program = torch.nn.Module()
    program.pp_rank, program.generation, program.replay_owner = 0, 0, None
    program.register_buffer("decoded_swa", torch.zeros(2, dtype=torch.bfloat16))
    source = torch.nn.Module()
    source.ratio = 2
    source.register_buffer("decoded_main", torch.zeros(2, dtype=torch.bfloat16))
    program.add_module("source", source)
    block_table = torch.empty(8, dtype=torch.int32)
    program.shared = SimpleNamespace(sources={0: source}, block_table=block_table)
    state = PagedStageState(program)
    state.allocate(5, "cpu")
    state.activate("a", [1], reset=True)
    program.decoded_swa.fill_(3)
    program.source.decoded_main.fill_(5)
    state.activate("b", [2], reset=True)
    assert not program.decoded_swa.any() and not program.source.decoded_main.any()
    state.activate("a", [1])
    assert (program.decoded_swa == 3).all() and (program.source.decoded_main == 5).all()


def test_target_capture_does_not_claim_draft_only_state():
    from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors
    program = torch.nn.Module()
    program.register_buffer("swa", torch.zeros(8, dtype=torch.uint8))
    program.draft = torch.nn.Module()
    program.draft.register_buffer("swa", torch.ones(8, dtype=torch.uint8))
    captured = stage_state_tensors(program)
    assert len(captured) == 1 and captured[0] is program.swa


def test_target_only_loading_never_reads_or_allocates_draft_tensors(tmp_path, monkeypatch):
    """The optional draft must not consume HBM or enter the C1 output graph."""
    import json
    from vllm_gaudi.models import deepseek_v41_program as program

    config = {"text_config": {"rms_norm_eps": 1e-20}}
    (tmp_path / "config.json").write_text(json.dumps(config))

    class Shard:

        def __init__(self, *args):
            self.specs = {
                "head.weight": {
                    "dtype": "BF16",
                    "shape": [3, 2]
                },
                "norm.weight": {
                    "dtype": "BF16",
                    "shape": [2]
                },
                "mtp.embed.weight": {
                    "dtype": "BF16",
                    "shape": [3, 2]
                }
            }
            self.specs.update({f"layers.{i}.weight": {"dtype": "BF16", "shape": [2]} for i in range(20, 40)})
            self.manifest = {"normal_scales": {f"layers.{i}.ffn.experts": [True, True] for i in range(20, 40)}}
            self.reads = []

        def tensor(self, name, device):
            self.reads.append(name)
            return torch.ones(self.specs[name]["shape"], dtype=torch.bfloat16, device=device)

        def check_identity(self):
            pass

    class Layer(torch.nn.Module):

        def __init__(self, weights, config, layer, *args, collect_target_state=False):
            super().__init__()
            self.layer, self.collect_target_state = layer, collect_target_state

        def forward(self, residual, pre, *args):
            assert not self.collect_target_state
            return residual, pre, None

    def unexpected_draft(*args):
        pytest.fail("Target-only initialization constructed a draft module")

    monkeypatch.setattr(program, "PreparedV41Shard", Shard)
    monkeypatch.setattr(program, "CSA2SharedState", lambda *args: torch.nn.Module())
    monkeypatch.setattr(program, "PreparedDecoderLayer", Layer)
    monkeypatch.setattr(program, "PreparedDraft", unexpected_draft)
    stage = program.PreparedStage(tmp_path, 1, 0, lambda x: x, lambda x, dim: x, "cpu", dspark=False)
    stage.load_prepared("cpu")
    assert stage.draft is None and not hasattr(stage.weights, "mtp")
    assert len(stage.layers) == 20 and len(stage.shard.reads) == 22
    assert not any(name.startswith("mtp.") for name in stage.shard.reads)
    output, _, aux = stage(torch.ones(1, 4, 2, dtype=torch.bfloat16), torch.full((1, 4), 0.25), torch.tensor([0]),
                           torch.tensor([1]), ())
    assert output.shape == (1, 2) and aux is None


def test_non_speculative_sampling_commits_exactly_one_real_token():
    from types import SimpleNamespace
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner._token_copy = None
    committed = []
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True),
                                device_commit=False,
                                finish_single=lambda consumed, token: (consumed, [token]),
                                commit=torch.tensor([1, 1, 1, 1]))
    runner.pp.commit_token = runner.pp.commit[3:4]
    runner.model = SimpleNamespace(complete_step=committed.append)
    state = RequestState("c1", [10], [], None, ([1], ))
    runner.pending = state, 1, 1, 1, [], True, torch.tensor([[1]])
    result = runner._sample_single()
    assert result.sampled_token_ids == [[1]] and state.output == [1]
    assert committed == [1] and runner.pending is None and runner.draft_token_ids is None
    assert runner._next_input[:2] == ("c1", 2) and runner._next_input[2].tolist() == [1]


def test_device_completion_waits_for_copy_and_rejects_stale_generation(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.distributed import tp2_fused_ar_norm as runtime
    from vllm_gaudi.v1.worker.deepseek_v41_runner import PPBuffers
    order = []
    buffers = object.__new__(PPBuffers)
    buffers.device_commit, buffers.dspark = True, False
    buffers.generation, buffers.commits, buffers.pending = 8, 0, []
    buffers.commit = torch.tensor([9, 1, 1, 42], dtype=torch.int32)
    buffers.commit_row = buffers.commit.view(1, 4)
    buffers.group = SimpleNamespace(broadcast=lambda *a, **k: order.append("broadcast"))
    buffers.packed = SimpleNamespace(complete=lambda: order.append("release"))

    def copy_record(value):
        order.append("copy")
        host = torch.full_like(value, -1)

        def synchronize():
            order.append("ready")
            host.copy_(value)

        return host, SimpleNamespace(synchronize=synchronize)

    bridge = SimpleNamespace(copy_integer_record_to_host=copy_record)
    monkeypatch.setattr(runtime, "_resolve_runtime", lambda: (bridge, None, None))
    assert buffers.finish_single_device() == (1, [42])
    assert order == ["broadcast", "copy", "ready", "release"] and buffers.commits == 1
    with pytest.raises(RuntimeError, match="Stale"):
        buffers.finish_single_device()
    assert order.count("release") == 1 and buffers.commits == 1


def test_device_sampling_updates_live_generation_and_preserves_next_token_alias():
    from types import SimpleNamespace
    from vllm_gaudi.models.deepseek_v41_program import PreparedStage
    stage = SimpleNamespace(sample_greedy=lambda value: value)
    record = torch.tensor([7, 6, 0, -1], dtype=torch.int32)
    next_input = record[3:4]
    for generation, token in ((8, 10), (9, 64645), (10, 0)):
        result = PreparedStage.sample_greedy_commit(stage, torch.tensor([[token]]), record)
        assert result.data_ptr() == record.data_ptr()
        assert record.tolist() == [generation, 1, 1, token] and next_input.tolist() == [token]


def test_sharded_greedy_selection_matches_full_logits_including_ties_and_nan():
    from vllm_gaudi.ops.deepseek_v41_sampling import local_greedy_candidate, select_greedy_candidate
    left = torch.tensor([[1., 5., 5.], [-float("inf")] * 3, [1., float("nan"), 3.], [1., 2., 3.]])
    right = torch.tensor([[5., 5., 1.], [-float("inf")] * 3, [float("nan"), 8., 7.], [4., 6., 5.]])
    pairs = torch.cat([local_greedy_candidate(left, 0), local_greedy_candidate(right, 1)], -1)
    assert torch.equal(select_greedy_candidate(pairs), torch.cat([left, right], -1).argmax(-1, keepdim=True))


def test_c1_replay_rejects_speculative_shapes_when_dspark_is_disabled():
    from vllm_gaudi.ops.deepseek_v41_replay import StageReplay
    program = torch.nn.Module()
    program.dspark = False
    replay = StageReplay(program)
    with pytest.raises(ValueError, match="replay shape"):
        replay(None, None, None, torch.zeros(6, dtype=torch.int64), ())


def test_all_prefill_tails_across_five_groups_use_bounded_compile_cache(monkeypatch):
    """Reproduce the real first-request cache overflow without loading weights."""
    # Dynamo's stream tracker must not acquire an accelerator for this CPU
    # backend/cache test merely because the HPU plugin has been imported.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    from types import SimpleNamespace
    from vllm_gaudi.models.deepseek_v41_program import CompiledStage

    class Layer(torch.nn.Module):

        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, residual, pre, positions, image_mask, rows, *, fp8_decode=False):
            assert not fp8_decode
            return residual + self.layer + positions[:, None, None], pre, None

    stage = SimpleNamespace(layers=torch.nn.ModuleList(Layer(i) for i in range(20)),
                            pp_rank=0,
                            config={"text_config": {
                                "rms_norm_eps": 1e-20
                            }})
    compiled_graphs = []

    def backend(graph, inputs):
        compiled_graphs.append(graph)
        return graph.forward

    compile_fn = torch.compile
    monkeypatch.setattr(
        torch, "compile",
        lambda fn, **kwargs: compile_fn(fn, backend=backend, fullgraph=kwargs["fullgraph"], dynamic=kwargs["dynamic"]))
    previous = (torch._dynamo.config.cache_size_limit, torch._dynamo.config.accumulated_cache_size_limit)
    torch._dynamo.reset()
    try:
        run = CompiledStage(stage)
        for counts in ((6, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 6)):
            before = len(compiled_graphs)
            for count in counts:
                hidden = torch.zeros(count, 4, 2)
                pre = torch.zeros(count, 4)
                positions = torch.arange(count)
                ids = torch.zeros(count, dtype=torch.int64)
                output, _, aux = run(hidden, pre, positions, ids, (hidden, hidden))
                assert torch.equal(output, (190 + 20 * positions[:, None, None]).expand_as(hidden))
                assert aux is None
            if before:
                assert len(compiled_graphs) == before
        assert len(compiled_graphs) == 30
        assert previous == (torch._dynamo.config.cache_size_limit, torch._dynamo.config.accumulated_cache_size_limit)
    finally:
        torch._dynamo.reset()


def test_long_prefill_can_reduce_compiled_layer_lifetime(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.models import deepseek_v41_program as program

    stage = SimpleNamespace(layers=torch.nn.ModuleList(torch.nn.Identity() for _ in range(20)),
                            pp_rank=0,
                            config={"text_config": {
                                "rms_norm_eps": 1e-20
                            }})
    monkeypatch.setattr(program, "_compile_group", lambda group, **_: group)
    run = program.CompiledStage(stage, group_size=2)
    assert len(run.groups) == 10
    assert all(len(group.layers) == 2 for group in run.groups)
    with pytest.raises(ValueError, match="group size"):
        program.CompiledStage(stage, group_size=3)


def test_decoded_kv_state_capture_clear_and_block_rebinding():
    from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors
    program = torch.nn.Module()
    program.pp_rank, program.generation, program.replay_owner = 0, 0, None
    program.shared = torch.nn.Module()
    program.shared.register_buffer("decoded_swa", torch.zeros(1024, 512, dtype=torch.bfloat16))
    program.shared.register_buffer("decoded_main", torch.zeros(768, 512, dtype=torch.bfloat16))
    program.reader = program.shared
    state = StageStateBlocks(program)
    state.allocate(2, "cpu")
    assert len(state.bindings) == 2
    assert len(stage_state_tensors(program)) == 2
    captured = tuple(x.clone() for x in stage_state_tensors(program))
    program.shared.decoded_swa.fill_(3.)
    program.shared.decoded_main.fill_(5.)
    for destination, saved in zip(stage_state_tensors(program), captured, strict=True):
        destination.copy_(saved)
    assert not program.reader.decoded_swa.any() and not program.reader.decoded_main.any()
    program.shared.decoded_main[0, 0] = 17
    state.bind(1)
    assert not program.reader.decoded_main.any()
    state.bind(0)
    assert program.reader.decoded_main[0, 0] == 17
    state.clear()
    assert not program.reader.decoded_main.any()
    assert state.allocated_bytes == 2 * (1024 + 768) * 512 * 2
