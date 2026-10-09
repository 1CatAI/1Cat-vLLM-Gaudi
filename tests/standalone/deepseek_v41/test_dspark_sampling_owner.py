# SPDX-License-Identifier: Apache-2.0
"""Sampling-stage DSpark ownership and the shared C6 prefix contract."""
from types import SimpleNamespace
import time
import json

import pytest
import torch

from vllm_gaudi.models import deepseek_v41_program as program_module
from vllm_gaudi.ops.deepseek_v41_verify import (
    RECORD_SIZE,
    VerifyRing,
    encode_record_wire,
    pack_record,
    verify_control_from_target,
    vocab_parallel_argmax,
)
from vllm_gaudi.ops import tp2_prepared_plan
from vllm_gaudi.v1.worker import deepseek_v41_runner as runner_module


@pytest.mark.parametrize("dspark", (False, True))
@pytest.mark.parametrize("roundtrip", (False, True))
def test_target_precision_configuration_is_shared_and_keeps_roundtrip_guard(monkeypatch, tmp_path, dspark,
                                                                          roundtrip):
    config = {"text_config": {"num_experts_per_tok": 6}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    woa = tmp_path / "woa.json"
    dense = tmp_path / "dense.json"
    woa.write_text(json.dumps({"version": 1, "layers": [20]}))
    dense.write_text(json.dumps({"version": 1, "wq_b": [20], "wo_b": [20]}))
    values = dict(RUNTIME_INDEXER=False, SHARED_GATE_UP=False, ENGRAM_FP8=True, ATTN_DENSE_FP8=True,
                  WO_A_FP8=True, WOA_OUTPUT_ROUNDTRIP=True, QUANT_ROUNDTRIP=roundtrip,
                  ATTN_DENSE_FP8_CONFIG=str(dense), WO_A_FP8_CONFIG=str(woa),
                  EXPERT_N256=False, EXPERT_N256_FP8=False, EXPERT_FUSED_QUANT=False,
                  EXPERT_FUSED_REDUCE=False, FP8_DECODE=False)
    for name, value in values.items():
        monkeypatch.setattr(program_module.gaudi_envs, f"VLLM_HPU_DSV41_{name}", value)
    from vllm_gaudi.ops import deepseek_v41_prefill_regions
    monkeypatch.setattr(deepseek_v41_prefill_regions, "validate_prefill_region_config", lambda: None)
    shard = SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1, specs={},
                            manifest={"pp_layer_ranges": [[0, 40]],
                                      "normal_scales": {f"layers.{i}.ffn.experts": [True] * 4 for i in range(40)}})
    monkeypatch.setattr(program_module, "PreparedV41Shard", lambda *args: shard)
    monkeypatch.setattr(program_module, "_weight_tree", lambda specs: SimpleNamespace(
        layers=SimpleNamespace(get_submodule=lambda key: None)))
    monkeypatch.setattr(program_module, "CSA2SharedState", lambda *args: SimpleNamespace())
    monkeypatch.setattr(program_module, "PreparedDecoderLayer", lambda *args, **kwargs: torch.nn.Module())
    monkeypatch.setattr(program_module, "PreparedDraft", lambda *args: torch.nn.Module())
    monkeypatch.setattr(program_module, "mxfp4_bf16_lut", lambda device: torch.empty(0))

    def make_stage():
        return program_module.PreparedStage(tmp_path, 0, 0, lambda x: x, lambda x, dim: x, "cpu",
                                            tensor_parallel_size=4, pipeline_parallel_size=1, dspark=dspark)

    if not roundtrip:
        with pytest.raises(ValueError, match="requires quantized execution"):
            make_stage()
        return
    stage = make_stage()
    assert stage.engram_fp8 and stage.woa_output_roundtrip
    assert stage.woa_config["layers"] == stage.dense_config["wo_b"] == [20]
    assert (stage.draft is not None) is dspark


@pytest.mark.parametrize("tp_size", (2, 4))
@pytest.mark.parametrize("pretranspose", (False, True))
def test_prepared_bf16_output_layout_owns_its_transpose(monkeypatch, tp_size, pretranspose):
    from vllm_gaudi.models import deepseek_v41_program as program
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    # Checkpoint wo_a is [local groups * 1024, heads per group * 512].
    # TP4 has two groups, while the old loader transpose assumes four.
    groups, heads = 8 // tp_size, 64 // tp_size
    inner = heads // groups * 512
    generator = torch.Generator().manual_seed(42)
    original = torch.randn(groups * 1024, inner, generator=generator).bfloat16()
    spec = {"layers.0.attn.wo_a.weight": {"dtype": "F8_E4M3", "shape": tuple(original.shape)}}
    shard = SimpleNamespace(dense=lambda *args: original.clone(), check_identity=lambda: None)
    tree = program._weight_tree(spec)
    monkeypatch.setattr(program.gaudi_envs, "VLLM_HPU_DSV41_PRETRANSPOSE_ATTN", pretranspose)
    program.load_weight_tree(shard, tree, "cpu", spec)
    owner = SimpleNamespace(weights=tree.layers.get_submodule("0").attn,
                            woa_fp8=False,
                            prepared_output=True,
                            output_gemm_layout=False,
                            heads=heads,
                            groups=groups)
    PagedCSA2Attention.prepare_output_weight(owner)
    value = torch.randn(6, groups, inner, generator=generator).bfloat16()
    expected = torch.einsum("tgd,grd->tgr", value, original.reshape(groups, 1024, inner)).flatten(1)
    actual = PagedCSA2Attention.project_output(owner, value)
    expected_weight = original.reshape(groups, 1024, inner).transpose(1, 2).contiguous()
    if pretranspose:
        assert not torch.equal(owner.weights.wo_a.weight, expected_weight)
        assert not torch.allclose(actual, expected, rtol=.02, atol=.01)
    else:
        torch.testing.assert_close(owner.weights.wo_a.weight, expected_weight, rtol=0, atol=0)
        # CPU BF16 einsum may choose a different reduction for strided and
        # contiguous operands. The weight-layout assertion remains exact;
        # the real HPU projection check separately requires exact outputs.
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.001)


def test_draft_context_shapes_have_independent_compile_budgets_and_reuse_plans(monkeypatch):
    original_compile = torch.compile
    code_objects = []

    def compile_cpu(entry, **options):
        code_objects.append(entry.__func__.__code__)
        options["backend"] = "eager"
        return original_compile(entry, **options)

    class Draft:

        def insert_context(self, aux, positions, valid_count=None):
            return aux + positions.unsqueeze(-1)

    monkeypatch.setattr(torch, "compile", compile_cpu)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    plans = runner_module.DraftContextPlans(Draft().insert_context)
    with torch._dynamo.config.patch(recompile_limit=2):
        for count in (8192, 4096, 2048, 1024, 512, 256, 128, 1, 2, 3, 4, 5, 6):
            values = torch.randn(count, 8)
            positions = torch.arange(count)
            for _ in range(2):
                assert torch.equal(plans(values, positions), values + positions.unsqueeze(-1))
        assert torch._dynamo.config.recompile_limit == 2
    assert len(plans.plans) == len(code_objects) == 13
    assert len({id(code) for code in code_objects}) == 13


@pytest.mark.parametrize("dspark", (False, True))
def test_memory_profile_covers_all_prompt_shapes_and_separates_dspark_warm_execution(monkeypatch, dspark):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    runner.use_dspark = dspark
    runner.state = runner_module.PagedStageState.__new__(runner_module.PagedStageState)
    runner.model = SimpleNamespace(pp_rank=0, tp_rank=0)
    runner.model_config = SimpleNamespace(max_model_len=65536)
    runner.prefill_capacity = 8192
    calls, resets = [], []
    runner._dummy_run = lambda tokens, *, start_position=0: calls.append((tokens, start_position))
    monkeypatch.setattr(runner_module,
                        "prefill_compute_buckets",
                        lambda capacity=8192: (8192, 4096, 2048, 1024, 512, 256, 128))
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(torch.hpu, "reset_peak_memory_stats", lambda: resets.append(len(calls)))
    monkeypatch.setattr(torch.hpu, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.hpu, "max_memory_allocated", lambda: 140)
    runner.profile_run()
    one_pass = [(tokens, position) for position in (0, runner_module.INDEX_MME_HOT_TOKENS)
                for tokens in (8192, 4096, 2048, 1024, 512, 256, 128)] + [(6 if dspark else 1, 0)]
    assert calls == one_pass * (2 if dspark else 1)
    assert resets == ([15] if dspark else [])
    if dspark:
        assert [row["phase"]
                for row in runner.profile_phase_memory] == ["compile_warmup"] * 15 + ["warmed_execution"] * 15
        assert all(row["peak_bytes"] == 140 and row["resident_bytes"] == 100 for row in runner.profile_phase_memory)
    else:
        assert len(runner.profile_memory_steps) == 16


@pytest.mark.parametrize("searches", (None, [512, 1024, 32768, 65536]))
def test_fixed_five_drafts_warm_all_active_target_buckets_and_all_commit_prefixes(monkeypatch, searches):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    ready, targets, commits = [], [], []
    program = SimpleNamespace(
        length=65536,
        runtime_indexer=False,
        replay_owner=SimpleNamespace(require_ready=lambda count, *, search: ready.append((count, search))))
    runner.model = SimpleNamespace(native=True, program=program, last_aux=torch.zeros(6, 3))
    runner.vllm_config = SimpleNamespace(
        additional_config={} if searches is None else {"dsv41_native_warmup_searches": searches})
    runner.positions = torch.arange(8192)
    runner.state = runner_module.PagedStageState.__new__(runner_module.PagedStageState)
    runner.state.clear = lambda: None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True, barrier=lambda: None))
    runner.request_batches = None
    runner.graphed_buckets = set()
    runner.use_dspark = True
    runner._dummy_run = lambda count, **kwargs: targets.append(count)
    runner._insert = lambda aux, positions: commits.append((len(aux), len(positions)))
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DSPARK", True)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_VERIFY_TIMING", False)
    runner.warmup_model()
    buckets = searches or (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    assert set(ready) == {(count, search) for count in (1, 6) for search in buckets}
    assert set(targets) == runner.graphed_buckets == {1, 6}
    assert targets[-1] == 6 and commits == [(2, 2), (3, 3), (4, 4), (5, 5)]


@pytest.mark.parametrize("searches", ([], [3], [65536, 65536], [True], "32768"))
def test_native_warmup_search_selection_rejects_invalid_or_unsupported_buckets(searches):
    with pytest.raises(ValueError, match="distinct supported search buckets"):
        runner_module.select_native_warmup_geometries(((0, 512), (512, 1024)),
                                                      {"dsv41_native_warmup_searches": searches})


@pytest.mark.parametrize("tp_rank", (0, 3))
def test_single_stage_transaction_commits_and_releases_every_sampling_owner(monkeypatch, tp_rank):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True, ranks=[tp_rank])
    monkeypatch.setattr(runner_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event",
                        lambda: SimpleNamespace(synchronize=lambda: None, record=lambda stream: None))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)
    for key in ("MHC_SCHEDULE", "DIRECT_PP_WIRE", "DEVICE_VERIFY", "PP_DIRECT_EXCHANGE", "INLINE_PP_COMMIT"):
        monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_" + key, False)
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    runner.pp = runner_module.PPBuffers("cpu", capacity=8192, dspark=True)
    runner.verify_ring = VerifyRing("cpu", last_rank=True)
    runner.verify_timing = None
    runner.verify_hidden = torch.zeros(6, 1)
    runner.positions = torch.arange(8192, dtype=torch.int32)
    runner.position_views = {6: runner.positions[:6]}
    runner.verify_positions = torch.zeros(6, dtype=torch.int32)
    runner.verify_aux = torch.zeros(6, 3)
    runner.verify_control_host = torch.zeros(runner_module.VERIFY_CONTROL_SIZE, dtype=torch.int64)
    runner.verify_control = torch.zeros_like(runner.verify_control_host)
    runner.verify_metadata = runner.verify_control[:7]
    runner.verify_proposed = runner.verify_control[7:12]
    runner.model_config = SimpleNamespace(max_model_len=65536)
    committed_counts = []
    runner.model = SimpleNamespace(pp_rank=0,
                                   tp_rank=tp_rank,
                                   last_aux=torch.zeros(6, 3),
                                   complete_step_device=committed_counts.append)
    runner.device = "cpu"
    runner.audit = {"accepted_drafts": 0, "rejected_drafts": 0}
    runner.verify_records = {}
    runner.round_timing_enabled = True
    runner.round_records = []
    request = SimpleNamespace(req_id="whole-round", output=[], sampling_params=SimpleNamespace(max_tokens=128))

    def prefix(hidden, proposed, metadata, states, positions):
        target, output, committed, count, anchor, enabled, status = verify_control_from_target(
            hidden[:, 0].long(), proposed, metadata)
        record = pack_record(metadata, committed, count, output, torch.full_like(proposed, -1),
                             torch.zeros_like(enabled), status)
        return target, output, committed, count, anchor, enabled, status, record, encode_record_wire(record)

    def draft(metadata, positions, output, committed, count, anchor, enabled, status):
        record = pack_record(metadata, committed, count, output, torch.arange(20, 25), enabled, status)
        return record, encode_record_wire(record), torch.ones(5)

    runner.verify_prefix, runner.draft_from_prefix = prefix, draft
    for accepted in range(6):
        target = torch.arange(10, 16).reshape(6, 1).float()
        proposals = target[:5, 0].long().tolist()
        if accepted < 5:
            proposals[accepted] = 0
        runner.pending = object()
        runner.round_context = dict(request_id=request.req_id,
                                    generation=runner.pp.generation + 1,
                                    target_count=6,
                                    start_ns=time.perf_counter_ns())
        result = runner._finish_request_device(request, 0, 6, 6, proposals, target)
        output = result.get_output()
        assert output.sampled_token_ids == [list(range(10, 11 + accepted))]
        assert output.execution_rounds == [(request.req_id, accepted + 1, 6)]
        assert runner.round_records[-1]["ring_released"]
        assert runner.take_draft_token_ids().draft_token_ids == [[20, 21, 22, 23, 24]]
        assert runner.pending is None
        with pytest.raises(RuntimeError, match="only be consumed once"):
            result.get_output()
    assert committed_counts == list(range(1, 7))
    assert runner.audit == {"accepted_drafts": 15, "rejected_drafts": 15}
    assert runner.pp.commits == runner.verify_ring.generation == 6
    runner.verify_ring.close()


@pytest.mark.parametrize("entry", ("warmup", "serving"))
def test_c6_verify_consumes_six_positions_from_the_full_prompt_buffer(monkeypatch, entry):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    hidden = torch.zeros(6, 4)
    runner.positions = torch.arange(8192, dtype=torch.int32)
    runner.position_views = {6: runner.positions[:6]}
    runner.verify_hidden = torch.zeros_like(hidden)
    runner.verify_aux = torch.zeros(6, 12)
    runner.verify_positions = torch.zeros(6, dtype=torch.int32)
    runner.verify_offsets = torch.arange(6, dtype=torch.int32)
    runner.verify_control_host = torch.zeros(runner_module.VERIFY_CONTROL_SIZE, dtype=torch.int64)
    runner.verify_control = torch.zeros_like(runner.verify_control_host)
    runner.verify_metadata_host = runner.verify_control_host[:7]
    runner.verify_metadata = runner.verify_control[:7]
    runner.verify_proposed = runner.verify_control[7:12]
    runner.model = SimpleNamespace(pp_rank=0, tensor_parallel_size=4, last_aux=torch.zeros(6, 12))
    runner.model_config = SimpleNamespace(max_model_len=65536)
    runner.verify_timing = None
    runner.state = SimpleNamespace(clear=lambda: None)
    runner.use_dspark = True
    runner.audit = {"target_steps": 0}
    runner._forward = lambda *args, **kwargs: hidden
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True), generation=0, wait_record=lambda value: None)
    ticket = SimpleNamespace(generation=1, record=object())
    runner.verify_ring = SimpleNamespace(acquire=lambda *args, **kwargs: ticket)
    runner.draft_from_prefix = object()
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DEVICE_VERIFY", True)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DSPARK", True)
    monkeypatch.setattr(tp2_prepared_plan, "prepared_group_stats", lambda: {
        "native_graphs": 0,
        "native_replays": 0,
        "native_entry_replays": 0
    })

    class VerifiedContract(Exception):
        pass

    def prefix(target, proposed, metadata, states, positions):
        assert target is hidden and states is runner.model.last_aux
        assert positions is runner.position_views[6]
        assert target.shape[0] == states.shape[0] == positions.numel() == 6
        assert positions.tolist() == list(range(6))
        raise VerifiedContract

    runner.verify_prefix = prefix
    with pytest.raises(VerifiedContract):
        if entry == "warmup":
            runner._dummy_run(6, native=True)
        else:
            request = SimpleNamespace(output=[], sampling_params=SimpleNamespace(max_tokens=64))
            runner._finish_request_device(request, 0, 6, 6, [1, 2, 3, 4, 5], hidden)


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("tokens", (1, 6, 128, 256, 8192))
@pytest.mark.parametrize("native", (False, True))
def test_dspark_warmup_never_sends_prefill_to_a_six_row_replay(monkeypatch, tp, tokens, native):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    seen = []
    runner.model = SimpleNamespace(pp_rank=0, tensor_parallel_size=tp, complete_step=lambda count: None)
    runner.state = SimpleNamespace(clear=lambda: None)
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=False, barrier=lambda: None),
                                drain=lambda: None,
                                complete_packet=lambda: None)
    runner.audit = {"target_steps": 0}
    runner.use_dspark = True
    runner._forward = lambda *args, **kwargs: seen.append(kwargs["decode"])
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(tp2_prepared_plan, "prepared_group_stats", lambda: {
        "native_graphs": 0,
        "native_replays": 0,
        "native_entry_replays": 0
    })
    runner._dummy_run(tokens, native=native)
    assert seen == [(native or tp == 4) and tokens <= 6]


@pytest.mark.parametrize("tp", (2, 4))
def test_large_prompt_commits_every_input_row_and_samples_only_its_tail(monkeypatch, tp):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    request = SimpleNamespace(req_id="prefill", output=[], sampling_params=SimpleNamespace(max_tokens=64))
    logits = torch.zeros(1, 64)
    logits[0, 45] = 10
    runner.pending = (request, 0, 8192, 8192, [], True, logits)
    runner.use_dspark = True
    committed_rows, inserted_rows = [], []
    runner.model = SimpleNamespace(pp_rank=0 if tp == 4 else 1,
                                   last_aux=torch.zeros(8192, 12),
                                   complete_step=committed_rows.append)
    runner.positions = torch.arange(8192)
    runner.model_config = SimpleNamespace(max_model_len=1048576)
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_first_rank=tp == 4, is_last_rank=True),
                                finish=lambda committed, output, draft: (committed, output, draft))
    runner.verify_prefix = object()
    runner._insert = lambda aux, positions: inserted_rows.append((len(aux), len(positions)))
    runner._propose = lambda anchor, position: [46, 47, 48, 49, 50]
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DEVICE_VERIFY", True)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DSPARK", True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Large prompt reached the C6 verify entry")

    runner._finish_request_device = forbidden
    result = runner._finish_request()
    assert result.sampled_token_ids == [[45]]
    assert committed_rows == [8192] and inserted_rows == [(8192, 8192)]
    assert runner.pending is None


def test_large_prompt_projects_only_the_final_hidden_row(monkeypatch):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    tokens = [7] * 8192
    request = SimpleNamespace(req_id="prefill",
                              output=[],
                              prompt=tokens,
                              num_computed_tokens=0,
                              decode_start=8192,
                              token_slice=lambda start, end: tokens[start:end],
                              mm_features=[],
                              sampling_params=SimpleNamespace(prompt_logprobs=None))
    runner.requests = {request.req_id: request}
    runner.round_timing_enabled = False
    runner.use_dspark = True
    runner.prefill_capacity = 8192
    runner.model_config = SimpleNamespace(max_model_len=1048576)
    projected = []
    hidden = torch.arange(8192 * 4).reshape(8192, 4).float()

    def project(value):
        projected.append(value.clone())
        return torch.zeros(1, 64)

    runner.model = SimpleNamespace(program=None, pp_rank=0, tp_rank=0, compute_logits=project)
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True), generation=0)
    runner.verify_timing = None
    runner._bind_request = lambda request: None
    runner._forward = lambda *args, **kwargs: hidden
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DEVICE_VERIFY", True)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", 8192)
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={}, num_scheduled_tokens={request.req_id: 8192})
    runner._execute_request(scheduled, request.req_id, 8192)
    assert len(projected) == 1 and torch.equal(projected[0], hidden[-1:])
    assert runner.pending[-1].shape == (1, 64)


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("bf16_head", (False, True))
def test_draft_layers_use_sampling_groups_actual_tp(monkeypatch, tp, bf16_head):
    config = dict(num_hidden_layers=40, dspark_num_experts_per_tok=3, rms_norm_eps=1e-6, dspark_noise_token_id=128799)
    weights = torch.nn.Module()
    for index in range(3):
        weights.add_module(str(index), torch.nn.Identity())
    stage = SimpleNamespace(
        weights=SimpleNamespace(mtp=weights, head=object()),
        tp_rank=tp - 1,
        tensor_parallel_size=tp,
        bf16_head=bf16_head,
        reduce=lambda value: value,
        all_gather=lambda value, dim: value,
        config={"text_config": config},
        shared=object(),
        shard=SimpleNamespace(
            manifest={"normal_scales": {
                f"mtp.{index}.ffn.experts": [True] * tp
                for index in range(3)
            }}))
    calls, rotary_bindings = [], []

    def decoder(*args, **kwargs):
        calls.append((args[2], args[1]["num_experts_per_tok"], kwargs["tensor_parallel_size"]))
        layer = torch.nn.Identity()
        layer.attention = SimpleNamespace(length=65536, set_search_length=rotary_bindings.append)
        return layer

    monkeypatch.setattr(program_module, "PreparedDecoderLayer", decoder)
    draft = program_module.PreparedDraft(stage, torch.empty(1), "cpu")
    assert calls == [(40, 3, tp), (41, 3, tp), (42, 3, tp)]
    assert rotary_bindings == [65536, 65536, 65536]
    assert draft.offsets.tolist() == [0, 1, 2, 3, 4]
    assert draft.bf16_head is bf16_head


@pytest.mark.parametrize("bf16", (False, True))
def test_speculative_verification_uses_the_same_head_as_c1(monkeypatch, bf16):
    generator = torch.Generator().manual_seed(6416)
    hidden = torch.randint(-2, 3, (6, 5120), generator=generator).bfloat16()
    weight = torch.randint(-2, 3, (17, 5120), generator=generator).to(
        torch.bfloat16 if bf16 else torch.float32)
    calls = []

    def bf16_projection(value, matrix):
        assert value.dtype == matrix.dtype == torch.bfloat16
        calls.append(tuple(value.shape))
        return torch.nn.functional.linear(value.float(), matrix.float())

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_bf16_linear_f32_gaudi2",
                        bf16_projection, raising=False)
    stage = SimpleNamespace(bf16_head=bf16, weights=SimpleNamespace(head=SimpleNamespace(weight=weight)))
    reference = program_module.PreparedStage._head_projection(stage, hidden).argmax(-1)
    draft = program_module.PreparedDraft.__new__(program_module.PreparedDraft)
    torch.nn.Module.__init__(draft)
    draft.bf16_head, draft.output_head = bf16, SimpleNamespace(weight=weight)
    draft._global_argmax = lambda logits: logits.argmax(-1)
    commits = []
    draft.insert_context = lambda states, positions, count: commits.append(int(count))
    metadata = torch.tensor([7, 6, 5, 100, 16384, 32768, 1])
    result = draft.verify_prefix(hidden, reference[:5], metadata,
                                 torch.zeros(6, 3), torch.arange(16384, 16390))
    assert torch.equal(result[0], reference)
    assert torch.equal(result[1], reference)
    assert commits == [6]
    assert calls == ([(6, 5120), (6, 5120)] if bf16 else [])


@pytest.mark.parametrize("bf16", (False, True))
def test_markov_checkpoint_values_survive_projection_precision_selection(monkeypatch, bf16):
    name = "mtp.2.markov_head.head.weight"
    original = torch.arange(17 * 256, dtype=torch.float32).reshape(17, 256).bfloat16() / 1024
    spec = {name: {"dtype": "BF16", "shape": tuple(original.shape)}}
    shard = SimpleNamespace(tensor=lambda *args: original.clone(), check_identity=lambda: None)
    tree = program_module._weight_tree(spec)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_BF16_LM_HEAD", bf16)
    program_module.load_weight_tree(shard, tree, "cpu", spec)
    loaded = tree.mtp.get_submodule("2").markov_head.head.weight
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded.float(), original.float())


@pytest.mark.parametrize("bf16", (False, True))
def test_markov_five_steps_keep_fp32_scores_and_predecessor_sampling(monkeypatch, bf16):
    generator = torch.Generator().manual_seed(6405)
    embedding = torch.nn.Embedding(17, 256, dtype=torch.bfloat16)
    embedding.weight = torch.nn.Parameter(torch.randint(-2, 3, (17, 256), generator=generator).bfloat16(),
                                         requires_grad=False)
    weight = torch.randint(-2, 3, (17, 256), generator=generator).float()
    confidence = torch.nn.Linear(5376, 1, bias=False)
    last = SimpleNamespace(markov_head=SimpleNamespace(
        embed=embedding, head=SimpleNamespace(weight=weight)),
        confidence_head=SimpleNamespace(proj=confidence))
    draft = program_module.PreparedDraft.__new__(program_module.PreparedDraft)
    torch.nn.Module.__init__(draft)
    draft.weights = SimpleNamespace(get_submodule=lambda name: last)
    draft.tp_rank, draft.reduce = 0, lambda value: value
    draft.bf16_head = bf16
    draft._global_argmax = lambda value: value.argmax(-1)
    logits, hidden = torch.randn(5, 17, generator=generator), torch.zeros(5, 5120, dtype=torch.bfloat16)
    expected, predecessor = [], torch.tensor([2])
    for row in logits:
        bias = torch.nn.functional.linear(embedding(predecessor).float(), weight)
        predecessor = (row.unsqueeze(0) + bias).argmax(-1)
        expected.append(predecessor)
    calls = []

    def bf16_projection(value, matrix):
        assert value.dtype == matrix.dtype == torch.bfloat16
        calls.append(tuple(value.shape))
        return torch.nn.functional.linear(value.float(), matrix.float())

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_bf16_linear_f32_gaudi2",
                        bf16_projection, raising=False)
    tokens, scores = draft.sample_greedy_local(torch.tensor([2]), hidden, logits)
    assert torch.equal(tokens, torch.cat(expected))
    assert scores.dtype == torch.float32 and torch.isfinite(scores).all()
    assert calls == [], "Markov K256 keeps FP32 independently of the K5120 target head"


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("accepted", range(6))
def test_vocab_sharded_target_and_c6_acceptance_match_greedy(tp, accepted):
    generator = torch.Generator().manual_seed(4406)
    logits = torch.randn(6, 8 * tp, generator=generator)
    # Exercise equal maxima across ranks: greedy keeps the earliest global ID.
    logits[0, 0] = logits[0, 8] = 100
    pairs = []
    for rank in range(tp):
        values, ids = logits[:, rank * 8:(rank + 1) * 8].max(-1)
        pairs.append(torch.stack((values, (ids + rank * 8).float()), -1))
    target = logits.argmax(-1)
    proposed = target[:5].clone()
    if accepted < 5:
        proposed[accepted] = (proposed[accepted] + 1) % logits.shape[1]
    metadata = torch.tensor([1, 6, 5, 128, 16384, 1048576, 1])
    for rank in range(tp):
        actual = vocab_parallel_argmax(logits[:, rank * 8:(rank + 1) * 8], rank,
                                       lambda value, dim: torch.cat(pairs, dim=dim))
        assert torch.equal(actual, target)
        result = verify_control_from_target(actual, proposed, metadata)
        assert result[2].item() == result[3].item() == accepted + 1
        assert result[1][:accepted + 1].tolist() == target[:accepted + 1].tolist()
        assert result[-1].item() == 0


def test_single_sampling_stage_never_sends_pipeline_commit(monkeypatch):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True, ranks=[0])
    monkeypatch.setattr(runner_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(synchronize=lambda: None))
    for key in ("MHC_SCHEDULE", "DIRECT_PP_WIRE", "DEVICE_VERIFY", "PP_DIRECT_EXCHANGE", "INLINE_PP_COMMIT"):
        monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_" + key, False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Single sampling stage has no PP peer")

    monkeypatch.setattr(runner_module.dist, "broadcast", forbidden)
    buffers = runner_module.PPBuffers("cpu", capacity=8192, dspark=True)
    assert buffers.exchange_wire.numel() == buffers.hidden.numel() == 0
    assert buffers.finish(1, [42], [1, 2, 3, 4, 5]) == (1, [42], [1, 2, 3, 4, 5])
    record = torch.zeros(RECORD_SIZE, dtype=torch.int64)
    assert buffers.finish_device(record, 6) is None
    assert buffers.generation == buffers.commits == 2
    assert buffers.sends == buffers.receives == 0


@pytest.mark.parametrize("capped_aux", (False, True))
def test_prefill_draft_context_scatter_only_uses_last_unique_window(monkeypatch, capped_aux):
    calls = []

    class Attention:
        swa = torch.zeros(256, 3)

        def insert_context(self, value, positions, valid_count):
            calls.append((value.clone(), positions.clone(), valid_count))

    layer = SimpleNamespace(attention=Attention())
    owner = SimpleNamespace(
        layers=[layer] * 3,
        weights=SimpleNamespace(
            get_submodule=lambda name: SimpleNamespace(main_proj=object(), main_norm=SimpleNamespace(weight=object()))),
        eps=1e-6)
    monkeypatch.setattr(program_module, "linear", lambda value, weight: value)
    monkeypatch.setattr(program_module, "rms_norm", lambda value, weight, eps: value)
    positions = torch.arange(8192, dtype=torch.int32)
    values = positions[:, None].float().expand(-1, 3)
    program_module.PreparedDraft.insert_context(owner, values[-256:] if capped_aux else values, positions)
    for value, ids, count in calls:
        assert ids.tolist() == list(range(7936, 8192))
        assert ids.remainder(256).unique().numel() == 256
        assert torch.equal(value, values[-256:]) and count is None
    calls.clear()
    valid = torch.tensor([3])
    program_module.PreparedDraft.insert_context(owner, values[:6], positions[:6], valid)
    assert all(ids.numel() == 6 and count is valid for _, ids, count in calls)


@pytest.mark.parametrize("tokens,grouped", ((4096, True), (4096, False), (6, True)))
def test_draft_auxiliary_tail_matches_full_mean_and_preserves_other_row_contracts(tokens, grouped):
    residual = torch.randn(tokens, 4, 12, generator=torch.Generator().manual_seed(37), dtype=torch.bfloat16)
    reference = residual.mean(1)
    result = program_module.draft_context_state(residual, 256, grouped_prefill=grouped)
    torch.testing.assert_close(result, reference[-256:] if grouped else reference, rtol=0, atol=0)
    assert result.shape[0] == (min(tokens, 256) if grouped else tokens)


def test_decoder_auxiliary_rows_use_physical_draft_ring_capacity(monkeypatch):
    # The model's sliding window is smaller than its physical packed ring.
    # The caller must retain the row count expected by PreparedDraft insertion.
    owner = SimpleNamespace(weights=SimpleNamespace(),
                            layer=37,
                            draft=False,
                            collect_target_state=True,
                            moe=SimpleNamespace(tensor_parallel_size=4),
                            attention=SimpleNamespace(window=128, swa=torch.empty(256, 3)))
    captured = []

    class AuxiliaryCaptured(Exception):
        pass

    def capture(residual, capacity, *, grouped_prefill):
        captured.append((capacity, grouped_prefill))
        raise AuxiliaryCaptured

    monkeypatch.setattr(program_module, "draft_context_state", capture)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_PREFILL_GROUPED", True)
    with pytest.raises(AuxiliaryCaptured):
        program_module.PreparedDecoderLayer.forward(owner, torch.zeros(512, 4, 12), None, torch.arange(512),
                                                    torch.zeros(512, dtype=torch.bool))
    assert captured == [(256, True)]
