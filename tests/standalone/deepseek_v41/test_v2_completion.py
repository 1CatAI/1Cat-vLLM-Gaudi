# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle tests; these do not qualify HPU event ordering or latency."""
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, V41ModelRunner
from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import CompletionRecord, V41AsyncOutput, V41V2ModelRunner


class Done:

    def __init__(self):
        self.calls = 0
        self.ready = True

    def synchronize(self):
        self.calls += 1
        if not self.ready:
            raise RuntimeError("not ready")


@pytest.mark.parametrize("batch_enabled", [False, True])
def test_device_engram_abi_is_required_even_with_batch_capacity(monkeypatch, batch_enabled, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_host as host
    monkeypatch.setattr(host, "host_native", lambda: SimpleNamespace(abi_version=1, c1_abi_version=1))
    flags = dict(VLLM_HPU_DSV41_BATCH_DECODE=batch_enabled,
                 VLLM_HPU_DSV41_V2_DEVICE_ENGRAM=True,
                 VLLM_HPU_DSV41_ENGRAM_NATIVE_C1=True,
                 VLLM_HPU_DSV41_ENGRAM_C1_PACKET=True,
                 VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT=True,
                 VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH=True,
                 VLLM_HPU_DSV41_GRAPH_REPLAY=True)
    for key, value in flags.items():
        monkeypatch.setenv(key, "1" if value else "0")
    with pytest.raises(RuntimeError, match="Device Engram requires native host C1 ABI 2"):
        host.EngramHost(tmp_path, 0, "hpu")


def fixture():
    runner = object.__new__(V41V2ModelRunner)
    runner.prefix_checkpoints = None
    runner.requests = {"a": RequestState("a", [1], [], None, ([1], ), output=[10], num_computed_tokens=1)}
    runner.active_request = "a"
    calls = []
    runner.model = SimpleNamespace(complete_step=lambda count: calls.append(("model", count)))
    token = torch.tensor([11], dtype=torch.int32)
    runner.pp = SimpleNamespace(generation=2,
                                commits=0,
                                commit_token=token,
                                complete_packet=lambda: calls.append(("packet", )))
    runner._next_input = None
    runner._input_committed = None
    runner._prefix_started = None
    runner.audit = {}
    runner._completion = CompletionRecord("a", 2, 1, torch.tensor([[11]]), Done(), token)
    return runner, calls


@pytest.mark.parametrize("rounds", [None, [("a", 2, 1)]])
def test_sync_batch_output_does_not_require_engine_diagnostic_field(rounds):
    runner = object.__new__(V41ModelRunner)
    runner.prefix_checkpoints = None
    runner.pending = None
    runner.request_batches = None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True))
    runner.draft_token_ids = None
    runner._update = lambda scheduled: None
    runner._execute_request = lambda scheduled, req_id, count: None
    request_output = ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[11]])
    if rounds is not None:
        request_output.execution_rounds = rounds
    runner._finish_request = lambda: request_output
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1})

    assert runner.execute_model(scheduled) is None
    assert runner.batch_result.sampled_token_ids == [[11]]
    assert getattr(runner.batch_result, "execution_rounds", None) == rounds


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_async_sample_releases_only_the_v2_batch_marker(tp_rank):
    runner, _ = fixture()
    record = runner._completion
    runner.model.tp_rank = tp_rank
    runner.pp.group = SimpleNamespace(is_last_rank=True)
    runner.pending = "batch_ready"
    runner.batch_result = V41AsyncOutput(record)

    result = runner.sample_tokens(None)

    if tp_rank == 0:
        assert isinstance(result, AsyncModelRunnerOutput)
    else:
        assert result.sampled_token_ids == [[11]]
    assert runner.pending is None and runner.batch_result is None
    assert runner._completion is record

    runner._update = lambda scheduled: None
    empty = SimpleNamespace(num_scheduled_tokens={}, finished_req_ids=set())
    assert runner.execute_model(empty) is EMPTY_MODEL_RUNNER_OUTPUT
    assert runner._completion is record


def test_output_thread_does_not_commit_worker_state():
    runner, calls = fixture()
    record = runner._completion
    output = V41AsyncOutput(record)
    assert output.get_output().sampled_token_ids == [[11]]
    assert record.done.calls == 1
    assert not calls and runner.requests["a"].output == [10]

    runner._consume_completion()

    assert calls == [("packet", ), ("model", 1)]
    assert runner.requests["a"].output == [10, 11]
    assert runner.pp.commits == 1 and runner._completion is None
    assert runner._next_input[:2] == ("a", 2)
    runner.pp.commit_token.fill_(99)
    assert output.get_output().sampled_token_ids == [[11]]
    assert record.done.calls == 1


def test_completion_record_memoizes_one_real_d2h_completion():
    done = Done()
    host = torch.tensor([[11]])
    record = CompletionRecord("a", 2, 1, host, done, torch.tensor([11]))
    assert record.token() == record.token() == 11
    host.fill_(99)
    assert record.token() == 11 and done.calls == 1


def test_multi_request_step_uses_synchronous_completion(monkeypatch):
    runner = object.__new__(V41V2ModelRunner)
    runner._v2_async_step = False
    expected = object()
    monkeypatch.setattr(V41ModelRunner, "_sample_single", lambda self: expected)

    assert runner._sample_single() is expected


@pytest.mark.parametrize("row", [[11, 12], [-1], []])
def test_invalid_completion_never_mutates_history(row):
    runner, calls = fixture()
    runner._completion = CompletionRecord("a", 2, 1, torch.tensor([row]), Done(), runner.pp.commit_token)
    with pytest.raises(RuntimeError, match="Invalid"):
        runner._consume_completion()
    assert not calls and runner._completion is not None


@pytest.mark.parametrize("change", ["request", "generation", "prefix", "not_ready"])
def test_stale_owner_and_unfinished_copy_are_rejected(change):
    runner, calls = fixture()
    if change == "request":
        runner.active_request = "b"
    elif change == "generation":
        runner.pp.generation += 1
    elif change == "prefix":
        runner.requests["a"].output.append(12)
    else:
        runner._completion.done.ready = False
    with pytest.raises(RuntimeError):
        runner._consume_completion()
    assert not calls


def test_v2_gate_requires_the_complete_segmented_device_contract(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_config as config_module

    values = {
        "VLLM_HPU_DSV41_V2": True,
        "VLLM_HPU_DSV41_DSPARK": False,
        "VLLM_HPU_DSV41_GRAPH_REPLAY": True,
        "VLLM_HPU_DSV41_DIRECT_TOKEN_IDS": True,
        "VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX": True,
        "VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT": True,
        "VLLM_HPU_DSV41_V2_DEVICE_ENGRAM": True,
        "VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH": True,
        "VLLM_HPU_DSV41_FIXED_POSITIONS": True,
        "VLLM_HPU_DSV41_FUSED_STAGE_IO": False,
        "VLLM_HPU_TP2_NATIVE_JOINT_PLAN": True,
        "VLLM_HPU_DSV41_TP_MHC_OVERLAP": True,
        "VLLM_HPU_DSV41_ENGRAM_NATIVE_C1": True,
        "VLLM_HPU_DSV41_ENGRAM_C1_PACKET": True,
        "VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT": False,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, "1" if value else "0")
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"),
                                                          max_model_len=512),
                             use_v2_model_runner=True,
                             scheduler_config=SimpleNamespace(async_scheduling=True),
                             speculative_config=None)
    with pytest.raises(ValueError, match="Device Engram"):
        config_module.validate_v2(config)
    monkeypatch.setenv("VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT", "1")
    config_module.validate_v2(config)
    config.model_config.max_model_len = 1_048_576
    config_module.validate_v2(config)
    monkeypatch.setenv("VLLM_HPU_TP2_NATIVE_JOINT_PLAN", "0")
    with pytest.raises(ValueError, match="segmented prefix"):
        config_module.validate_v2(config)


@pytest.mark.parametrize("last", [False, True])
def test_sample_submits_device_broadcast_and_copy_without_waiting(monkeypatch, last):
    from vllm_gaudi.distributed import tp2_fused_ar_norm

    runner, calls = fixture()
    done = Done()
    done.ready = False
    runner._completion = None
    runner.pp.generation = 1
    runner.pp.drain = lambda: calls.append("drain")
    runner._relay_token = torch.tensor([[11]], dtype=torch.int32)
    runner.pp.group = SimpleNamespace(is_last_rank=last, broadcast=lambda value, src: calls.append(("broadcast", src)))

    def copy(value):
        calls.append("copy")
        return value.clone(), done

    monkeypatch.setattr(tp2_fused_ar_norm, "_resolve_runtime", lambda:
                        (SimpleNamespace(copy_sampled_tokens_to_host=copy), None, None))
    selected = torch.tensor([[11]], dtype=torch.int32) if last else None
    runner.pending = (runner.requests["a"], 1, 1, 1, [], True, selected)

    output = runner._sample_single()

    assert calls == ["drain", ("broadcast", 1), "copy"]
    assert done.calls == 0 and runner.pending is None
    assert runner.requests["a"].output == [10]
    assert isinstance(output, V41AsyncOutput) if last else output is None
    done.ready = True
    runner._consume_completion()
    assert runner.requests["a"].output == [10, 11]
    assert runner._next_input[2].dtype == torch.int32


def test_early_input_commit_keeps_token_ownership_until_ready(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT", "1")
    runner, calls = fixture()
    record = runner._completion
    record.done.ready = False
    for _ in range(2):
        with pytest.raises(RuntimeError, match="not ready"):
            runner._consume_completion()
        assert calls == [("model", 1)]
        assert runner.requests["a"].output == [10]
        assert runner.pp.commits == 0 and runner._completion is record
    record.done.ready = True
    runner._consume_completion()
    assert calls == [("model", 1), ("packet", )]
    assert runner.requests["a"].output == [10, 11]
    assert runner.audit["v2_early_input_commits"] == 1


def test_device_engram_precedes_prefix_and_host_token_wait(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_DEVICE_ENGRAM", "1")
    runner, calls = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    runner.position_bank = SimpleNamespace(view=lambda start, count: torch.tensor([start], dtype=torch.int32))

    class OrderedDone:

        def synchronize(self):
            calls.append(("token_wait", ))

    runner.model = SimpleNamespace(
        complete_step=lambda count: calls.append(("model", count)),
        prepare_device_engram=lambda request, token: calls.append(("device_engram", request, token)),
        begin_decode_prefix=lambda token, position: calls.append(("prefix", token, position.tolist())),
        decode_prefix_ready=lambda search: True,
        program=SimpleNamespace(length=1 << 20, search_length=512),
    )
    runner._completion = CompletionRecord("a", 2, 1, torch.tensor([[11]]), OrderedDone(), runner.pp.commit_token)
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(), scheduled_spec_decode_tokens={})

    runner._consume_completion(scheduled)

    assert [entry[0] for entry in calls] == ["model", "device_engram", "prefix", "token_wait", "packet"]
    assert runner.audit["v2_device_engram_starts"] == 1
    assert runner._prefix_started == ("a", 2, 1)


def test_new_search_bucket_captures_complete_plan_before_segmenting(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_DEVICE_ENGRAM", "1")
    runner, calls = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    runner.position_bank = SimpleNamespace(view=lambda start, count: torch.tensor([start], dtype=torch.int32))
    runner.requests["a"].prompt = [1] * 1024
    runner.requests["a"].output = [10]
    runner._completion = CompletionRecord("a", 2, 1024, torch.tensor([[11]]), Done(), runner.pp.commit_token)
    ready = []
    runner.model = SimpleNamespace(
        complete_step=lambda count: calls.append(("model", count)),
        prepare_device_engram=lambda request, token: calls.append(("device_engram", request, token)),
        begin_decode_prefix=lambda token, position: calls.append(("prefix", token, position.tolist())),
        decode_prefix_ready=lambda search: ready.append(search) or False,
        program=SimpleNamespace(length=1 << 20, search_length=1024),
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(), scheduled_spec_decode_tokens={})

    runner._consume_completion(scheduled)

    assert ready == [2048]
    assert calls == [("model", 1), ("packet", )]
    assert runner._prefix_started is None
    assert runner.audit["v2_prefix_bucket_captures"] == 1


@pytest.mark.parametrize("search", [512, 1024, 2048, 524288])
def test_warmed_next_bucket_cannot_start_prefix_with_previous_bindings(monkeypatch, search):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner, _ = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    queries = []
    runner.model = SimpleNamespace(
        decode_prefix_ready=lambda value: queries.append(value) or True,
        program=SimpleNamespace(length=1 << 20, search_length=search),
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(), scheduled_spec_decode_tokens={})
    record = SimpleNamespace(request_id="a", start=search - 1)
    assert not runner._prefix_authorized(record, scheduled)
    assert queries == [search * 2]
    assert runner.audit["v2_prefix_bucket_transitions"] == 1
    assert "v2_prefix_bucket_captures" not in runner.audit

    # The complete native entry binds the new attention geometry; early
    # continuation can then resume on the following token without recapture.
    runner.model.program.search_length = search * 2
    record.start += 1
    assert runner._prefix_authorized(record, scheduled)


def test_bounded_stage_does_not_need_a_paged_search_binding(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner, _ = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    runner.model = SimpleNamespace(decode_prefix_ready=lambda search: search == 512,
                                   program=SimpleNamespace(length=512))
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(), scheduled_spec_decode_tokens={})
    assert runner._prefix_authorized(SimpleNamespace(request_id="a", start=10), scheduled)


@pytest.mark.parametrize("position", [510, 511, 512, 1023, 1024, 2558, 2559, 2560, 524287, 1048574])
def test_runtime_indexer_reuses_bound_prefix_across_history_boundaries(monkeypatch, position):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner, _ = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    from vllm_gaudi.v1.worker.deepseek_v41_runner import runtime_search_length
    queries = []
    bound = runtime_search_length(position + 1, 1, 1 << 20)
    runner.model = SimpleNamespace(decode_prefix_ready=lambda search: queries.append(search) or True,
                                   program=SimpleNamespace(length=1 << 20, search_length=bound, runtime_indexer=True))
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(), scheduled_spec_decode_tokens={})
    assert runner._prefix_authorized(SimpleNamespace(request_id="a", start=position), scheduled)
    assert queries == [bound]
    # A prompt path can temporarily bind another geometry. Never run an old
    # prefix merely because the persistent C1 recipe already exists.
    runner.model.program.search_length = 8192
    assert not runner._prefix_authorized(SimpleNamespace(request_id="a", start=position), scheduled)


def test_device_engram_skips_only_matching_host_layer1(monkeypatch):
    from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM

    monkeypatch.setenv("VLLM_HPU_DSV41_V2_DEVICE_ENGRAM", "1")

    class Host:

        def __init__(self, pending=None):
            self.pending = None
            self.device_pending = pending
            self.calls = []

        def reset(self, request):
            self.calls.append(("reset", request))

        def prepare(self, request, tokens, mask, *, defer_wait=False, device_layer1=False):
            self.calls.append(("prepare", request, tokens, mask, defer_wait, device_layer1))
            return SimpleNamespace(buffers=("host-layer1", "host-layer14"))

        def stage_device_c1_reference(self, request, rows):
            self.calls.append(("stage", request, rows))
            self.device_pending = request

    def model(host):
        value = object.__new__(HpuDeepseekV41ForCausalLM)
        value.pp_rank = 0
        value.step_ticket = None
        value.step_is_decode = False
        value._decode_prefix = None
        value._step_request_id = None
        value.engram_host = host
        return value

    warmup = Host()
    model(warmup).prepare_step("warmup", [1], is_decode=True, reset=True)
    assert warmup.calls == [
        ("reset", "warmup"),
        ("prepare", "warmup", [1], [False], True, False),
        ("stage", "warmup", "host-layer1"),
    ]
    decode = Host("decode")
    model(decode).prepare_step("decode", [12], is_decode=True)
    assert decode.calls == [("prepare", "decode", [12], [False], True, True)]
