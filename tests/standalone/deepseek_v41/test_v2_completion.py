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


@pytest.mark.parametrize("position", [127, 16384, 131072, 524287])
@pytest.mark.parametrize("temperature", [0., 1.])
def test_tail_next_position_is_independent_of_candidate_certificate(position, temperature):
    from vllm_gaudi.models.deepseek_v41_program import PreparedGreedyTail

    logits = torch.linspace(-1, 1, 512).reshape(1, -1)
    owner = SimpleNamespace(
        _head_projection=lambda hidden: logits, device_sampling=True, device_next_position=True,
        sampling_params=torch.tensor([[temperature, .95, -1.]]),
        sampling_seed=torch.tensor([42], dtype=torch.int32),
        sampling_origin=torch.tensor([position - 1], dtype=torch.int32), tp_rank=0,
        all_gather=lambda packet, dim: torch.cat([packet] * 4, dim),
    )
    positions = torch.tensor([position], dtype=torch.int32)
    payload = PreparedGreedyTail.forward(owner, torch.empty(1, 5120), positions)
    assert len(payload) == 5
    assert payload[4].dtype == torch.int32 and payload[4].tolist() == [position + 1]
    assert positions.tolist() == [position]
    assert payload[4].data_ptr() != positions.data_ptr()
    if temperature:
        # Duplicate candidate scores force exact full fallback, while the
        # continuation coordinate remains valid and token independent.
        assert (payload[0] & 1).item() == 0


@pytest.mark.parametrize("failed", [False, True])
def test_sampler_shutdown_drains_before_retirement_and_is_idempotent(failed):
    from concurrent.futures import Future

    runner = object.__new__(V41V2ModelRunner)
    future = Future()
    calls = []
    if failed:
        future.set_exception(RuntimeError("repair failed"))
    else:
        future.set_result(42)
    runner._sampling_completion_future = future
    runner._sampling_completion_executor = SimpleNamespace(shutdown=lambda **kwargs: calls.append(kwargs))
    if failed:
        with pytest.raises(RuntimeError, match="repair failed"):
            runner.prepare_shutdown()
    else:
        runner.prepare_shutdown()
    assert calls == [{"wait": True}]
    assert runner._sampling_completion_future is None
    runner.prepare_shutdown()
    assert calls == [{"wait": True}]


@pytest.mark.parametrize("value", ["11-14;16-19", "11,12,13,14;16,17,18,19", "11,13-14;19,16-18"])
def test_sampling_helper_uses_shared_cpu_list_and_range_parser(monkeypatch, value):
    from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import sampling_completion_helper_cpu

    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_HELPER_CPUS", value)
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert sampling_completion_helper_cpu() == 16
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert sampling_completion_helper_cpu() is None


@pytest.mark.parametrize("decode_start", [None, 16384])
def test_sampling_admission_uploads_once_including_startup_request(decode_start):
    runner = object.__new__(V41V2ModelRunner)
    program = SimpleNamespace(device_sampling=True, sampling_params=torch.zeros(1, 3),
                              sampling_seed=torch.zeros(1, dtype=torch.int32),
                              sampling_counter=torch.zeros(1, dtype=torch.int32),
                              sampling_origin=torch.zeros(1, dtype=torch.int32))
    runner.model = SimpleNamespace(program=program)
    runner.v2_completion = True
    request = SimpleNamespace(req_id="admission", output=[], sampling_params=SimpleNamespace(
        temperature=1., top_p=.95, top_k=-1, seed=42))
    if decode_start is not None:
        request.decode_start = decode_start
    runner._prepare_device_sampling_request(request)
    assert program.sampling_seed.item() == 42
    assert program.sampling_origin.item() == (decode_start - 1 if decode_start is not None else 0)
    program.sampling_counter.fill_(7)
    program.sampling_origin.fill_(123)
    runner._prepare_device_sampling_request(request)
    assert program.sampling_counter.item() == 7
    assert program.sampling_origin.item() == 123


@pytest.mark.parametrize("covered", [False, True])
def test_sync_sampling_repairs_before_publishing_token(covered):
    runner = object.__new__(V41V2ModelRunner)
    runner._v2_async_step = False
    request = SimpleNamespace(req_id="sync", decode_start=1, output=[])
    selected = torch.tensor([[11]], dtype=torch.int32)
    status = torch.tensor([[22 + int(covered)]], dtype=torch.int32)
    payload = status, torch.zeros(1, 4), torch.zeros(1, 4), selected
    runner._device_sampling_payload = payload
    runner.pending = request, 1, 1, 1, False, True, selected
    runner.tp4_token_readback = lambda value: (value, Done())
    runner._token_copy = None
    runner.pp = SimpleNamespace(device_commit_enabled=False, group=SimpleNamespace(is_last_rank=True),
                                finish_single=lambda count, token: (count, [token]))
    runner.model = SimpleNamespace(complete_step=lambda count: None)
    repairs = []
    runner._repair_device_sample = lambda value, destination: repairs.append(value) or 19
    result = runner._sample_single()
    assert result.sampled_token_ids == [[11 if covered else 19]]
    assert request.output == [11 if covered else 19]
    assert repairs == ([] if covered else [payload])
    assert runner._device_sampling_payload is None


@pytest.mark.parametrize("tp_size,pp_size", [(2, 2), (4, 1)])
def test_shared_v2_configuration_accepts_complete_native_dependencies(monkeypatch, tp_size, pp_size):
    from vllm_gaudi.ops.deepseek_v41_config import validate_v2

    for name in (
        "V2", "GRAPH_REPLAY", "DIRECT_TOKEN_IDS", "FIXED_POSITIONS", "NATIVE_INPUT_GRAPH",
        "V2_EARLY_INPUT_COMMIT", "V2_SEGMENTED_PREFIX", "V2_DEVICE_ENGRAM", "TP_MHC_OVERLAP",
        "ENGRAM_NATIVE_C1", "ENGRAM_C1_PACKET", "ENGRAM_DIRECT_INPUT",
    ):
        monkeypatch.setenv("VLLM_HPU_DSV41_" + name, "1")
    monkeypatch.setenv("VLLM_HPU_TP2_NATIVE_JOINT_PLAN", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_FUSED_STAGE_IO", "0")
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "0")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size),
        use_v2_model_runner=True, scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=None,
    )
    validate_v2(config)
    config.scheduler_config.async_scheduling = False
    with pytest.raises(ValueError, match="V2 requires"):
        validate_v2(config)


@pytest.mark.parametrize("batch_enabled", [False, True])
def test_device_engram_abi_is_required_even_with_batch_capacity(monkeypatch, batch_enabled, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_host as host

    monkeypatch.setattr(host, "host_native", lambda: SimpleNamespace(abi_version=1, c1_abi_version=1))
    flags = dict(
        VLLM_HPU_DSV41_BATCH_DECODE=batch_enabled,
        VLLM_HPU_DSV41_V2_DEVICE_ENGRAM=True,
        VLLM_HPU_DSV41_ENGRAM_NATIVE_C1=True,
        VLLM_HPU_DSV41_ENGRAM_C1_PACKET=True,
        VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT=True,
        VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH=True,
        VLLM_HPU_DSV41_GRAPH_REPLAY=True,
    )
    for key, value in flags.items():
        monkeypatch.setenv(key, "1" if value else "0")
    with pytest.raises(RuntimeError, match="Device Engram requires native host C1 ABI 2"):
        host.EngramHost(tmp_path, 0, "hpu")


def fixture():
    runner = object.__new__(V41V2ModelRunner)
    runner.prefix_checkpoints = None
    runner.requests = {"a": RequestState("a", [1], [], None, ([1],), output=[10], num_computed_tokens=1)}
    runner.active_request = "a"
    calls = []
    runner.model = SimpleNamespace(complete_step=lambda count: calls.append(("model", count)))
    token = torch.tensor([11], dtype=torch.int32)
    runner.pp = SimpleNamespace(
        generation=2, commits=0, commit_token=token, complete_packet=lambda: calls.append(("packet",))
    )
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

    assert calls == [("packet",), ("model", 1)]
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
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"), max_model_len=512),
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=2),
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=None,
    )
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

    monkeypatch.setattr(
        tp2_fused_ar_norm, "_resolve_runtime", lambda: (SimpleNamespace(copy_sampled_tokens_to_host=copy), None, None)
    )
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
    assert calls == [("model", 1), ("packet",)]
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
            calls.append(("token_wait",))

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


@pytest.mark.parametrize("covered", [False, True])
@pytest.mark.parametrize("device_position", [False, True])
def test_sampling_stages_only_independent_inputs_before_certifying_prefix(monkeypatch, covered, device_position):
    for suffix in ("EARLY_INPUT_COMMIT", "SEGMENTED_PREFIX", "DEVICE_ENGRAM"):
        monkeypatch.setenv("VLLM_HPU_DSV41_V2_" + suffix, "1")
    runner, calls = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    runner.position_views = {1: torch.empty(1, dtype=torch.int32)}
    runner.position_bank = SimpleNamespace(copy_into=lambda *args: calls.append(("position",)))
    runner._prefix_authorized = lambda *args: True
    runner._sampling_prefix_handoff = True
    monkeypatch.setattr("vllm_gaudi.v1.worker.deepseek_v41_v2_runner.time.sleep",
                        lambda value: calls.append(("handoff", value)))

    class OrderedDone:
        def synchronize(self):
            calls.append(("certificate",))

    runner.model = SimpleNamespace(
        tensor_parallel_size=4,
        complete_step=lambda count: calls.append(("model", count)),
        prepare_device_engram=lambda *args: calls.append(("device_engram",)),
        begin_decode_prefix=lambda *args: calls.append(("prefix",)),
    )
    runner._completion = CompletionRecord(
        "a", 2, 1, torch.tensor([[22 + int(covered)]]), OrderedDone(), runner.pp.commit_token,
        lambda: calls.append(("repair",)) or 19,
        torch.tensor([2], dtype=torch.int32) if device_position else None,
    )
    runner._consume_completion(SimpleNamespace())
    assert [row[0] for row in calls] == (
        ["model"] + ([] if device_position else ["position"]) + ["certificate"] + ([] if covered else ["repair"])
        + ["device_engram", "prefix", "handoff", "packet"]
    )
    assert runner.requests["a"].output[-1] == (11 if covered else 19)
    if device_position:
        assert runner._next_position[:2] == ("a", 2)
        assert runner._next_position[2].tolist() == [2]
    else:
        assert runner._next_position is None


@pytest.mark.parametrize("policy,before,handoff", [
    ("original", True, False), ("prepare", False, False),
    ("yield", True, True), ("combined", False, True),
])
def test_private_continuation_control_requires_retired_request(monkeypatch, policy, before, handoff):
    from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

    worker = object.__new__(HPUWorker)
    runner = SimpleNamespace(model=SimpleNamespace(program=SimpleNamespace(device_sampling=True), tp_rank=1),
                             _completion=None, _prefix_started=None, active_request=None)
    worker.model_runner = runner
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "0")
    with pytest.raises(RuntimeError, match="development endpoints"):
        worker.set_decode_continuation_diagnostic(policy)
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    runner.active_request = "live"
    with pytest.raises(RuntimeError, match="retirement"):
        worker.set_decode_continuation_diagnostic(policy)
    assert not hasattr(runner, "_sampling_prefix_handoff")
    runner.active_request = None
    assert worker.set_decode_continuation_diagnostic(policy)["gpu_plan_changed"] is False
    assert runner._certificate_before_staging == before
    assert runner._sampling_prefix_handoff == handoff


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
    assert calls == [("model", 1), ("packet",)]
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
    runner.model = SimpleNamespace(
        decode_prefix_ready=lambda search: search == 512, program=SimpleNamespace(length=512)
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1}, finished_req_ids=set(), scheduled_spec_decode_tokens={})
    assert runner._prefix_authorized(SimpleNamespace(request_id="a", start=10), scheduled)


@pytest.mark.parametrize("append_safe", [False, True])
def test_next_prefix_publishes_new_page_before_readback_only_for_same_owner(monkeypatch, append_safe):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner, _ = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    calls = []
    runner.state = SimpleNamespace(blocks=32,
                                   append_single_pages=lambda *args: calls.append(args) or append_safe)
    runner.requests["a"].block_ids = ([5, 3], )
    runner.model = SimpleNamespace(
        tensor_parallel_size=4,
        decode_prefix_ready=lambda search: search == 512,
        program=SimpleNamespace(length=512),
    )
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"a": 1},
        finished_req_ids=set(),
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a"],
                                              resumed_req_ids=set(),
                                              num_computed_tokens=[256],
                                              new_block_ids=[([7], )]),
    )
    assert runner._prefix_authorized(SimpleNamespace(request_id="a", start=255), scheduled) == append_safe
    assert calls == [("a", [5, 3], [7], 32)]
    assert runner.requests["a"].block_ids == ([5, 3], )
    assert runner.audit.get("v2_prefix_page_appends", 0) == int(append_safe)


@pytest.mark.parametrize("position", [510, 511, 512, 1023, 1024, 2558, 2559, 2560, 524287, 1048574])
def test_runtime_indexer_reuses_bound_prefix_across_history_boundaries(monkeypatch, position):
    monkeypatch.setenv("VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX", "1")
    runner, _ = fixture()
    runner.pp.group = SimpleNamespace(is_first_rank=True)
    from vllm_gaudi.v1.worker.deepseek_v41_runner import runtime_search_length

    queries = []
    bound = runtime_search_length(position + 1, 1, 1 << 20)
    runner.model = SimpleNamespace(
        decode_prefix_ready=lambda search: queries.append(search) or True,
        program=SimpleNamespace(length=1 << 20, search_length=bound, runtime_indexer=True),
    )
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
        value.is_first_stage, value.is_last_stage, value.tensor_parallel_size = True, False, 2
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


def test_covered_sampling_certificate_does_not_repair():
    repaired = []
    record = CompletionRecord('a', 1, 16, torch.tensor([[63]], dtype=torch.int32), Done(),
                              torch.tensor([31]), lambda: repaired.append(1))
    assert record.token() == 31
    assert record.token() == 31
    assert not repaired


def test_sampling_repair_runs_once_across_two_consumers():
    from concurrent.futures import ThreadPoolExecutor

    calls = []
    record = CompletionRecord('a', 1, 16, torch.tensor([[62]], dtype=torch.int32), Done(),
                              torch.tensor([31]), lambda: calls.append(1) or 47)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: record.token(), range(16)))
    assert results == [47] * 16 and calls == [1]


def test_sampling_bad_certificate_is_rejected():
    record = CompletionRecord('a', 1, 16, torch.tensor([[31, 2]], dtype=torch.int32), Done(),
                              torch.tensor([31]), lambda: 47)
    with pytest.raises(RuntimeError, match='certificate'):
        record.token()


def test_native_sampling_warmup_exercises_completion_and_same_draw_full_fallback():
    runner = object.__new__(V41V2ModelRunner)
    selected = torch.tensor([[31]], dtype=torch.int32)
    payload = torch.tensor([[62]], dtype=torch.int32), torch.zeros(1, 4), torch.zeros(1, 4), selected
    runner.model = SimpleNamespace(tp_rank=0, program=SimpleNamespace(
        replay_owner=SimpleNamespace(sampling_tail_values=lambda hidden: payload)))
    runner._device_sampling_owner = (123, 'warm', 1., 1., -1, 42)
    runner.device_sampling_stats = {'warm': {'fallbacks': 0}}
    calls = []

    def sample():
        assert runner.pending[-1] is selected
        assert runner._device_sampling_payload is payload and runner._v2_async_step
        runner.device_sampling_stats['warm']['fallbacks'] += 1
        runner._completion = SimpleNamespace(token=lambda: 47)
        calls.append('completion')

    def full(local, controls, *, filtered):
        assert local is payload[1] and controls is payload[2] and not filtered
        calls.append('same-draw-reference')
        return torch.tensor([[47]], dtype=torch.int32)

    runner._sample_single = sample
    runner._sample_full_local = full
    runner.tp4_token_readback = lambda value: (value, Done())
    runner._validate_device_sampling_warmup(torch.zeros(1, 4))
    runner._validate_device_sampling_warmup(torch.zeros(1, 4))
    assert calls == ['completion', 'same-draw-reference']
    assert runner.pending is None and runner._completion is None and not runner._v2_async_step


@pytest.mark.parametrize('top_p,top_k,filtered', [(.95, -1, True), (1., -1, False), (1., 4, True)])
def test_sampling_repair_retains_original_filter_mode(top_p, top_k, filtered):
    runner = object.__new__(V41V2ModelRunner)
    runner._device_sampling_owner = (123, 'repair', 1., top_p, top_k, 42)
    runner.device_sampling_stats = {'repair': {'fallbacks': 0}}
    runner.audit = {}
    local, controls = torch.zeros(1, 4), torch.zeros(1, 4)
    destination = torch.tensor([[31]], dtype=torch.int32)

    def full(value, params, *, filtered):
        assert value is local and params is controls
        assert filtered == (top_p < 1 or top_k > 0)
        return torch.tensor([[47]], dtype=torch.int32)

    runner._sample_full_local = full
    runner.tp4_token_readback = lambda value: (value, Done())
    assert runner._repair_device_sample((torch.zeros(1, 1), local, controls, destination), destination) == 47
    assert destination.item() == 47 and runner.device_sampling_stats['repair']['fallbacks'] == 1
