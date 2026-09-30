# SPDX-License-Identifier: Apache-2.0
"""Reject missing acquisition metadata before spending a full-model run."""

from contextlib import contextmanager
import importlib
from pathlib import Path

import pytest

from tools.run_deepseek_v41 import configure_trace_artifacts


def test_trace_symbols_and_graphs_are_selected_before_compilation(tmp_path):
    environment = {}
    identity = configure_trace_artifacts(environment, tmp_path, dump_plans=True, enable_profiler=True)
    assert identity["ENABLE_PROFILER"] == "true" and identity["GRAPH_VISUALIZATION"] == "1"
    assert identity["HABANA_PROFILE"] == "1" and len(identity["profiler_config_sha256"]) == 64
    assert (tmp_path / "profiler-config.json").is_file()
    assert environment["GRAPH_VISUALIZATION_DIR"] == str(tmp_path / "graphs")
    assert environment["VLLM_TORCH_PROFILER_DIR"] == str(tmp_path / "traces")
    plain = configure_trace_artifacts({}, tmp_path, dump_plans=False, enable_profiler=False)
    assert plain != identity


def test_raw_sdk_capture_keeps_compiler_debug_disabled(tmp_path):
    import json

    environment = {"GRAPH_VISUALIZATION": "0"}
    identity = configure_trace_artifacts(
        environment, tmp_path, dump_plans=False, enable_profiler=False, raw_profiler=True
    )
    assert identity["ENABLE_PROFILER"] == "false"
    assert identity["HABANA_PROFILE"] == "1" and identity["GRAPH_VISUALIZATION"] == "0"
    assert environment["VLLM_HPU_DSV41_RAW_TRACE"] == "1"
    assert environment["VLLM_HPU_DSV41_RAW_SCOPE_ONLY"] == "1"
    config = json.loads((tmp_path / "profiler-config.json").read_text())
    host = next(plugin["values"] for plugin in config["Plugins"] if plugin["name"] == "HostProfiler")
    assert host["start_disabled"]["value"]
    assert not host["api_group"]["SYNAPSE"]["value"] and not host["api_group"]["SCAL"]["value"]


@pytest.mark.parametrize("ignore_frontend", [True, False])
def test_profile_control_reaches_workers_when_frontend_capture_is_disabled(monkeypatch, tmp_path, ignore_frontend):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from vllm.v1.engine import async_llm

    core = MagicMock(profile_async=AsyncMock())
    frontend = MagicMock()
    create_frontend = MagicMock(return_value=frontend)
    monkeypatch.setattr(async_llm, "maybe_register_config_serialize_by_value", lambda: None)
    monkeypatch.setattr(async_llm, "load_stat_logger_plugin_factories", lambda: [])
    monkeypatch.setattr(async_llm, "renderer_from_config", lambda *_: MagicMock())
    monkeypatch.setattr(async_llm, "InputProcessor", MagicMock())
    monkeypatch.setattr(async_llm, "OutputProcessor", MagicMock())
    monkeypatch.setattr(async_llm.EngineCoreClient, "make_async_mp_client", lambda **_: core)
    monkeypatch.setattr(async_llm, "TorchProfilerWrapper", create_frontend)
    config = SimpleNamespace(
        model_config=None,
        scheduler_config=SimpleNamespace(stream_interval=1),
        observability_config=SimpleNamespace(otlp_traces_endpoint=None),
        profiler_config=SimpleNamespace(
            profiler="torch", ignore_frontend=ignore_frontend, torch_profiler_dir=str(tmp_path)
        ),
    )
    engine = async_llm.AsyncLLM(config, executor_class=object, log_stats=False)
    asyncio.run(engine.start_profile("scope-only"))
    asyncio.run(engine.stop_profile())
    assert core.profile_async.await_count == 2
    core.profile_async.assert_any_await(True, "scope-only")
    core.profile_async.assert_any_await(False)
    if ignore_frontend:
        create_frontend.assert_not_called()
    else:
        frontend.start.assert_called_once()
        frontend.stop.assert_called_once()


def test_prefill_scopes_are_retained_by_torch_profiler(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_native_trace as trace

    labels = []

    @contextmanager
    def record(label):
        labels.append(label)
        yield

    monkeypatch.setattr(trace, "_active", False)
    monkeypatch.setattr(trace, "_torch_active", False)
    monkeypatch.setattr(trace.torch.profiler, "record_function", record)
    with trace.scope("inactive"):
        pass
    assert not labels
    trace.set_torch_annotations(True)
    try:
        assert trace.annotations_enabled()
        with trace.scope("v41::prefill::layer::layer39::C8192"):
            pass
        assert labels == ["v41::prefill::layer::layer39::C8192"]
    finally:
        trace.set_torch_annotations(False)


def test_tp4_expert_and_dense_shapes_keep_their_projection_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    for shape, expected in (([6, 5120, 1280], "W13"), ([6, 640, 5120], "W2")):
        assert expected in report.classify("expert_n256/gemm", "BatchGemm", [{}, {"shape": shape}], [])[1]
    for shape, expected in (([8192, 1280], "wq_b"), ([5120, 2048], "wo_b")):
        assert expected in report.classify("layer/attention/gemm", "GEMM", [{"dtype": "bf16"}, {"shape": shape}], [])[1]


@pytest.mark.parametrize(
    "kernel,category",
    [
        ("deepseek_v41_mhc_post_collapse_gaudi2", "mHC"),
        ("deepseek_v41_main_publish_gather_gaudi2", "Attention"),
        ("deepseek_v41_main_reuse_gather_gaudi2", "Attention"),
        ("deepseek_v41_silu_activate_tile_gaudi2", "路由专家"),
        ("deepseek_v41_silu_quant_tile_gaudi2", "路由专家"),
    ],
)
def test_tp4_new_native_kernels_retain_functional_attribution(monkeypatch, kernel, category):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    assert report.classify("opaque_custom_node", kernel, [], [])[0] == category


@pytest.mark.parametrize("operation", ["main_publish", "main_reuse"])
@pytest.mark.parametrize("dtype,projection", [("bf16", "QK"), ("float32", "PV")])
def test_shared_main_matrices_retain_qk_pv_identity(monkeypatch, operation, dtype, projection):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    category, role = report.classify(f"custom_deepseek_v41_{operation}_mla_gaudi2", "BatchGemm", [{"dtype": dtype}], [])
    assert category == "Attention" and projection in role


def test_activity_export_retains_packet_api_and_merges_lanes(tmp_path, monkeypatch):
    import gzip
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    rank = tmp_path / "rank0"
    rank.mkdir()
    nodes = [
        dict(recipe="0", engine=engine, kernel=kernel, node=kernel)
        for engine, kernel in [("TPC", "null"), ("DMA", "pdma_tx_commands"), ("DMA", "hcl_transfer")]
    ]
    for name, value in {
        "inventory.json": dict(
            nodes=nodes, host_enqueues=[], cpu_markers=[], trace_sha256="fixture", base_time_nanoseconds=0
        ),
        "recipe-symbols.json": dict(recipes=[]),
        "node-contracts.json": [],
        "device-windows.json": dict(capture_order=[], topology=dict(tensor_parallel_size=4)),
    }.items():
        (rank / name).write_text(json.dumps(value))
    (tmp_path / "graph-manifest.json").write_text("[]")
    with gzip.open(rank / "hardware.jsonl.gz", "wt") as stream:
        for packet in [
            [100, 200, "TPC0", 0, 0, "api0"],
            [100, 200, "TPC1", 0, 0, "api0"],
            [400, 100, "PDMA0", 1, 0, "api1"],
            [600, 100, "HCL dedicated DMA", 2, 0, "api2"],
        ]:
            stream.write(json.dumps(packet) + "\n")
    with gzip.open(rank / "host.jsonl.gz", "wt"):
        pass
    report.analyze(tmp_path, 0, dict(windows_us=[[0, 1000]], tokens=[1], export_activity_intervals=True))
    with gzip.open(rank / "activity-intervals.json.gz", "rt") as stream:
        exported = json.load(stream)
    groups = {row["category"]: row["intervals_us"] for row in exported["groups"]}
    assert groups == {"设备调度": [[100, 300]], "命令搬运": [[400, 500]], "通信相关DMA": [[600, 700]]}
    accounting = json.loads((rank / "kernel-breakdown.json").read_text())
    assert sum(accounting["partition"].values()) == pytest.approx(1.0)


def test_tp4_decode_windows_require_forty_layers_and_completed_consumer():
    from tools.analyze_deepseek_v41_trace import tp4_windows

    marks = [[0, 1, "v41::verify_and_commit::PP0::decode::P16384::C1::emit1"]]
    for offset in (10, 100):
        for layer in range(0, 40, 4):
            marks.append([offset + layer, 1, f"v41::compiled::layers{layer}-{layer + 3}::C1"])
    marks += [
        [60, 5, "v41::verify_and_commit::PP0::decode::P16385::C1::emit1"],
        [150, 5, "v41::verify_and_commit::PP0::decode::P16386::C1::emit1"],
    ]
    inv = {"cpu_markers": marks, "base_time_nanoseconds": 1000000}
    actual = tp4_windows(inv, "decode")
    assert actual["tokens"] == [16385, 16386]
    assert actual["windows_us"] == [(1, 65), (65, 155)]
    # A truncated final group must never become a complete cycle.
    inv["cpu_markers"] = [row for row in marks if row[0] != 136]
    assert tp4_windows(inv, "decode")["tokens"] == [16385]


def test_tp4_async_cycles_use_worker_consumption_not_early_sampling_return():
    from tools.analyze_deepseek_v41_trace import tp4_windows

    marks = [
        [0, 1, "v41::worker_commit::PP0::decode::P16384::C1::emit1"],
        [2, 1, "v41::verify_and_commit::PP0::decode::P16385::C1::emit1"],
        [3, 1, "v41::verify_and_commit::PP0::decode::P16386::C1::emit1"],
    ]
    for index, layer in enumerate([*range(12, 40, 4), 0, 4, 8]):
        marks.append([10 + 4 * index, 1, f"v41::compiled::layers{layer}-{layer + 3}::C1"])
    marks.append([55, 5, "v41::worker_commit::PP0::decode::P16385::C1::emit1"])
    result = tp4_windows({"cpu_markers": marks}, "decode")
    assert result["asynchronous_completion"] is True
    assert result["tokens"] == [16385] and result["windows_us"] == [(1, 60)]
    assert "next-token prefix" in result["boundary"]


def test_native_stage_windows_use_completed_target_without_per_group_scopes():
    from tools.analyze_deepseek_v41_trace import tp4_windows

    marks = [
        [0, 1, "v41::worker_commit::PP0::decode::P16384::C1::emit1"],
        [3, 2, "v41::target::PP0::decode::C1"],
        [6, 14, "v41::worker_commit::PP0::decode::P16385::C1::emit1"],
        [22, 2, "v41::target::PP0::decode::C1"],
        [25, 15, "v41::worker_commit::PP0::decode::P16386::C1::emit1"],
    ]
    with pytest.raises(ValueError):
        tp4_windows({"cpu_markers": marks}, "decode")
    proof = dict(kind="complete native stage replay", decode_steps=3, native_entry_replays=3)
    actual = tp4_windows({"cpu_markers": marks}, "decode", native_coverage=proof)
    assert actual["tokens"] == [16385, 16386]
    assert actual["windows_us"] == [(1, 20), (20, 40)]
    # A native counter alone does not cover a missing or truncated target.
    marks[3][1] = 100
    assert tp4_windows({"cpu_markers": marks}, "decode", native_coverage=proof)["tokens"] == [16385]


def test_tp4_prefill_combines_both_chunks_through_first_token():
    from tools.analyze_deepseek_v41_trace import tp4_windows

    marks = []
    for start in (0, 100):
        marks.append([start, 50, "v41::target::PP0::prefill::C8192"])
        marks.extend([start + layer, 1, f"v41::prefill::layer::layer{layer}::C8192"] for layer in range(40))
    marks.extend(
        [
            [60, 1, "v41::verify_and_commit::PP0::prefill::P0::C8192::emit0"],
            [160, 15, "v41::verify_and_commit::PP0::prefill::P8192::C8192::emit1"],
        ]
    )
    result = tp4_windows({"cpu_markers": marks}, "prefill")
    assert result["unit"] == "request" and result["windows_us"] == [(0, 175)]
    assert set(result["coverage_proof"]["layer_counts"].values()) == {2}
    complete = tp4_windows(
        {"cpu_markers": marks, "base_time_nanoseconds": 1_000_000, "all_activity_start_us": -100},
        "prefill",
        request_start_ns=950_000,
    )
    assert complete["windows_us"] == [(-50, 175)]
    assert complete["coverage_proof"]["before_first_target_ms"] == 0.05


def test_eager_kernel_calls_use_enqueues_and_reject_ambiguous_packets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    rows = [(0, 2, "lane0"), (1, 3, "lane1"), (10, 12, "lane0"), (11, 13, "lane1")]
    samples, count, error = report.eager_invocation_samples(rows, 2)
    assert samples == [0.003, 0.003] and count == 2 and error is None
    assert report.eager_invocation_samples(rows[:-1], 2)[1] is None
    assert report.eager_invocation_samples([(0, 20, "lane0"), *rows[1:]], 2)[1] is None


def test_four_engine_partition_does_not_count_overlap_twice():
    from tools.deepseek_v41_trace_accounting import device_partition

    result = device_partition({"TPC": [(0, 4)], "MME": [(2, 6)], "DMA": [(5, 8)], "NIC": [(7, 9)]}, [(0, 10)], 1)
    assert result == {
        "TPC": 2,
        "TPC+MME": 2,
        "MME": 1,
        "MME+DMA": 1,
        "DMA": 1,
        "DMA+NIC": 1,
        "NIC": 1,
        "no_recorded_device_activity": 1,
    }
    assert sum(result.values()) == 10


def test_host_overlap_counts_duplicated_device_lanes_once(tmp_path):
    import gzip
    import json
    from tools.deepseek_v41_trace_accounting import host_accounting

    with gzip.open(tmp_path / "host.jsonl.gz", "wt") as stream:
        for row in [
            [0, 10000, "1", "1", "cpu_op", "aten::copy_"],
            [5000, 5000, "1", "2", "runtime", "synchronizeEvent"],
        ]:
            stream.write(json.dumps(row) + "\n")
    compute = [(0, 3000), (2000, 8000)] * 24
    result = host_accounting(tmp_path, [(0, 10000)], compute, 1000, export_intervals=True)
    assert result["CPU_operator_dispatch"] == dict(activity_ms=10, overlap_with_compute_ms=8, outside_compute_ms=2)
    assert result["observed_synchronization_or_wait"] == dict(
        activity_ms=5, overlap_with_compute_ms=3, outside_compute_ms=2
    )
    saved = json.loads((tmp_path / "host-breakdown.json").read_text())
    assert saved["activity_intervals_us"]["CPU_operator_dispatch"] == [[0, 10000]]


def test_tp4_ledger_distinguishes_peer_work_from_all_rank_gaps(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from report_deepseek_v41_latency_ledger import sweep

    device = [
        (0, 4, ("device", 0, "TPC", "Attention")),
        (2, 6, ("device", 1, "MME", "路由专家")),
        (6, 8, ("device", 2, "DMA", "通信相关DMA")),
    ]
    host = [(0, 10, ("host", "compiled分组入口内")), (7, 9, ("host", "HCCL API"))]
    result = sweep(device, host, [(0, 5), (5, 10)])
    assert result["totals_us"] == {
        "设备kernel：Attention": 2,
        "设备kernel：跨模块重叠": 2,
        "设备kernel：路由专家": 2,
        "仅设备搬运：通信相关DMA": 2,
        "四卡无可见设备活动：compiled分组入口内": 2,
    }
    assert result["rank0_no_device_other_rank_active_us"] == 4
    assert result["hccl_overlap_with_gap_us"] == {"四卡无可见设备活动：compiled分组入口内": 1}
    assert result["per_window_totals_us"] == [
        {"设备kernel：Attention": 2, "设备kernel：跨模块重叠": 2, "设备kernel：路由专家": 1},
        {"设备kernel：路由专家": 1, "仅设备搬运：通信相关DMA": 2, "四卡无可见设备活动：compiled分组入口内": 2},
    ]


def test_streaming_interval_accounting_rejects_coalesced_tokens():
    from tools.qualify_deepseek_v41_request import streaming_intervals

    events = [dict(arrival_s=value, count=1) for value in (1, 2, 4)]
    assert streaming_intervals(events)["all"]["mean_ms"] == 1500
    events[-1]["count"] = 2
    assert not streaming_intervals(events)["valid"]


def test_tp4_rejects_an_interior_missing_decode_cycle():
    from tools.analyze_deepseek_v41_trace import tp4_windows

    marks = [[0, 1, "v41::verify_and_commit::PP0::decode::P16384::C1::emit1"]]
    for cycle in range(3):
        offset = 100 * cycle + 10
        for layer in range(0, 40, 4):
            if cycle == 1 and layer == 36:
                continue
            marks.append([offset + layer, 1, f"v41::compiled::layers{layer}-{layer + 3}::C1"])
        marks.append([offset + 60, 5, f"v41::verify_and_commit::PP0::decode::P{16385 + cycle}::C1::emit1"])
    with pytest.raises(ValueError, match="interior"):
        tp4_windows({"cpu_markers": marks}, "decode")


@pytest.mark.parametrize(
    "kernel,category",
    [
        ("custom_deepseek_v41_router_logits_top6_gaudi2", "Router"),
        ("custom_deepseek_v41_selected_mla_gather_gaudi2", "Attention"),
        ("custom_deepseek_v41_kv_norm_rope_bf16_gaudi2", "Attention"),
    ],
)
def test_actual_tp4_fused_guids_keep_their_model_roles(monkeypatch, kernel, category):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    assert report.classify(kernel, kernel, [], [])[0] == category


def test_report_tensor_details_distinguish_engine_context_namespaces():
    from tools.render_deepseek_v41_trace import node_details

    details = [
        dict(recipe_id="42", engine=engine, context_id=0, tensor_shape=shape)
        for engine, shape in (("TPC", [5120]), ("MME", [1, 5120]))
    ]
    for engine in ("TPC", "MME"):
        for key in (["42", 0], ["42", engine, 0]):
            selected = node_details(dict(engine=engine, node_keys=[key]), details)
            assert len(selected) == 1 and selected[0]["engine"] == engine


def test_mme_multiple_rois_do_not_become_multiple_physical_calls(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    rows = [(0, 1, "MME0"), (0, 1, "MME1"), (2, 3, "MME0"), (2, 3, "MME1")]
    samples, count, error = report.invocation_samples(rows, {"device_type": 0}, expected=1)
    assert samples == [0.003] and count == 1 and error is None


def test_selected_mla_fp32_pv_is_not_misclassified_as_compressor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_trace")
    node = "layers/2/attention/custom_deepseek_v41_paged_mla_mme_gaudi2/batch_gemm/op_1"
    category, purpose = report.classify(
        node,
        "gemm",
        [dict(dtype="float32", shape=[1, 16, 640]), dict(dtype="float32", shape=[1, 640, 512])],
        [dict(dtype="bf16", shape=[1, 16, 512])],
    )
    assert category == "Attention" and "PV" in purpose and "Compressor" not in purpose
    category, purpose = report.classify(
        node,
        "gemm",
        [dict(dtype="bf16", shape=[1, 16, 512]), dict(dtype="bf16", shape=[1, 640, 512])],
        [dict(dtype="float32", shape=[1, 16, 640])],
    )
    assert category == "Attention" and "QK" in purpose


def test_raw_recipe_name_reuse_needs_exact_device_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from normalize_deepseek_v41_raw_trace import resolve_recipe_starts

    starts = [(1.0, "1000", "reused"), (2.0, "2000", "reused"), (3.0, "3000", "reused")]
    resolved, unknown = resolve_recipe_starts(
        starts, {"reused": {10, 20}}, {("1000", "reused"): {10}, ("2000", "reused"): {20}}, {"reused": {10, 20}}
    )
    assert [row[2] for row in resolved] == ["10@reused:", "20@reused:"]
    assert len(unknown) == 1 and unknown[0]["candidate_ids"] == [10, 20]


@pytest.mark.parametrize("wall_step_ns", [-500_000_000, 700_000_000])
def test_scope_only_raw_clock_survives_wall_clock_steps(monkeypatch, wall_step_ns):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from normalize_deepseek_v41_raw_trace import Clock

    raw0, wall0 = 432_000_000_000_000, 1_790_000_000_000_000_000
    samples = []
    for delta in (0, 2_000_000_000):
        wall = wall0 + delta + (wall_step_ns if delta else 0)
        samples.append(
            dict(
                monotonic_raw_before_ns=raw0 + delta,
                monotonic_raw_after_ns=raw0 + delta,
                wall_before_ns=wall,
                wall_after_ns=wall,
                synapse_clock_ns=2 * (raw0 + delta),
            )
        )
    metadata = dict(clock_samples=samples, scope_clock_domain="CLOCK_MONOTONIC_RAW")
    clock = Clock(metadata, raw0)
    assert clock.raw((raw0 + 123_000_000) / 1000) == 123_000
    assert clock.synapse(2 * (raw0 + 123_000_000) / 1000) == 123_000
    assert clock.raw((raw0 + 2_000_000_000) / 1000) == 2_000_000
    # Legacy Torch traces still need a valid wall-clock calibration.
    del metadata["scope_clock_domain"]
    with pytest.raises(ValueError, match="clock calibration"):
        Clock(metadata, wall0)


def test_raw_unknown_tpc_and_mme_ports_pair_by_identity(tmp_path, monkeypatch):
    import csv

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from normalize_deepseek_v41_raw_trace import owned_csv_rows

    fields = ["Engine", "Recipe ID", "Context ID", "Unique Node ID", "Name", "Event type", "Timestamp[usec]"]
    rows = [
        ["TPC 1", "32", "1", "", "TPC_SPU_START 1", "BEGIN", "1"],
        ["TPC 1", "32", "1", "", "TPC_SPU_START_TO_SPU_HALT 1", "END", "2"],
        ["MME 0", "", "0", "", "START_EVENT", "BEGIN", "3"],
        ["MME 0", "", "0", "", "MMEH_WB1_MON_TS_BIT1", "END", "4"],
    ]
    path = tmp_path / "sdk.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    pairs = list(owned_csv_rows([(path, 0, 5000, 10000)]))
    assert len(pairs) == 2 and all(first["Engine"] == last["Engine"] for first, last in pairs)


def test_mme_role_requires_unambiguous_weight_producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from report_deepseek_v41_trace import expert_mme_owners

    graph = {"path": "captured.json"}

    def decode(name, width):
        return dict(
            graph=graph,
            symbol=dict(device_type=1, node=name, kernel="custom_deepseek_v41_prefill_permuted_bf16_gaudi2"),
            inputs=[dict(name="ids"), dict(name="packed-" + name, shape=[384, 1, width])],
            outputs=[dict(name=name)],
        )

    rows = {
        "w13": decode("w13", 327680),
        "w2": decode("w2", 40960),
        "mme": dict(
            graph=graph,
            symbol=dict(device_type=0, node="tile", kernel="batch_gemm"),
            inputs=[dict(name="activation"), dict(name="tile-view", alias="w13")],
            outputs=[],
        ),
    }
    assert expert_mme_owners(rows)["mme"]["role"] == "W13"
    rows["mixed"] = dict(
        graph=graph,
        symbol=dict(device_type=1, node="mixed", kernel="add"),
        inputs=[dict(name="w13"), dict(name="w2")],
        outputs=[dict(name="mixed")],
    )
    rows["mme"]["inputs"][1] = dict(name="mixed")
    assert "mme" not in expert_mme_owners(rows)


def test_tp4_prefill_single_16k_chunk_includes_first_token_consumer():
    from tools.analyze_deepseek_v41_trace import tp4_windows

    marks = [[100, 50, "v41::target::PP0::prefill::C16384"]]
    marks.extend([101 + layer, 1, f"v41::prefill::layer::layer{layer}::C16384"] for layer in range(40))
    marks.append([160, 15, "v41::verify_and_commit::PP0::prefill::P0::C16384::emit1"])
    result = tp4_windows({"cpu_markers": marks}, "prefill")
    assert result["windows_us"] == [(100, 175)]
    assert result["coverage_proof"]["prompt_tokens"] == 16384
    assert set(result["coverage_proof"]["layer_counts"].values()) == {1}
    with pytest.raises(ValueError, match="forty-layer"):
        tp4_windows({"cpu_markers": marks[:20] + marks[21:]}, "prefill")
