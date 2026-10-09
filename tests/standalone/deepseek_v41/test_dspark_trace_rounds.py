# SPDX-License-Identifier: Apache-2.0
"""Keep C6 round cycles distinct from accepted output-token counts."""
import pytest

from tools.analyze_deepseek_v41_trace import speculative_windows


def trace_fixture():
    positions = [16384, 16386, 16392, 16393, 16396]
    marks = []
    for index, position in enumerate(positions):
        start = index * 1000
        marks += [[start + 10, 100, "v41::target::PP0::decode::C6"],
                  [start + 120, 50, f"v41::verify_and_commit::PP0::decode::P{position}::C6::emit1"],
                  [start + 180, 20, "v41::verify_and_commit::final_consume"]]
    return dict(cpu_markers=marks, base_time_nanoseconds=123000000), marks


def windows(inventory, coverage=None):
    return speculative_windows(inventory, dict(tensor_parallel_size=4, pipeline_parallel_size=1), coverage
                               or dict(kind="complete native stage replay"))


def test_variable_acceptance_counts_actual_output_not_six_target_rows():
    inventory, _ = trace_fixture()
    actual = windows(inventory)
    assert actual["unit"] == "round"
    assert actual["tokens"] == [16386, 16392, 16393]
    assert actual["windows_us"] == [(200, 1200), (1200, 2200), (2200, 3200)]
    assert actual["committed_tokens"] == [6, 1, 3]
    assert actual["total_committed_tokens"] == 10
    assert actual["mean_committed"] == pytest.approx(10 / 3)


def test_truncated_capture_tail_never_becomes_a_round():
    inventory, marks = trace_fixture()
    inventory["cpu_markers"] = marks[:-1]
    actual = windows(inventory)
    assert actual["tokens"] == [16386, 16392]
    assert actual["total_committed_tokens"] == 7


def test_missing_interior_final_consumer_is_rejected():
    inventory, marks = trace_fixture()
    inventory["cpu_markers"] = [row for row in marks if row[0] != 2180]
    with pytest.raises(ValueError, match="final-consume ownership"):
        windows(inventory)


def test_impossible_accepted_prefix_is_rejected():
    inventory, marks = trace_fixture()
    marks[7][2] = "v41::verify_and_commit::PP0::decode::P16400::C6::emit1"
    with pytest.raises(ValueError, match="committed speculative prefix"):
        windows(inventory)


def test_missing_complete_target_is_rejected():
    inventory, marks = trace_fixture()
    inventory["cpu_markers"] = [row for row in marks if row[0] != 2010]
    with pytest.raises(ValueError, match="complete target"):
        windows(inventory)


def test_rounds_require_native_coverage():
    inventory, _ = trace_fixture()
    with pytest.raises(ValueError, match="native target coverage"):
        speculative_windows(inventory, dict(tensor_parallel_size=4, pipeline_parallel_size=1), None)


def test_latency_report_normalizes_by_observed_commits(tmp_path, monkeypatch):
    import gzip
    import importlib
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    report = importlib.import_module("report_deepseek_v41_latency_ledger")
    contract = dict(phase="decode", unit="round", tokens=[10, 12], committed_tokens=[2, 3])
    for rank in range(4):
        root = tmp_path / "analysis" / f"rank{rank}"
        root.mkdir(parents=True)
        (root / "device-windows.json").write_text(json.dumps(contract))
        (root / "kernel-breakdown.json").write_text("{}")
        (root / "host-breakdown.json").write_text("{}")
        with gzip.open(root / "activity-intervals.json.gz", "wt") as stream:
            json.dump(
                dict(tokens=[10, 12],
                     windows_us=[[100, 1100], [1100, 2100]],
                     base_time_nanoseconds=0,
                     groups=[],
                     host_markers=[]), stream)
    report.report(tmp_path / "analysis", tmp_path / "report")
    actual = json.loads((tmp_path / "report" / "latency-ledger.json").read_text())
    assert actual["period_mean_ms"] == 1
    assert actual["total_committed_tokens"] == 5
    assert actual["trace_ms_per_committed_token"] == .4
    assert actual["formal_qualification"] is False
