# SPDX-License-Identifier: Apache-2.0
"""Offline accounting must preserve time, align clocks and reject decode data."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("prefill_ledger", TOOLS / "report_deepseek_v41_prefill_ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def fixture(root):
    for rank in range(4):
        path = root / f"rank{rank}"
        path.mkdir(parents=True)
        shift = rank * 1000
        windows = [[-shift, 10000 - shift]]
        groups = []
        if rank == 0:
            groups = [dict(engine="MME", category="Attention", intervals_us=[[0, 3000]])]
        if rank == 1:
            groups = [dict(engine="TPC", category="路由专家", intervals_us=[[2000 - shift, 5000 - shift]])]
        markers = [[0, 6000, "v41::prefill::attention::layer0::C16384"],
                   [6000, 2000, "v41::prefill::moe::layer0::C16384"]] if rank == 0 else []
        with gzip.open(path / "activity-intervals.json.gz", "wt") as stream:
            json.dump(
                dict(rank=rank,
                     base_time_nanoseconds=shift * 1000,
                     windows_us=windows,
                     tokens=[0],
                     trace_sha256=f"fixture-rank{rank}",
                     groups=groups,
                     host_markers=markers), stream)
        (path / "device-windows.json").write_text(
            json.dumps(
                dict(phase="prefill", unit="request", windows_us=windows, coverage_proof=dict(prompt_tokens=16384))))
        (path / "host-breakdown.json").write_text(
            json.dumps(dict(activity_intervals_us={"observed_synchronization_or_wait": [[-shift, 10000 - shift]]})))


def test_prefill_clock_alignment_overlap_and_request_units(tmp_path):
    root, output = tmp_path / "input", tmp_path / "output"
    fixture(root)
    ledger.report(root, output)
    result = json.loads((output / "ledger.json").read_text())
    assert result["period_ms"] == 10
    assert result["rank0_no_activity_other_rank_active_ms"] == 2
    rows = {row["label"]: row["ms"] for row in result["additive_rows"]}
    assert sum(rows.values()) == 10
    assert rows["设备kernel：跨模块重叠"] == 1
    assert rows["四卡无可见设备活动：" + ledger.STAGES[0]] == 1
    assert rows["四卡无可见设备活动：" + ledger.STAGES[1]] == 2
    assert rows["四卡无可见设备活动：" + ledger.FALLBACK] == 2
    host = result["observed_host_activity_by_rank"][1]["observed_synchronization_or_wait"]
    assert host == dict(activity_ms=10, overlap_with_compute_ms=3, outside_compute_ms=7)
    assert "ms/request" in (output / "REPORT.md").read_text()


def test_prefill_rejects_decode_window(tmp_path):
    root = tmp_path / "input"
    fixture(root)
    path = root / "rank2/device-windows.json"
    data = json.loads(path.read_text())
    data.update(phase="decode", unit="token")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="matching prefill"):
        ledger.report(root, tmp_path / "output")


def test_prefill_32k_preserves_request_units(tmp_path):
    root, output = tmp_path / "input", tmp_path / "output"
    fixture(root)
    for rank in range(4):
        path = root / f"rank{rank}/device-windows.json"
        data = json.loads(path.read_text())
        data["coverage_proof"]["prompt_tokens"] = 32768
        path.write_text(json.dumps(data))
    ledger.report(root, output, expected_prompt_tokens=32768)
    result = json.loads((output / "ledger.json").read_text())
    assert result["prompt_tokens"] == 32768
    assert result["period_ms"] == sum(row["ms"] for row in result["additive_rows"]) == 10
    assert "32768 token" in (output / "REPORT.md").read_text()


def test_prefill_rejects_inconsistent_rank_prompt(tmp_path):
    root = tmp_path / "input"
    fixture(root)
    path = root / "rank2/device-windows.json"
    data = json.loads(path.read_text())
    data["coverage_proof"]["prompt_tokens"] = 32768
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="same complete prompt"):
        ledger.report(root, tmp_path / "output")
