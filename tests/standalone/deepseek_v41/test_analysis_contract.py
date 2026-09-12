# SPDX-License-Identifier: Apache-2.0
"""Guard false speed qualification and lost/duplicated overlap accounting."""

from pathlib import Path
import sys
import gzip
import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from analyze_deepseek_v41_resources import clip_windows, occupancy  # noqa: E402
from compare_deepseek_v41_candidate import REQUIRED_CHECKS, decide  # noqa: E402
import analyze_deepseek_v41_gaps as gaps_tool  # noqa: E402


def recorded_rounds(means):
    return {
        "profile":
        False,
        "qualified_measurement":
        True,
        "results": [{
            "error": None,
            "token_count": 192,
            "intervals": 181,
            "discarded_intervals": 10,
            "coalesced_token_events": 0,
            "timing_valid": True,
            "steady_ms": mean,
            "steady_intervals_ms": [mean] * 181
        } for mean in means]
    }


def compare(means, **checks):
    contracts = dict.fromkeys((*REQUIRED_CHECKS, "trace_mechanism"), True)
    contracts.update(checks)
    return decide(recorded_rounds([23.5] * 3), recorded_rounds(means), contracts, 15., .5, True)


def test_activity_crosses_tokens_without_filling_an_unrecorded_gap():
    windows = [(0, 10), (10, 20), (30, 40)]
    assert list(clip_windows(5, 35, windows, [10, 20, 40])) == [(0, 5, 10), (1, 10, 20), (2, 30, 35)]
    assert list(clip_windows(10, 10, windows, [10, 20, 40])) == []


def test_overlapping_packets_on_one_core_do_not_inflate_core_usage():
    value = occupancy({0: [(0, 6), (4, 8)], 1: [(5, 10)]}, [(0, 10)])
    assert value["busy_cores_mean_in_selected_window"] == pytest.approx(1.3)
    assert value["peak_busy_cores"] == 2
    assert occupancy({}, [])['peak_busy_cores'] is None


def test_target_is_strict_per_round_and_does_not_skip_quality():
    assert not compare([14.9, 15., 14.9])["all_rounds_below_target"]
    result = compare([14.9] * 3)
    assert result["decision"] == "start_full_quality_and_lifecycle_qualification"
    assert not result["production_qualified"]


def test_gain_threshold_and_missing_contracts_prevent_false_retention():
    assert compare([23.1] * 3)["decision"] == "archive_insufficient_gain"
    assert compare([23.] * 3)["decision"] == "retain_as_next_iteration_parent"
    assert compare([14.] * 3, numerics=False)["decision"] == "archive_contract_failure"
    assert compare([14.] * 3, trace_mechanism=None)["decision"] == "pending_contract_evidence"


def test_coalesced_events_cannot_qualify_fast_timing():
    candidate = recorded_rounds([14.] * 3)
    candidate["results"][0]["coalesced_token_events"] = 1
    with pytest.raises(ValueError, match="Invalid ITL"):
        decide(recorded_rounds([23.5] * 3), candidate, {}, 15., .5, True)


def test_gap_endpoints_preserve_work_without_inventing_nic_duration(tmp_path, monkeypatch):
    rank = tmp_path / "rank0"
    rank.mkdir()
    inventory = {"nodes": [{"recipe": "r", "engine": kind} for kind in ("TPC", "DMA", "NIC")]}
    (rank / "inventory.json").write_text(json.dumps(inventory))
    (rank / "recipe-symbols.json").write_text(json.dumps({"recipes": {}}))
    (rank / "node-breakdown.json").write_text(
        json.dumps([{
            "recipe_id": "r",
            "engine": "DMA",
            "context_id": 1,
            "purpose": "transfer",
            "compiler_contract": None,
            "inputs": [],
            "outputs": []
        }]))
    monkeypatch.setattr(gaps_tool, "symbols", lambda *_: {})
    with gzip.open(rank / "hardware.jsonl.gz", "wt") as stream:
        for event in [(1, 4, 0, 0), (6, 2, 0, 1), (9, 0, 0, 2), (50, 2, 0, 0)]:
            stream.write(json.dumps(event) + "\n")
    pairs = gaps_tool.boundaries(tmp_path, 0, [(10, 20), (30, 40), (60, 70)])
    assert [a["end_us"] for a, _ in pairs] == [8, 8, 52]
    assert [b["start_us"] if b else None for _, b in pairs] == [50, 50, None]
    assert pairs[0][0]["purpose"] == "transfer"
    assert pairs[0][0]["execution_index"] is None
    with pytest.raises(ValueError, match="ordered"):
        gaps_tool.boundaries(tmp_path, 0, [(30, 40), (10, 20)])
