# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(SOURCE))
try:
    spec = importlib.util.spec_from_file_location("batch_trace", SOURCE / "analyze_deepseek_v41_batch_trace.py")
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
finally:
    sys.path.pop(0)


def markers(batch, entries):
    return [(100, 200, f"v41::request_batch::PP1::generation42::requests{batch}")
            ] + [(start, 1, "vllm_gaudi::native_decoder_enqueue") for start in entries]


def test_two_lane_requires_both_entries_and_preserves_cross_rank_clock():
    rows = markers(64, [120, 180]) + [(150, 4, "Torch-Compiled Region: sampler")]
    assert trace.native_windows(rows, 1000, 2) == {42: (1100, 1300, 64)}
    assert not trace.native_windows(rows, 0, 1)
    assert not trace.native_windows(markers(64, [120]), 0, 2)


def test_small_batch_stays_single_entry_with_two_lane_policy():
    assert trace.native_windows(markers(8, [120]), 0, 2) == {42: (100, 300, 8)}
    assert not trace.native_windows(markers(8, [120, 180]), 0, 2)


def test_capture_and_out_of_window_entries_cannot_qualify():
    rows = markers(32, [120, 180]) + [(110, 4, "Torch-Compiled Region: decoder")]
    assert not trace.native_windows(rows, 0, 2)
    assert not trace.native_windows(markers(32, [120, 300]), 0, 2)


def test_pipeline_overlap_is_exclusive_not_added_twice():
    a, b = trace.GROUPS[:2]
    ledger = trace.partition({a: [(0, 60)], b: [(40, 90)]}, 0, 100)
    assert ledger[a] == 40 and ledger[b] == 30
    assert ledger[trace.GROUPS[-2]] == 20
    assert ledger[trace.GROUPS[-1]] == 10
    assert sum(ledger.values()) == 100
