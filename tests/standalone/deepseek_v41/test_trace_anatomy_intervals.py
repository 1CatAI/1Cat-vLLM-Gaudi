# SPDX-License-Identifier: Apache-2.0
"""Compare indexed clipping with the original raw-interval union oracle."""
import importlib.util
import random
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "trace_interval_index", Path(__file__).resolve().parents[3] / "tools/trace_anatomy/interval_index.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def reference(intervals, lo, hi):
    result = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def test_overlap_touching_nested_and_exact_window_boundaries():
    intervals = [(0, 100), (2, 3), (4, 5), (100, 102), (103, 104), (105, 105)]
    index = _MODULE.IntervalIndex(intervals)
    for lo, hi in [(-1, 200), (100, 103), (102, 103), (103, 104), (0, 100), (104, 106)]:
        assert index.clip(lo, hi) == reference(intervals, lo, hi)
    assert _MODULE.IntervalIndex([]).clip(0, 1) == []


def test_random_windows_preserve_original_floating_point_union():
    rng = random.Random(42)
    for _ in range(100):
        intervals = [(a, a + rng.uniform(0, 50)) for a in [rng.uniform(-100, 100) for _ in range(100)]]
        index = _MODULE.IntervalIndex(intervals)
        for _ in range(100):
            lo = rng.uniform(-150, 150)
            hi = lo + rng.uniform(.01, 100)
            assert index.clip(lo, hi) == reference(intervals, lo, hi)
