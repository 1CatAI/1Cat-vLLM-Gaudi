# SPDX-License-Identifier: Apache-2.0
"""Index disjoint raw engine activity for repeated trace-window queries."""
import bisect

class IntervalIndex:
    """Disjoint engine activity; clip a window without rescanning the capture."""
    def __init__(self, intervals):
        self.intervals = []
        for a, b in sorted(intervals):
            if self.intervals and a <= self.intervals[-1][1]:
                self.intervals[-1][1] = max(self.intervals[-1][1], b)
            else:
                self.intervals.append([a, b])
        self.starts = [a for a, _ in self.intervals]
        self.ends = [b for _, b in self.intervals]

    def clip(self, lo, hi):
        first = bisect.bisect_right(self.ends, lo)
        last = bisect.bisect_left(self.starts, hi)
        return [[max(a, lo), min(b, hi)] for a, b in self.intervals[first:last]]
