# SPDX-License-Identifier: Apache-2.0
"""Intersect explicit host phases with the same capture's global device gaps.

Nested ranges are made exclusive on the worker thread. Intersections identify
where the host was, not why a device was idle; waits can cover peer work.
"""
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

from account_deepseek_v41_trace import merge, subtract
from vllm_gaudi.ops.deepseek_v41_diagnostics import phase_fields


def intersections(spans, windows):
    ends = [b for _, b in windows]
    result = []
    for start, end in spans:
        index = bisect.bisect_right(ends, start)
        while index < len(windows) and windows[index][0] < end:
            low, high = windows[index]
            result.append((max(start, low), min(end, high)))
            index += 1
    return merge(result)


def exclusive_phases(records):
    """Return one thread's nested phases with direct children removed."""
    records = sorted(records, key=lambda row: (row["start_us"], -row["end_us"]))
    stack = []
    children = collections.defaultdict(list)
    for index, row in enumerate(records):
        while stack and records[stack[-1]]["end_us"] <= row["start_us"]:
            stack.pop()
        if stack:
            parent = records[stack[-1]]
            if row["end_us"] > parent["end_us"]:
                raise ValueError("Crossing phase scopes cannot be treated as a nested worker stack")
            children[stack[-1]].append((row["start_us"], row["end_us"]))
        stack.append(index)
    return [(row, subtract([(row["start_us"], row["end_us"])], merge(children[index])))
            for index, row in enumerate(records)]


def analyze(root):
    ledger = json.loads((root / "four-rank-disjoint-accounting.json").read_text())
    common = json.loads((root / "common-windows.json").read_text())
    windows, gaps = common["windows_us"], ledger["unattributed_intervals_us"]
    divisor = len(common["tokens"]) * 1000
    reports = []
    for rank in range(4):
        records = []
        with gzip.open(root / f"rank{rank}/host.jsonl.gz", "rt") as stream:
            for line in stream:
                start, duration, pid, tid, _, name = json.loads(line)
                if not name.startswith("dsv41_phase:"):
                    continue
                fields = phase_fields(name)
                records.append(dict(fields, start_us=start, end_us=start + duration, pid=pid, tid=tid))
        counts = collections.Counter(
            (row["pid"], row["tid"]) for row in records if row["phase"].endswith(".execute_model"))
        if not counts:
            raise ValueError(f"No explicit worker phase scopes for rank {rank}")
        worker_thread = counts.most_common(1)[0][0]
        worker = [row for row in records if (row["pid"], row["tid"]) == worker_thread]
        by_phase = collections.defaultdict(list)
        samples = collections.Counter()
        for row, spans in exclusive_phases(worker):
            by_phase[row["phase"]].extend(spans)
            if intersections([(row["start_us"], row["end_us"])], windows):
                samples[row["phase"]] += 1
        rows = []
        covered = merge([(row["start_us"], row["end_us"]) for row in worker])
        for name, spans in by_phase.items():
            spans = merge(spans)
            rows.append(
                dict(phase=name,
                     calls=samples[name],
                     exclusive_host_ms_per_token=sum(b - a for a, b in intersections(spans, windows)) / divisor,
                     global_gap_intersection_ms_per_token=sum(b - a for a, b in intersections(spans, gaps)) / divisor))
        outside = sum(b - a for a, b in subtract(gaps, covered)) / divisor
        gap_sum = sum(row["global_gap_intersection_ms_per_token"] for row in rows) + outside
        assert abs(gap_sum - ledger["groups"][-1]["exclusive_ms"]) < 1e-7
        rows.sort(key=lambda row: -row["global_gap_intersection_ms_per_token"])
        starts = {
            (row.get("request"), row.get("generation")): row
            for row in worker if row["phase"].endswith(".execute_model")
        }
        handoffs = []
        for row in worker:
            if not row["phase"].endswith("._sample_single") or row.get("generation") is None:
                continue
            following = starts.get((row.get("request"), row["generation"] + 1))
            if following is None or following["start_us"] < row["end_us"]:
                continue
            span = [(row["end_us"], following["start_us"])]
            selected = intersections(span, windows)
            if selected:
                handoffs.append(
                    dict(request=row["request"],
                         generation=row["generation"],
                         start_us=span[0][0],
                         end_us=span[0][1],
                         ms=sum(b - a for a, b in selected) / 1000,
                         global_gap_ms=sum(b - a for a, b in intersections(selected, gaps)) / 1000))
        reports.append(
            dict(rank=rank,
                 worker_thread=worker_thread,
                 phases=rows,
                 outside_marked_worker_gap_ms_per_token=outside,
                 unclassified_gap_reconciliation_ms=gap_sum,
                 inter_iteration_handoffs=handoffs,
                 scopes=records))
    result = dict(capture=str(root),
                  common_tokens=common["tokens"],
                  period_ms=ledger["period_ms"],
                  ranks=reports,
                  interpretation="Per-rank exclusive host location; four ranks overlap and must not be added.",
                  causal_attribution_complete=False)
    (root / "host-phase-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps([
            dict(rank=row["rank"], phases=row["phases"][:10], outside_ms=row["outside_marked_worker_gap_ms_per_token"])
            for row in reports
        ],
                   indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    analyze(parser.parse_args().analysis)
