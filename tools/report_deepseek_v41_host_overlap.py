# SPDX-License-Identifier: Apache-2.0
"""Join native Engram row-copy clocks to the same acquisition's device events."""

import argparse
from bisect import bisect_left
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from report_deepseek_v41_acquisition import coverage_qualification, duration, merged, timestamp_marker


def intersections(span, intervals, ends):
    start, end = span
    result = []
    for index in range(bisect_left(ends, start), len(intervals)):
        a, b = intervals[index]
        if a >= end:
            break
        if min(end, b) > max(start, a):
            result.append((max(start, a), min(end, b)))
    return result


def analyze(analysis, capture, rank):
    directory = analysis / f"rank{rank}"
    inventory = json.loads((directory / "inventory.json").read_text())
    source = capture / f"rank{rank}-engram-profile.json"
    profile = json.loads(source.read_text())
    base = inventory["base_time_nanoseconds"]
    if base is None or profile["tp_rank"] != rank or profile["units"] != "ns":
        raise ValueError("Missing clock origin or mismatched Engram owner")
    counter = lambda phase: json.loads((capture / f"rank{rank}-native-profile-{phase}.json").read_text())["engram"]
    before, after = counter("start"), counter("stop")
    if len(profile["records"]) != after["gathers"] - before["gathers"]:
        raise ValueError("Native Engram timing records do not cover the acquisition counters")
    device = {"TPC": [], "MME": [], "DMA": []}
    with gzip.open(directory / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            ts, dur, _, index, kind = json.loads(line)
            engine = inventory["nodes"][index]["engine"]
            if engine in device and not timestamp_marker(inventory["hw_event_names"][kind]):
                device[engine].append((ts, ts + dur))
    device["TPC_or_MME"] = merged(device["TPC"] + device["MME"])
    device = {key: merged(spans) for key, spans in device.items()}
    ends = {key: [b for _, b in spans] for key, spans in device.items()}
    phases = []
    with gzip.open(directory / "host.jsonl.gz", "rt") as stream:
        for line in stream:
            ts, dur, _, _, _, name = json.loads(line)
            if name.startswith("v41::target::PP0::"):
                phases.append((ts, ts + dur, name))
    rows, host_spans, overlap = [], [], []
    for record in profile["records"]:
        a, b = (record[key] - base for key in ("start_unix_ns", "end_unix_ns"))
        if b < a:
            raise ValueError("Native Engram clock ran backwards")
        span = (a / 1000, b / 1000)
        host_spans.append(span)
        matched = {key: intersections(span, spans, ends[key]) for key, spans in device.items()}
        overlap.extend(matched["TPC_or_MME"])
        enclosing = [name for start, end, name in phases if start <= span[0] <= span[1] <= end]
        rows.append({**record, "row_copy_ms": (b - a) / 1e6, "enclosing_target_phases": enclosing,
                     "overlap_ms": {key: duration(spans) / 1000 for key, spans in matched.items()}})
    summary = []
    for layer in (1, 14):
        selected = [row for row in rows if row["layer"] == layer]
        times = sorted(row["row_copy_ms"] for row in selected)
        summary.append({"layer": layer, "calls": len(selected), "mean_row_copy_ms": statistics.mean(times),
                        "p50_row_copy_ms": statistics.median(times), "max_row_copy_ms": max(times),
                        "sum_row_copy_ms": sum(times),
                        "calls_overlapping_recorded_compute": sum(row["overlap_ms"]["TPC_or_MME"] > 0
                                                                  for row in selected),
                        "sum_recorded_compute_overlap_ms": sum(row["overlap_ms"]["TPC_or_MME"] for row in selected),
                        "major_faults": sum(row["major_faults"] for row in selected)})
    return {"rank": rank, "pid": profile["pid"], "trace_sha256": inventory["trace_sha256"],
            "host_profile_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "layers": summary,
            "row_copy_union_ms": duration(host_spans) / 1000,
            "recorded_compute_overlap_union_ms": duration(overlap) / 1000, "records": rows,
            "host_clock_containment_check": {"target_annotations": len(phases),
                "gathers_inside_exactly_one_target_phase": sum(len(row["enclosing_target_phases"]) == 1
                                                               for row in rows), "gathers": len(rows)},
            "clock_join": "native CLOCK_REALTIME nanoseconds minus profiler baseTimeNanoseconds",
            "limitations": ["TPC/MME overlap is measured concurrency, not an end-to-end saving.",
                            "Hardware timestamps use profiler host-clock calibration; residual error is unknown.",
                            "Row-copy time excludes thread queue delay, staging copies and HPU DMA.",
                            "Compute includes all recorded TPC/MME, including vision and capture where present."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    rows = [analyze(args.analysis, args.capture, rank) for rank in (0, 1)]
    coverage, qualification = coverage_qualification(args.analysis)
    result = {"report_units": "ms", "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "hardware_coverage_status": coverage["status"], "coverage_qualification": qualification,
              "ranks": rows}
    (args.analysis / "engram-host-overlap.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
