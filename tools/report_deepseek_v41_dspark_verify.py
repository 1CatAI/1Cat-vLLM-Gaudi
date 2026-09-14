# SPDX-License-Identifier: Apache-2.0
"""Report DSpark draft/target/verify cost from a preserved HPU trace.

The hardware export has one record per lane/monitor.  This reporter keeps
packet counts for auditability but uses a timestamp union for activity, so
parallel lanes are not incorrectly added together.  CPU phase annotations are
used only as attribution windows; they are not treated as device time.
"""

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


MEASURED_KIND = {"TPC": {"TPC_SPU_START_TO_SPU_HALT"},
                 "MME": {"MMEH_WB0_MON_TS_BIT1", "MMEH_WB1_MON_TS_BIT1"},
                 # The DMA write-last event is the complete DMA interval in
                 # this export.  The read-first event is a start marker.
                 "DMA": {"DBG_DMA_TRC_WR_DATA_LAST"}}


class UnionAccumulator:
    __slots__ = ("last_start", "last_end", "activity", "segments", "packets",
                 "raw_us", "out_of_order")

    def __init__(self):
        self.last_start = None
        self.last_end = None
        self.activity = 0.0
        self.segments = 0
        self.packets = 0
        self.raw_us = 0.0
        self.out_of_order = 0

    def add(self, start, end):
        if end <= start:
            return
        self.packets += 1
        self.raw_us += end - start
        if self.last_end is not None and start < self.last_start:
            self.out_of_order += 1
        if self.last_end is None or start > self.last_end:
            self.activity += end - start
            self.segments += 1
        elif end > self.last_end:
            self.activity += end - self.last_end
        if self.last_end is None or end > self.last_end:
            self.last_end = end
        if self.last_start is None or start < self.last_start:
            self.last_start = start

    def json(self):
        return {"activity_ms": self.activity / 1000.0,
                "raw_packet_duration_ms": self.raw_us / 1000.0,
                "segments": self.segments,
                "packets": self.packets,
                "out_of_order_packets": self.out_of_order}


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def span_duration(spans):
    return sum(end - start for start, end in merged(spans))


def read_phases(path):
    phases = collections.defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            ts, dur, _pid, _tid, cat, name = json.loads(line)
            if cat != "user_annotation" or not name.startswith("v41::"):
                continue
            if dur <= 0:
                continue
            if "verify_and_commit" in name:
                phase = "verify_and_commit"
            elif "draft_propose" in name:
                phase = "draft_propose"
            elif "target::" in name and "::decode::" in name:
                phase = name.split("::decode::", 1)[1]
                phase = "target_decode_" + phase
            elif "target::" in name and "::prefill::" in name:
                phase = name.split("::prefill::", 1)[1]
                phase = "target_prefill_" + phase
            else:
                phase = name
            phases[phase].append((ts, ts + dur, name))
    return phases


def phase_overlaps(ts, end, intervals):
    for start, stop, _name in intervals:
        if end <= start or ts >= stop:
            continue
        yield max(ts, start), min(end, stop)


def process_rank(root, rank):
    path = root / f"rank{rank}"
    inventory = json.loads((path / "inventory.json").read_text())
    phases = read_phases(path / "host.jsonl.gz")
    phase_all = [(a, b, n) for values in phases.values() for a, b, n in values]
    phase_order = ["draft_propose", "target_decode_C6", "target_decode_C1",
                   "verify_and_commit"]
    phases = {name: phases.get(name, []) for name in phase_order}
    kinds = inventory["hw_event_names"]
    nodes = inventory["nodes"]
    # Per phase and kernel union.  Kernel names are kept separate from the
    # EventName/source node to match the user's kernel-level accounting.
    by_phase = {phase: {} for phase in phases}
    by_engine = {phase: collections.defaultdict(UnionAccumulator) for phase in phases}
    by_phase_total = {phase: UnionAccumulator() for phase in phases}
    by_kernel = {phase: collections.defaultdict(UnionAccumulator) for phase in phases}
    all_engine = collections.defaultdict(UnionAccumulator)
    event_count = 0
    measured_count = 0
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            ts, dur, lane, node_index, kind_index = json.loads(line)
            event_count += 1
            kind = kinds[kind_index]
            node = nodes[node_index]
            engine = node["engine"]
            if kind not in MEASURED_KIND.get(engine, set()):
                continue
            measured_count += 1
            end = ts + dur
            kernel = node["kernel"]
            all_engine[engine].add(ts, end)
            for phase, intervals in phases.items():
                for start, stop in phase_overlaps(ts, end, intervals):
                    by_phase_total[phase].add(start, stop)
                    by_engine[phase][engine].add(start, stop)
                    by_kernel[phase][(engine, kernel, node.get("reported_dtype", ""))].add(start, stop)
    phase_rows = []
    for phase, intervals in phases.items():
        cpu_spans = [(a, b) for a, b, _ in intervals]
        cpu_us = span_duration(cpu_spans)
        engines = {engine: acc.json() for engine, acc in by_engine[phase].items()}
        rows = []
        for (engine, kernel, dtype), acc in by_kernel[phase].items():
            row = {"engine": engine, "kernel": kernel, "reported_dtype": dtype,
                   **acc.json()}
            row["cpu_phase_share_pct"] = (row["activity_ms"] * 100000.0 / cpu_us
                                           if cpu_us else None)
            row["device_union_share_pct"] = (row["activity_ms"] * 100000.0 /
                                              by_phase_total[phase].activity
                                              if by_phase_total[phase].activity else None)
            rows.append(row)
        rows.sort(key=lambda row: (-row["activity_ms"], row["engine"], row["kernel"]))
        phase_rows.append({"phase": phase, "calls": len(intervals),
                           "cpu_annotation_ms": cpu_us / 1000.0,
                           "cpu_mean_call_ms": statistics.mean((b - a) / 1000 for a, b in cpu_spans)
                           if cpu_spans else None,
                           "device_activity_union_ms": by_phase_total[phase].activity / 1000.0,
                           "device_activity_raw_packet_ms": by_phase_total[phase].raw_us / 1000.0,
                           "device_activity_by_engine": engines,
                           "kernel_rows": rows})
    return {"rank": rank, "pp": rank // 2, "tp": rank % 2,
            "trace": inventory["trace"], "trace_sha256": inventory["trace_sha256"],
            "trace_events": inventory["events"], "hardware_records": event_count,
            "measured_records": measured_count,
            "window_ms": (inventory["last_us"] - inventory["first_us"]) / 1000.0,
            "phase_rows": phase_rows,
            "device_activity_by_engine_window": {k: v.json() for k, v in all_engine.items()},
            "phase_note": "CPU annotations delimit attribution; hardware rows are timestamp-unioned per engine/kernel."
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    results = [process_rank(args.analysis, rank) for rank in range(4)]
    payload = {"schema_version": 1, "units": {"time": "ms", "raw_trace": "us"},
               "measured_hw_kinds": {k: sorted(v) for k, v in MEASURED_KIND.items()},
               "ranks": results}
    out = args.analysis / "dspark-verify-breakdown.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "ranks": [
        {"rank": r["rank"], "window_ms": r["window_ms"],
         "phases": [{"phase": p["phase"], "calls": p["calls"],
                      "cpu_annotation_ms": p["cpu_annotation_ms"],
                      "device": p["device_activity_by_engine"]}
                     for p in r["phase_rows"]]}
        for r in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
