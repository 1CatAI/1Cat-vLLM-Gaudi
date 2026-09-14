# SPDX-License-Identifier: Apache-2.0
"""Extract CPU synchronization candidates from a preserved profiler trace."""

import argparse
import gzip
import json
import statistics
from pathlib import Path

import ijson


def overlaps(ts, dur, spans):
    end = ts + dur
    return any(end > start and ts < stop for start, stop in spans)


def scan(path):
    opener = gzip.open if path.suffix == ".gz" else open
    phases = {"verify_and_commit": [], "draft_propose": [], "target_decode_C6": [], "target_decode_C1": []}
    candidates = []
    total = 0
    with opener(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            total += 1
            name = event.get("name", "")
            ts = event.get("ts")
            dur = event.get("dur", 0) or 0
            if ts is None or dur <= 0:
                continue
            if event.get("cat") == "user_annotation" and name.startswith("v41::"):
                if "verify_and_commit" in name:
                    phase = "verify_and_commit"
                elif "draft_propose" in name:
                    phase = "draft_propose"
                elif "target::" in name and "::decode::C6" in name:
                    phase = "target_decode_C6"
                elif "target::" in name and "::decode::C1" in name:
                    phase = "target_decode_C1"
                else:
                    phase = None
                if phase:
                    phases[phase].append((ts, ts + dur))
            if event.get("cat") != "cpu_op":
                continue
            if name not in {"aten::to", "aten::_to_copy", "aten::copy_", "aten::argmax", "aten::tolist", "aten::item"}:
                continue
            args = event.get("args") or {}
            candidates.append({
                "name": name,
                "ts": ts,
                "dur": dur,
                "input_dims": args.get("Input Dims", []),
                "input_type": args.get("Input type", []),
                "output_dims": args.get("Output Dims", []),
                "output_type": args.get("Output type", [])
            })
    rows = []
    for phase, spans in phases.items():
        selected = [row for row in candidates if overlaps(row["ts"], row["dur"], spans)]
        by_name = {}
        for row in selected:
            key = (row["name"], json.dumps(row["input_dims"],
                                           sort_keys=True), json.dumps(row["input_type"], sort_keys=True))
            by_name.setdefault(key, []).append(row)
        groups = []
        for (name, dims, types), values in by_name.items():
            groups.append({
                "kernel": name,
                "input_dims": json.loads(dims),
                "input_type": json.loads(types),
                "calls": len(values),
                "sum_ms": sum(v["dur"] for v in values) / 1000,
                "mean_ms": statistics.mean(v["dur"] for v in values) / 1000,
                "max_ms": max(v["dur"] for v in values) / 1000
            })
        groups.sort(key=lambda x: (-x["sum_ms"], x["kernel"], str(x["input_dims"])))
        rows.append({
            "phase": phase,
            "calls": len(spans),
            "annotation_ms": sum(b - a for a, b in spans) / 1000,
            "cpu_candidates": groups
        })
    return {"trace": str(path), "trace_events": total, "phases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "units": {
            "time": "ms",
            "raw_trace": "us"
        },
        "ranks": [scan(path) for path in args.traces]
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ranks": [{
                    "trace": r["trace"],
                    "phases": r["phases"]
                } for r in payload["ranks"]]
            },
            ensure_ascii=False,
            indent=2))


if __name__ == "__main__":
    main()
