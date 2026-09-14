# SPDX-License-Identifier: Apache-2.0
"""Correlate preserved Python scopes with gaps in the four-rank device ledger.

Scope overlap identifies where a host thread was, not why hardware was idle.
Parent and child scopes overlap; their times must never be added as costs.
"""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path

import ijson

from account_deepseek_v41_trace import length, merge, subtract
from analyze_deepseek_v41_resources import clip_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.trace_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((root / f"rank{args.rank}/inventory.json").read_text())
    windows = json.loads((root / "common-windows.json").read_text())["windows_us"]
    ledger = json.loads((root / "four-rank-disjoint-accounting.json").read_text())
    gaps = ledger["unattributed_intervals_us"]
    ends = [b for a, b in windows]
    cache = out / f"rank{args.rank}-python-scopes.jsonl.gz"
    identity = {"trace_sha256": inventory["trace_sha256"], "windows_us": windows}
    identity_file = out / f"rank{args.rank}-identity.json"
    if cache.exists():
        if not identity_file.exists() or json.loads(identity_file.read_text()) != identity:
            raise ValueError("Cached scopes do not match the selected capture and windows")
    else:
        source = Path(inventory["trace"])
        opener = gzip.open if source.suffix == ".gz" else open
        temporary = cache.with_suffix(".partial")
        with opener(source, "rb") as stream, gzip.open(temporary, "wt", compresslevel=1) as dst:
            for event in ijson.items(stream, "traceEvents.item", use_float=True):
                if event.get("cat") != "python_function" or event.get("ph") != "X":
                    continue
                start, duration = event["ts"], event.get("dur", 0)
                if duration <= 0 or start >= windows[-1][1] or start + duration <= windows[0][0]:
                    continue
                detail = event.get("args", {})
                row = [
                    start, duration, event["tid"], event["name"],
                    detail.get("Python id"),
                    detail.get("Python parent id")
                ]
                dst.write(json.dumps(row, separators=(",", ":")) + "\n")
        temporary.rename(cache)
        identity_file.write_text(json.dumps(identity, indent=2) + "\n")
    spans = collections.defaultdict(list)
    calls = collections.Counter()
    with gzip.open(cache, "rt") as stream:
        for line in stream:
            start, duration, tid, name, _, _ = json.loads(line)
            name = re.sub(r" at 0x[0-9a-f]+", "", name)
            key = (tid, name)
            overlaps = list(clip_windows(start, start + duration, windows, ends))
            if overlaps:
                calls[key] += 1
                spans[key].extend((a, b) for _, a, b in overlaps)
    denominator = len(windows) * 1000
    rows = []
    for (tid, name), intervals in spans.items():
        union = merge(intervals)
        overlap = length(union) - length(subtract(union, gaps))
        rows.append({
            "tid": tid,
            "function": name,
            "calls": calls[tid, name],
            "calls_per_token": calls[tid, name] / len(windows),
            "scope_union_ms_per_token": length(union) / denominator,
            "scope_intersection_with_global_gap_ms_per_token": overlap / denominator
        })
    rows.sort(key=lambda row: row["scope_intersection_with_global_gap_ms_per_token"], reverse=True)
    result = {
        "rank":
        args.rank,
        "periods":
        len(windows),
        "capture":
        identity,
        "global_gap_ms_per_token":
        length(gaps) / denominator,
        "functions":
        rows,
        "limitations": [
            "Nested scopes and different threads overlap; rows are not additive.",
            "A blocked host thread can coexist with useful device activity.",
            "Even a scope inside a device gap does not prove causal ownership or removable time."
        ]
    }
    (out / f"rank{args.rank}-host-scopes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows[:25], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
