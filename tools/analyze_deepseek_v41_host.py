# SPDX-License-Identifier: Apache-2.0
"""Measure host events in the same four-rank trace windows as device activity."""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

from analyze_deepseek_v41_trace import union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    common = json.loads((args.analysis / "common-windows.json").read_text())
    windows = common["windows_us"]
    ends = [w[1] for w in windows]
    scale = len(windows) * 1000
    for rank in range(4):
        groups = collections.defaultdict(list)
        events = []
        with gzip.open(args.analysis / f"rank{rank}/host.jsonl.gz", "rt") as stream:
            for line in stream:
                start, duration, pid, tid, category, name = json.loads(line)
                chosen = bisect.bisect_right(ends, start)
                if chosen >= len(windows):
                    continue
                lo, hi = windows[chosen]
                begin, end = max(start, lo), min(start + duration, hi)
                if end <= begin:
                    continue
                groups[(category, name)].append((begin, end))
                if "compileGraph" in name or "native_decoder_enqueue" in name:
                    events.append({"token": common["tokens"][chosen], "pid": pid, "tid": tid,
                                   "category": category, "name": name, "start_us": start,
                                   "duration_ms": duration / 1000})
        rows = [{"category": category, "name": name, "calls_per_token": len(spans) / len(windows),
                 "inclusive_sum_ms_per_token": sum(b-a for a, b in spans) / scale,
                 "activity_union_ms_per_token": union(spans) / scale}
                for (category, name), spans in groups.items()]
        rows.sort(key=lambda row: -row["inclusive_sum_ms_per_token"])
        result = {"rank": rank, "tokens": common["tokens"], "rows": rows, "events": events,
                  "semantics": "Inclusive CPU events may overlap across threads and nesting; do not add as overhead"}
        (args.analysis / f"rank{rank}/host-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"rank": rank, "selected": [r for r in rows if any(
            word in r["name"] for word in ("compileGraph", "native_decoder_enqueue", "syncEvent"))]}), flush=True)


if __name__ == "__main__":
    main()
