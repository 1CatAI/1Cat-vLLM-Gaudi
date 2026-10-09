# SPDX-License-Identifier: Apache-2.0
"""Existing raw trace TPC activity by physical engine; no ISA utilization claim."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from report_deepseek_v41_raw_entry_periods import merge, length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    folder = args.analysis / f"rank{args.rank}"
    inventory = json.loads((folder / "inventory.json").read_text())
    ledger = json.loads((args.analysis / "RAW_ENTRY_PERIOD_LEDGER.json").read_text())
    low, high = ledger["periods"][0]["start_raw_us"], ledger["periods"][-1]["end_raw_us"]
    base = inventory["base_time_nanoseconds"] / 1000
    intervals = defaultdict(lambda: defaultdict(list))
    with gzip.open(folder / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            node = inventory["nodes"][event[3]]
            if node["engine"] != "TPC":
                continue
            a, b = base + event[0], base + event[0] + event[1]
            if b <= low or a >= high or b <= a:
                continue
            intervals[node["kernel"]][event[2]].append((max(a, low), min(b, high)))
    rows = []
    for kernel, cores in intervals.items():
        cores = {name: merge(spans) for name, spans in cores.items()}
        total = sum(length(spans) for spans in cores.values())
        union = length([span for spans in cores.values() for span in spans])
        rows.append(dict(kernel=kernel, activity_union_ms_per_period=union / ledger["rounds"],
                         mean_active_physical_tpcs=total / union,
                         physical_engine_count=len(cores),
                         per_engine_active_ms_per_period={name: length(spans) / ledger["rounds"]
                                                          for name, spans in sorted(cores.items())}))
    rows.sort(key=lambda row: -row["activity_union_ms_per_period"])
    result = dict(rank=args.rank, rounds=ledger["rounds"],
                  scope="Async host entry periods; kernel activity, not completion latency or ISA throughput",
                  hardware_event_resource_column=inventory["hardware_columns"], kernels=rows)
    out = args.analysis / f"TPC_SUPPLY_RANK{args.rank}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(out), hotspots=[{k: v for k, v in row.items()
                                                    if not k.startswith("per_engine")} for row in rows[:15]])))


if __name__ == "__main__":
    main()
