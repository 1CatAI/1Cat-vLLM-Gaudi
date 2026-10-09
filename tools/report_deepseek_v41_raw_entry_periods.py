# SPDX-License-Identifier: Apache-2.0
"""Coarse hardware accounting between raw-clock host round submission starts.

These are asynchronous submission periods, not completion/critical-path
windows. Keep unknown cached native symbols explicit; never fit unrelated
perf_counter timestamps or infer kernel names from neighboring recipes.
"""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import re
import statistics


def merge(values):
    result = []
    for a, b in sorted(values):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def length(values):
    return sum(b - a for a, b in merge(values)) / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    inventories = [json.loads((args.analysis / f"rank{r}/inventory.json").read_text()) for r in range(4)]
    pattern = re.compile(r"v41::verify_and_commit::PP0::decode::P(\d+)::C6::emit1")
    markers = []
    for inv in inventories:
        base = inv["base_time_nanoseconds"] / 1000
        rows = {}
        for start, duration, name in inv["cpu_markers"]:
            if match := pattern.fullmatch(name):
                position = int(match[1])
                if position in rows:
                    raise ValueError("Duplicate rank/position scope")
                rows[position] = [base + start, duration]
        markers.append(rows)
    positions = sorted(set.intersection(*(set(rows) for rows in markers)))
    starts = [markers[0][p][0] for p in positions]
    periods = [
        dict(
            position=a,
            next_position=b,
            start_raw_us=x,
            end_raw_us=y,
            submitted_tokens=b - a,
            period_ms=(y - x) / 1000,
            host_entry_skew_us=max(m[a][0] for m in markers) - min(m[a][0] for m in markers),
        )
        for a, b, x, y in zip(positions, positions[1:], starts, starts[1:])
    ]
    if len(periods) < 3:
        raise ValueError("Need at least three matched four-rank entry periods")
    groups = [[collections.defaultdict(list) for _ in periods] for _ in range(4)]
    kernels = [collections.defaultdict(lambda: [0, 0.0]) for _ in range(4)]
    for rank, inv in enumerate(inventories):
        base = inv["base_time_nanoseconds"] / 1000
        nodes = inv["nodes"]
        with gzip.open(args.analysis / f"rank{rank}/hardware.jsonl.gz", "rt") as stream:
            for line in stream:
                event = json.loads(line)
                a, b = base + event[0], base + event[0] + event[1]
                if b <= starts[0] or a >= starts[-1] or b <= a:
                    continue
                node = nodes[event[3]]
                engine = node.get("engine", "UNKNOWN")
                kernel = node.get("kernel", "unresolved")
                i = max(0, bisect.bisect_right(starts, a) - 1)
                while i < len(periods) and a < periods[i]["end_raw_us"]:
                    low, high = max(a, starts[i]), min(b, starts[i + 1])
                    if high > low:
                        groups[rank][i][engine].append((low, high))
                    i += 1
                kernels[rank][(engine, kernel)][0] += 1
                kernels[rank][(engine, kernel)][1] += (min(b, starts[-1]) - max(a, starts[0])) / 1000
        print(f"rank {rank}: hardware parsed", flush=True)
    for i, period in enumerate(periods):
        per_rank, all_cards = [], []
        for rank in range(4):
            engines = groups[rank][i]
            active = [span for values in engines.values() for span in values]
            compute = [span for name, values in engines.items() if name in ("TPC", "MME") for span in values]
            per_rank.append(
                dict(
                    rank=rank,
                    engine_union_ms={k: length(v) for k, v in engines.items()},
                    any_device_activity_ms=length(active),
                    compute_union_ms=length(compute),
                    no_recorded_device_activity_ms=period["period_ms"] - length(active),
                )
            )
            all_cards.extend(active)
        period["ranks"] = per_rank
        period["four_cards_no_recorded_activity_ms"] = period["period_ms"] - length(all_cards)
    result = dict(
        method="consecutive matched rank0 raw-clock host entry starts",
        asynchronous_submission_periods=True,
        completion_windows=False,
        critical_path=False,
        collective_arrivals_measured=False,
        cached_native_symbols_incomplete=True,
        rounds=len(periods),
        mean_period_ms=statistics.mean(p["period_ms"] for p in periods),
        mean_four_cards_no_recorded_activity_ms=statistics.mean(
            p["four_cards_no_recorded_activity_ms"] for p in periods
        ),
        host_entry_skew_us=[p["host_entry_skew_us"] for p in periods],
        periods=periods,
        kernel_event_duration_sums_not_exclusive=[
            [
                dict(engine=e, kernel=k, calls=c, event_duration_sum_ms=t)
                for (e, k), (c, t) in sorted(table.items(), key=lambda item: -item[1][1])[:30]
            ]
            for table in kernels
        ],
    )
    (args.analysis / "RAW_ENTRY_PERIOD_LEDGER.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("periods", "kernel_event_duration_sums_not_exclusive")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
