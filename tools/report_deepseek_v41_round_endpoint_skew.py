# SPDX-License-Identifier: Apache-2.0
"""Compare recorded operator completion proxies across the same physical rounds.

These are operator endpoints, not NIC arrivals or drained Target boundaries.
Only equal-count rounds are aligned by invocation order. Existing hardware
events are reused; this tool does not acquire a trace or infer missing traffic.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import statistics


def collect(arguments):
    analysis, rank, periods, kernels, gap = arguments
    directory = analysis / f"rank{rank}"
    inventory = json.loads((directory / "inventory.json").read_text())
    nodes = inventory["nodes"]
    base = inventory["base_time_nanoseconds"] / 1000
    spans = {kernel: [] for kernel in kernels}
    low, high = periods[0]["start_raw_us"], periods[-1]["end_raw_us"]
    with gzip.open(directory / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            node = nodes[event[3]]
            kernel = node["kernel"]
            if kernel not in spans:
                continue
            start = base + event[0]
            end = start + event[1]
            if low <= start < high and end > start:
                spans[kernel].append((start, end, node["recipe"]))
    result = {kernel: [] for kernel in kernels}
    for kernel, values in spans.items():
        values.sort()
        cursor = 0
        for period in periods:
            begin, end = period["start_raw_us"], period["end_raw_us"]
            merged = []
            while cursor < len(values) and values[cursor][0] < end:
                a, b, recipe = values[cursor]
                cursor += 1
                if a < begin:
                    continue
                if merged and a - merged[-1]["end_raw_us"] <= gap:
                    merged[-1]["end_raw_us"] = max(b, merged[-1]["end_raw_us"])
                    merged[-1]["recipes"].add(recipe)
                else:
                    merged.append(dict(start_raw_us=a, end_raw_us=b, recipes={recipe}))
            for row in merged:
                row["recipes"] = sorted(row["recipes"])
            result[kernel].append(merged)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-us", type=float, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing analysis; choose another output")
    ledger = json.loads((args.analysis / "RAW_DEVICE_CYCLE_LEDGER.json").read_text())
    periods = ledger["periods"]
    kernels = ("custom_deepseek_v41_expert_n256_scale_reduce_gaudi2",
               "custom_deepseek_v41_probability_draw_gaudi2")
    with ProcessPoolExecutor(max_workers=4) as pool:
        ranks = list(pool.map(collect, [(args.analysis, rank, periods, kernels, args.gap_us) for rank in range(4)]))
    categories = []
    for kernel in kernels:
        rounds, starts, ends, mismatches = [], [], [], []
        for index, period in enumerate(periods):
            actual = [rank[kernel][index] for rank in ranks]
            counts = [len(rows) for rows in actual]
            if not counts[0] or len(set(counts)) != 1:
                mismatches.append(dict(round=index, counts=counts))
                continue
            endpoints = []
            for invocation in range(counts[0]):
                first = [rows[invocation]["start_raw_us"] for rows in actual]
                last = [rows[invocation]["end_raw_us"] for rows in actual]
                start_skew, end_skew = max(first) - min(first), max(last) - min(last)
                starts.append(start_skew)
                ends.append(end_skew)
                endpoints.append(dict(invocation=invocation, start_skew_us=start_skew,
                                      end_skew_us=end_skew, rank_intervals=[rows[invocation] for rows in actual],
                                      elapsed_from_round_start_us=max(last) - period["start_raw_us"]))
            rounds.append(dict(round=index, count=counts[0], endpoints=endpoints))
        categories.append(dict(kernel=kernel, aligned_rounds=len(rounds), mismatched_rounds=mismatches,
                               mean_start_skew_us=statistics.mean(starts) if starts else None,
                               mean_end_skew_us=statistics.mean(ends) if ends else None,
                               max_end_skew_us=max(ends) if ends else None,
                               rounds=rounds))
    result = dict(source=str(args.analysis), method="Existing exact-symbol hardware intervals; gap-clustered endpoints",
                  gap_us=args.gap_us, completion_proxy=True, collective_arrival_measured=False,
                  critical_path=False, limit="Equal ordinal/count does not prove semantic identity or a NIC arrival; "
                  "no drained Target time, traffic or causal wait attribution inferred.", categories=categories)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "rounds"} for row in categories], indent=2))


if __name__ == "__main__":
    main()
