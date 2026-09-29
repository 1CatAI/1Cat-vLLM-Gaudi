# SPDX-License-Identifier: Apache-2.0
"""Join request generations across PP/TP without assuming a C1 layer pattern.

Exclusive accounting assigns intersections of different functional groups to
an explicit overlap row. It is an activity ledger, not a critical-path saving
estimate. Host wait envelopes and NIC point activity are reported separately.
"""
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import re
import statistics

from analyze_deepseek_v41_trace import symbols, union
from report_deepseek_v41_trace import classify, invocation_samples

GROUPS = (
    "路由专家",
    "Attention/CSA2",
    "mHC",
    "Router/共享专家",
    "Engram",
    "输入/输出头",
    "其他/未映射设备活动",
    "跨功能组重叠",
    "无已记录设备活动",
)


def group_for(symbol):
    if symbol is None:
        return GROUPS[6]
    node, kernel = symbol["node"], symbol["kernel"]
    if "deepseek_v41_index" in kernel or "deepseek_v41_state_rows" in kernel or "selected_mla" in node:
        return GROUPS[1]
    group, _ = classify(node, kernel, [], [])
    return {
        "路由专家": GROUPS[0],
        "Attention": GROUPS[1],
        "CSA2": GROUPS[1],
        "mHC": GROUPS[2],
        "Router": GROUPS[3],
        "共享专家": GROUPS[3],
        "Engram": GROUPS[4],
        "输出头": GROUPS[5],
        "输入/尾部": GROUPS[5]
    }.get(group, GROUPS[6])


def partition(groups, start, end):
    points = [(start, None, 0), (end, None, 0)]
    for name, spans in groups.items():
        for a, b in spans:
            a, b = max(a, start), min(b, end)
            if b > a:
                points.extend(((a, name, 1), (b, name, -1)))
    points.sort(key=lambda row: row[0])
    active = collections.Counter()
    result = dict.fromkeys(GROUPS, 0.0)
    previous = start
    for at, name, change in points:
        if at > previous:
            live = [key for key, value in active.items() if value]
            key = GROUPS[-1] if not live else live[0] if len(live) == 1 else GROUPS[-2]
            result[key] += at - previous
        if name is not None:
            active[name] += change
        previous = at
    if abs(sum(result.values()) - (end - start)) > 1e-3:
        raise ValueError("Incomplete exclusive batch ledger")
    return result


def native_windows(markers, shift, microbatches):
    """Select complete native transactions using the recorded run's policy."""
    windows = {}
    for start, duration, name in markers:
        match = re.fullmatch(r"v41::request_batch::PP([01])::generation(\d+)::requests(\d+)", name)
        if not match or duration <= 0:
            continue
        generation, batch = int(match[2]), int(match[3])
        if generation in windows:
            raise ValueError("Duplicate rank/generation")
        entries = sorted(m[0] for m in markers
                         if m[2] == "vllm_gaudi::native_decoder_enqueue" and start <= m[0] < start + duration)
        expected = 2 if microbatches == 2 and 32 <= batch <= 64 else 1
        # PP1 sampling between native entries is expected. Decoder discovery
        # before its first entry, or a missing lane, cannot qualify as steady.
        discovery = any("Compiled Region" in m[2] and start <= m[0] < entries[0] for m in markers) if entries else False
        if len(entries) == expected and not discovery:
            windows[generation] = (start + shift, start + duration + shift, batch)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--microbatches",
                        type=int,
                        choices=(1, 2),
                        default=1,
                        help="Exact PP policy from the acquisition's launch manifest")
    args = parser.parse_args()
    root = args.analysis
    inventories = [json.loads((root / f"rank{rank}/inventory.json").read_text()) for rank in range(4)]
    if any(inv["base_time_nanoseconds"] is None for inv in inventories):
        raise ValueError("Cross-rank clock origins are unavailable")
    base = min(inv["base_time_nanoseconds"] for inv in inventories)
    shifts = [(inv["base_time_nanoseconds"] - base) / 1000 for inv in inventories]
    rank_windows = []
    mapped = []
    for rank, (inv, shift) in enumerate(zip(inventories, shifts)):
        windows = native_windows(inv["cpu_markers"], shift, args.microbatches)
        rank_windows.append(windows)
        recipes = json.loads((root / f"rank{rank}/recipe-symbols.json").read_text())["recipes"]
        mapped.append(symbols(inv, recipes))
    generations = sorted(set.intersection(*(set(w) for w in rank_windows)))
    periods = []
    for generation in generations:
        stages = [w[generation] for w in rank_windows]
        if len({w[2] for w in stages}) != 1:
            raise ValueError("Ranks disagree on the real request count")
        # Different workers can finish host bookkeeping after PP0 has begun
        # preparing the next generation. Use consecutive PP0/TP0 entry
        # periods, rather than summing overlapping worker host envelopes.
        following = rank_windows[0].get(generation + 1)
        if following is None:
            continue
        periods.append(
            dict(generation=generation,
                 batch=stages[0][2],
                 start=stages[0][0],
                 end=following[0],
                 rank_bounds=[w[:2] for w in stages]))
    if not periods:
        raise ValueError("No complete four-rank native request transactions")
    if any(a["end"] > b["start"] for a, b in zip(periods, periods[1:])):
        raise ValueError("Batch transactions overlap; pipeline-specific windows required")
    starts = [p["start"] for p in periods]
    grouped = [collections.defaultdict(list) for _ in periods]
    nodes = collections.defaultdict(list)
    resources = collections.defaultdict(list)
    counters = collections.Counter()
    for rank, (inv, shift) in enumerate(zip(inventories, shifts)):
        with gzip.open(root / f"rank{rank}/hardware.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                start, end = row[0] + shift, row[0] + row[1] + shift
                i = bisect.bisect_right(starts, start) - 1
                if i < 0 or end > periods[i]["end"] or start < periods[i]["start"]:
                    continue
                node = inv["nodes"][row[3]]
                symbol = mapped[rank].get(row[3])
                group = group_for(symbol)
                grouped[i][group].append((start, end))
                nodes[rank, row[3], i].append((start, end, row[2]))
                resources[rank, node["engine"], i].append((start, end))
                counters[rank, "mapped" if symbol else "unmapped"] += 1
    details = []
    for (rank, index, period), rows in nodes.items():
        node = inventories[rank]["nodes"][index]
        symbol = mapped[rank].get(index)
        samples, count, reason = invocation_samples(rows, symbol, None) if symbol else ([], None, "No recipe symbol")
        details.append(
            dict(rank=rank,
                 generation=periods[period]["generation"],
                 group=group_for(symbol),
                 recipe=node["recipe"],
                 node=symbol["node"] if symbol else node["node"],
                 kernel=symbol["kernel"] if symbol else node["kernel"],
                 dtype=symbol["dtype"] if symbol else node["reported_dtype"],
                 shape=None,
                 shape_status="Requires exact compiler tensor contract; not inferred from dtype",
                 lane_events=len(rows),
                 calls=count,
                 calls_status=reason,
                 mean_call_ms=statistics.mean(samples) if samples else None,
                 activity_union_ms=union(rows_item[:2] for rows_item in rows) / 1000))
    accounting, resource_rows = [], []
    for i, (period, groups) in enumerate(zip(periods, grouped)):
        duration = period["end"] - period["start"]
        ledger = partition(groups, period["start"], period["end"])
        accounting.append(dict(**period, window_ms=duration / 1000, groups_ms={k: v / 1000 for k, v in ledger.items()}))
        for rank in range(4):
            a, b = period["rank_bounds"][rank]
            for engine in ("TPC", "MME", "DMA", "NIC"):
                spans = resources[rank, engine, i]
                active = union(spans)
                resource_rows.append(
                    dict(rank=rank,
                         generation=period["generation"],
                         engine=engine,
                         active_ms=active / 1000,
                         fraction_global=active / duration,
                         fraction_rank_host_scope=union(
                             (max(a, s), min(b, e)) for s, e in spans if e > a and s < b) / (b - a)))
    averages = {group: statistics.mean(row["groups_ms"][group] for row in accounting) for group in GROUPS}
    mean = statistics.mean(row["window_ms"] for row in accounting)
    report = dict(scope="Four-rank activity in consecutive PP0/TP0 batch-entry periods; ms/batch, not ms/output-token",
                  microbatch_policy=args.microbatches,
                  count=len(periods),
                  batch_sizes=sorted({p["batch"]
                                      for p in periods}),
                  mean_window_ms=mean,
                  groups=[dict(group=k, ms=v, percent=100 * v / mean) for k, v in averages.items()],
                  proof="Cross-group activity is assigned to overlap, not fixed-priority critical-path attribution",
                  limitations=[
                      "NIC points do not establish complete communication latency",
                      "Host waits are envelopes, not additive communication cost",
                      "Activity fraction is not peak FLOP or bandwidth utilization",
                      "Shape and unmapped symbols require compiler metadata joins"
                  ],
                  mapped_events={
                      f"rank{r}_{kind}": value
                      for (r, kind), value in counters.items()
                  })
    for filename, value in (("batch-summary.json", report), ("batch-windows.json", accounting),
                            ("batch-kernels.json", details), ("batch-resources.json", resource_rows)):
        (root / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
