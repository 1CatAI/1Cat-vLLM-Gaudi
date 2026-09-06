import argparse
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import ijson


def merge_intervals(intervals, max_gap=0.0):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(lhs, rhs):
    result = []
    left = right = 0
    while left < len(lhs) and right < len(rhs):
        start = max(lhs[left][0], rhs[right][0])
        end = min(lhs[left][1], rhs[right][1])
        if start < end:
            result.append([start, end])
        if lhs[left][1] <= rhs[right][1]:
            left += 1
        else:
            right += 1
    return result


def event_engine(event):
    tid = event.get("tid", -1)
    name = event.get("name", "")
    if name in ("BatchGemm", "GEMM") or 6000 <= tid < 7000:
        return "mme"
    if 7000 <= tid < 8000:
        return "tpc"
    if name == "DmaMemcpy" or 4000 <= tid < 5000:
        return "dma"
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="number of kernel resource-time hotspots to print per engine",
    )
    parser.add_argument(
        "--node-pattern",
        default="",
        help="only include device events whose graph EventName contains this text",
    )
    parser.add_argument(
        "--cluster-gap-us",
        type=float,
        default=500.0,
        help="maximum internal gap when grouping filtered events into calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    by_engine = {"mme": [], "tpc": [], "dma": []}
    names = Counter()
    resource_us = Counter()
    kernel_intervals = defaultdict(list)
    kernel_nodes = defaultdict(set)
    with gzip.open(args.trace, "rb") as trace_file:
        for event in ijson.items(trace_file, "traceEvents.item"):
            if event.get("ph") != "X" or event.get("pid") != 0:
                continue
            event_args = event.get("args", {})
            event_name = event_args.get("EventName", "")
            if args.node_pattern and args.node_pattern not in event_name:
                continue
            engine = event_engine(event)
            if engine is None or event.get("dur", 0) <= 0:
                continue
            start = float(event["ts"])
            event_duration = float(event["dur"])
            interval = (start, start + event_duration)
            by_engine[engine].append(interval)
            dtype = event_args.get("dataType", "unknown")
            key = (engine, event["name"], dtype)
            names[key] += 1
            resource_us[key] += event_duration
            kernel_intervals[key].append(interval)
            if event_name:
                kernel_nodes[key].add(event_name)

    merged = {
        engine: merge_intervals(intervals)
        for engine, intervals in by_engine.items()
    }
    all_intervals = merge_intervals(
        interval for intervals in by_engine.values() for interval in intervals
    )
    if not all_intervals:
        raise RuntimeError("Trace has no HPU engine events")
    if args.node_pattern:
        call_clusters = merge_intervals(
            all_intervals,
            max_gap=args.cluster_gap_us,
        )
        call_spans = [end - start for start, end in call_clusters]
        wall = sum(call_spans)
    else:
        call_clusters = []
        call_spans = []
        wall = all_intervals[-1][1] - all_intervals[0][0]
    active = duration(all_intervals)
    mme_active = duration(merged["mme"])
    mme_tpc_overlap = duration(intersection(merged["mme"], merged["tpc"]))
    mme_overlap_share = (
        f"{mme_tpc_overlap / mme_active:.4%}" if mme_active else "n/a"
    )

    print(f"trace={args.trace}")
    if args.node_pattern:
        print(
            f"node_pattern={args.node_pattern} calls={len(call_spans)} "
            f"call_spans_us={call_spans}"
        )
    print(f"device_span_us={wall:.6f}")
    print(f"any_engine_active_us={active:.6f} utilization={active / wall:.4%}")
    for engine in ("mme", "tpc", "dma"):
        engine_duration = duration(merged[engine])
        other_intervals = merge_intervals(
            interval
            for other_engine, intervals in by_engine.items()
            if other_engine != engine
            for interval in intervals
        )
        overlap_with_other = duration(
            intersection(merged[engine], other_intervals)
        )
        print(
            f"{engine}_active_us={engine_duration:.6f} "
            f"utilization={engine_duration / wall:.4%} "
            f"intervals={len(merged[engine])} "
            f"overlap_other_us={overlap_with_other:.6f} "
            f"exclusive_us={engine_duration - overlap_with_other:.6f}"
        )
    print(
        f"mme_tpc_overlap_us={mme_tpc_overlap:.6f} "
        f"share_of_span={mme_tpc_overlap / wall:.4%} "
        f"share_of_mme={mme_overlap_share}"
    )
    print(f"all_engine_idle_us={wall - active:.6f}")
    for engine in ("mme", "tpc", "dma"):
        engine_resource_us = sum(
            value
            for (event_engine, _, _), value in resource_us.items()
            if event_engine == engine
        )
        hotspots = sorted(
            (
                (key, value)
                for key, value in resource_us.items()
                if key[0] == engine
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: args.top]
        for key, value in hotspots:
            _, name, dtype = key
            coverage_us = duration(merge_intervals(kernel_intervals[key]))
            share = value / engine_resource_us if engine_resource_us else 0.0
            print(
                f"hotspot engine={engine} name={name} dtype={dtype} "
                f"resource_us={value:.6f} resource_share={share:.4%} "
                f"wall_coverage_us={coverage_us:.6f} "
                f"events={names[key]} nodes={len(kernel_nodes[key])}"
            )


if __name__ == "__main__":
    main()
