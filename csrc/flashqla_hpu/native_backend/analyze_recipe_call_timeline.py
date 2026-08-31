#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import ijson


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersect_intervals(
    lhs: list[tuple[float, float]],
    rhs: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    lhs = merge_intervals(lhs)
    rhs = merge_intervals(rhs)
    result: list[tuple[float, float]] = []
    left = right = 0
    while left < len(lhs) and right < len(rhs):
        start = max(lhs[left][0], rhs[right][0])
        end = min(lhs[left][1], rhs[right][1])
        if start < end:
            result.append((start, end))
        if lhs[left][1] <= rhs[right][1]:
            left += 1
        else:
            right += 1
    return result


def engine_name(event: dict) -> str | None:
    tid = int(event.get("tid", -1))
    name = str(event.get("name", ""))
    args = event.get("args") or {}
    hardware_name = str(args.get("HW event name", "")).upper()
    if name in {"BatchGemm", "GEMM"} or 6000 <= tid < 7000:
        return "MME"
    if "MME" in hardware_name:
        return "MME"
    if 7000 <= tid < 8000 or "TPC" in hardware_name:
        return "TPC"
    if name.startswith("Dma") or 4000 <= tid < 5000:
        return "DMA"
    if "DMA" in hardware_name:
        return "DMA"
    return None


def device_events(trace: Path):
    with gzip.open(trace, "rb") as trace_file:
        for event in ijson.items(trace_file, "traceEvents.item"):
            if event.get("ph") != "X" or event.get("pid") != 0:
                continue
            if float(event.get("dur", 0.0)) <= 0:
                continue
            engine = engine_name(event)
            if engine is not None:
                yield event, engine


def cluster_starts(starts: list[float], gap_us: float) -> list[float]:
    clusters: list[list[float]] = []
    for start in sorted(starts):
        if not clusters or start - clusters[-1][-1] > gap_us:
            clusters.append([start])
        else:
            clusters[-1].append(start)
    return [min(cluster) for cluster in clusters]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--recipe-handle", required=True)
    parser.add_argument(
        "--marker",
        required=True,
        help="EventName substring that occurs once per recipe call",
    )
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument("--marker-gap-us", type=float, default=1_000.0)
    parser.add_argument(
        "--max-window-us",
        type=float,
        help="Cap the selected call window when no following marker exists.",
    )
    parser.add_argument("--minimum-node-us", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker_events: list[float] = []
    recipe_events: list[tuple[dict, str]] = []
    for event, engine in device_events(args.trace):
        event_args = event.get("args") or {}
        if str(event_args.get("recipeHandle", "")) != args.recipe_handle:
            continue
        recipe_events.append((event, engine))
        if args.marker in str(event_args.get("EventName", "")):
            marker_events.append(float(event["ts"]))

    call_starts = cluster_starts(marker_events, args.marker_gap_us)
    if not call_starts:
        raise RuntimeError(f"No marker events matched {args.marker!r}")
    if not 0 <= args.call_index < len(call_starts):
        raise ValueError(
            f"call-index {args.call_index} is outside 0..{len(call_starts) - 1}"
        )

    window_start = call_starts[args.call_index]
    window_end = (
        call_starts[args.call_index + 1]
        if args.call_index + 1 < len(call_starts)
        else float("inf")
    )
    if args.max_window_us is not None:
        if args.max_window_us <= 0:
            raise ValueError("max-window-us must be positive")
        window_end = min(window_end, window_start + args.max_window_us)
    by_node: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event, engine in recipe_events:
        event_args = event.get("args") or {}
        start = float(event["ts"])
        if not window_start <= start < window_end:
            continue
        end = start + float(event["dur"])
        node = str(event_args.get("EventName", "")) or str(event["name"])
        node = node.split("/", 1)[0] if node.startswith("fusedTPCNode") else node
        by_node[node][engine].append((start, end))
        all_intervals[engine].append((start, end))

    combined = [
        interval
        for engine_intervals in all_intervals.values()
        for interval in engine_intervals
    ]
    if not combined:
        raise RuntimeError("Selected recipe call has no device events")
    device_start = min(start for start, _ in combined)
    device_end = max(end for _, end in combined)
    device_union = interval_duration(combined)
    span = device_end - device_start
    print(
        f"trace={args.trace} recipe_handle={args.recipe_handle} "
        f"calls={len(call_starts)} selected_call={args.call_index}"
    )
    print(
        f"device_span_ms={span / 1_000:.6f} "
        f"any_engine_union_ms={device_union / 1_000:.6f} "
        f"active_fraction={device_union / span:.4%}"
    )
    for engine in ("MME", "TPC", "DMA"):
        duration = interval_duration(all_intervals[engine])
        print(
            f"engine={engine} union_ms={duration / 1_000:.6f} "
            f"span_share={duration / span:.4%}"
        )

    rows = []
    for node, engines in by_node.items():
        intervals = [
            interval
            for engine_intervals in engines.values()
            for interval in engine_intervals
        ]
        union = interval_duration(intervals)
        if union < args.minimum_node_us:
            continue
        first = min(start for start, _ in intervals)
        last = max(end for _, end in intervals)
        mme = interval_duration(engines["MME"])
        tpc = interval_duration(engines["TPC"])
        dma = interval_duration(engines["DMA"])
        tpc_mme = interval_duration(
            intersect_intervals(engines["TPC"], engines["MME"])
        )
        rows.append((first, union, last, mme, tpc, dma, tpc_mme, node))
    if args.top:
        rows = sorted(rows, key=lambda row: row[1], reverse=True)[: args.top]
    else:
        rows.sort()
    for first, union, last, mme, tpc, dma, tpc_mme, node in rows:
        print(
            f"{(first - device_start) / 1_000:8.3f}.."
            f"{(last - device_start) / 1_000:8.3f} "
            f"union={union / 1_000:7.3f} "
            f"MME={mme / 1_000:7.3f} "
            f"TPC={tpc / 1_000:7.3f} "
            f"DMA={dma / 1_000:7.3f} "
            f"TPCxMME={tpc_mme / 1_000:7.3f} {node}"
        )


if __name__ == "__main__":
    main()
