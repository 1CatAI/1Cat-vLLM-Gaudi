#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import ijson

from analyze_full_prefill_trace import (
    component_name,
    engine_name,
    normalize_layers,
    operation_name,
)


Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_duration(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def device_events(trace: Path):
    with gzip.open(trace, "rb") as trace_file:
        for event in ijson.items(trace_file, "traceEvents.item"):
            if event.get("ph") != "X" or event.get("pid") != 0:
                continue
            duration = float(event.get("dur", 0.0))
            if duration <= 0:
                continue
            engine = engine_name(event)
            if engine in {"MME", "TPC", "DMA"}:
                yield event, engine


def complement(intervals: list[Interval], start: float, end: float) -> list[Interval]:
    gaps: list[Interval] = []
    cursor = start
    for interval_start, interval_end in merge_intervals(intervals):
        if cursor < interval_start:
            gaps.append((cursor, interval_start))
        cursor = max(cursor, interval_end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def intersecting_gap_indices(
    gap_starts: list[float],
    gaps: list[Interval],
    start: float,
    end: float,
):
    index = max(0, bisect.bisect_right(gap_starts, start) - 1)
    while index < len(gaps) and gaps[index][0] < end:
        gap_start, gap_end = gaps[index]
        overlap_start = max(start, gap_start)
        overlap_end = min(end, gap_end)
        if overlap_start < overlap_end:
            yield index, overlap_start, overlap_end
        index += 1


def event_label(event: dict) -> str:
    args = event.get("args") or {}
    label = str(args.get("EventName", "")) or str(event.get("name", ""))
    return normalize_layers(label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--minimum-gap-us", type=float, default=50.0)
    parser.add_argument("--top-windows", type=int, default=40)
    parser.add_argument("--top-events", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mme_intervals: list[Interval] = []
    device_start = float("inf")
    device_end = float("-inf")
    for event, engine in device_events(args.trace):
        start = float(event["ts"])
        end = start + float(event["dur"])
        device_start = min(device_start, start)
        device_end = max(device_end, end)
        if engine == "MME":
            mme_intervals.append((start, end))

    if not mme_intervals or device_start >= device_end:
        raise RuntimeError("Trace contains no usable MME/device intervals")

    all_gaps = complement(mme_intervals, device_start, device_end)
    gaps = [
        gap
        for gap in all_gaps
        if gap[1] - gap[0] >= args.minimum_gap_us
    ]
    gap_starts = [start for start, _ in gaps]
    active_by_gap: dict[int, list[Interval]] = defaultdict(list)
    engine_by_gap: dict[int, dict[str, list[Interval]]] = defaultdict(
        lambda: defaultdict(list)
    )
    operation_intervals: dict[str, list[Interval]] = defaultdict(list)
    component_intervals: dict[str, list[Interval]] = defaultdict(list)
    event_resource_us: Counter[str] = Counter()
    event_gap_resource_us: dict[int, Counter[str]] = defaultdict(Counter)

    for event, engine in device_events(args.trace):
        if engine == "MME":
            continue
        start = float(event["ts"])
        end = start + float(event["dur"])
        label = event_label(event)
        operation = operation_name(label)
        component = component_name(label)
        for gap_index, overlap_start, overlap_end in intersecting_gap_indices(
            gap_starts,
            gaps,
            start,
            end,
        ):
            overlap = (overlap_start, overlap_end)
            duration = overlap_end - overlap_start
            active_by_gap[gap_index].append(overlap)
            engine_by_gap[gap_index][engine].append(overlap)
            operation_intervals[operation].append(overlap)
            component_intervals[component].append(overlap)
            event_resource_us[label] += duration
            event_gap_resource_us[gap_index][label] += duration

    mme_busy_us = interval_duration(mme_intervals)
    device_span_us = device_end - device_start
    kept_gap_us = sum(end - start for start, end in gaps)
    active_gap_us = interval_duration(
        [interval for intervals in active_by_gap.values() for interval in intervals]
    )
    print(f"trace={args.trace}")
    print(
        f"device_span_ms={device_span_us / 1_000:.3f} "
        f"mme_busy_union_ms={mme_busy_us / 1_000:.3f} "
        f"mme_busy_share={mme_busy_us / device_span_us:.2%}"
    )
    print(
        f"gaps_ge_{args.minimum_gap_us:g}us={len(gaps)} "
        f"gap_ms={kept_gap_us / 1_000:.3f} "
        f"tpc_dma_active_gap_ms={active_gap_us / 1_000:.3f} "
        f"true_idle_gap_ms={(kept_gap_us - active_gap_us) / 1_000:.3f}"
    )

    print("\nMME-idle wall coverage by operation")
    for name, intervals in sorted(
        operation_intervals.items(),
        key=lambda item: interval_duration(item[1]),
        reverse=True,
    ):
        print(f"{name:16s} {interval_duration(intervals) / 1_000:9.3f} ms")

    print("\nMME-idle wall coverage by component")
    for name, intervals in sorted(
        component_intervals.items(),
        key=lambda item: interval_duration(item[1]),
        reverse=True,
    ):
        print(f"{name:24s} {interval_duration(intervals) / 1_000:9.3f} ms")

    print("\nTop TPC/DMA event resource time while MME is idle")
    for label, duration in event_resource_us.most_common(args.top_events):
        print(f"{duration / 1_000:9.3f} ms  {label}")

    rows = []
    for index, (start, end) in enumerate(gaps):
        duration = end - start
        active = interval_duration(active_by_gap[index])
        tpc = interval_duration(engine_by_gap[index]["TPC"])
        dma = interval_duration(engine_by_gap[index]["DMA"])
        top_labels = ", ".join(
            f"{label}={resource / 1_000:.3f}ms"
            for label, resource in event_gap_resource_us[index].most_common(3)
        )
        rows.append((duration, start, active, tpc, dma, top_labels))

    print("\nLargest MME-idle windows")
    for duration, start, active, tpc, dma, top_labels in sorted(
        rows,
        reverse=True,
    )[: args.top_windows]:
        print(
            f"at={(start - device_start) / 1_000:9.3f}ms "
            f"gap={duration / 1_000:7.3f}ms "
            f"active={active / 1_000:7.3f}ms "
            f"TPC={tpc / 1_000:7.3f}ms DMA={dma / 1_000:7.3f}ms "
            f"{top_labels}"
        )


if __name__ == "__main__":
    main()
