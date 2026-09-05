#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import ijson


ENGINE_CAPACITY = {
    "TPC": 24,
    "MME": 4,
    "DMA": 5,
}
LAYER_RE = re.compile(r"(?:_layers\[|layers\.)(\d+)\]?")


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def engine_name(event: dict) -> str | None:
    args = event.get("args") or {}
    hw_name = str(args.get("HW event name", "")).upper()
    event_name = str(event.get("name", "")).upper()
    if hw_name.startswith("TPC") or "TPC" in hw_name:
        return "TPC"
    if hw_name.startswith("MME") or "MME" in hw_name or event_name == "GEMM":
        return "MME"
    if "DMA" in hw_name or event_name.startswith("DMA"):
        return "DMA"
    if event.get("cat") == "kernel" or event.get("pid") == 0:
        return "OTHER_DEVICE"
    return None


def normalize_layers(name: str) -> str:
    return LAYER_RE.sub(lambda match: match.group(0).replace(match.group(1), "*"), name)


def operation_name(name: str) -> str:
    lowered = name.lower()
    if "/linear_attn/" in lowered or "gdn" in lowered:
        return "GDN"
    if "/self_attn/" in lowered or "attention" in lowered or "fsdpa" in lowered:
        return "ATTENTION"
    if "/mlp/" in lowered:
        return "MLP"
    if "rms_norm" in lowered or "layernorm" in lowered:
        return "NORM"
    if "quant" in lowered or "cast" in lowered or "convert" in lowered:
        return "QUANT_CAST"
    return "OTHER"


def component_name(name: str) -> str:
    lowered = name.lower()
    if "/linear_attn/" in lowered or "gdn" in lowered:
        if "/in_proj_ba/" in lowered:
            return "GDN_IN_PROJ_BA"
        if "/in_proj_qkvz/" in lowered:
            return "GDN_IN_PROJ_QKVZ"
        if "/out_proj/" in lowered:
            return "GDN_OUT_PROJ"
        if "conv" in lowered:
            return "GDN_CONV"
        if any(token in lowered for token in ("batchgemm", "batch_gemm", "bmm", "matmul")):
            return "GDN_STATE_MATH"
        if any(token in lowered for token in ("index", "gather", "scatter", "select")):
            return "GDN_STATE_IO"
        return "GDN_ELEMENTWISE"
    if "/mlp/gate_up_proj/" in lowered:
        return "MLP_GATE_UP"
    if "/mlp/down_proj/" in lowered:
        return "MLP_DOWN"
    if "/mlp/" in lowered:
        return "MLP_ELEMENTWISE"
    if "/self_attn/qkv_proj/" in lowered:
        return "ATTN_QKV"
    if "/self_attn/o_proj/" in lowered:
        return "ATTN_OUT"
    if "/self_attn/" in lowered or "attention" in lowered or "fsdpa" in lowered:
        return "ATTN_CORE"
    if "rms_norm" in lowered or "layernorm" in lowered:
        return "NORM"
    if "quant" in lowered or "cast" in lowered or "convert" in lowered:
        return "QUANT_CAST"
    return "OTHER"


def counter_rows(
    durations: Counter,
    counts: Counter,
    top: int,
) -> list[dict]:
    return [
        {
            "name": name,
            "cumulative_ms": duration / 1000.0,
            "count": counts[name],
            "mean_us": duration / counts[name],
        }
        for name, duration in durations.most_common(top)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    category_counts: Counter[str] = Counter()
    pid_counts: Counter[str] = Counter()
    kernel_durations: Counter[str] = Counter()
    kernel_counts: Counter[str] = Counter()
    event_durations: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    cpu_durations: Counter[str] = Counter()
    cpu_counts: Counter[str] = Counter()
    recipe_counts: Counter[str] = Counter()
    recipe_names: dict[str, str] = {}
    engine_cumulative_us: Counter[str] = Counter()
    recipe_engine_cumulative_us: Counter[tuple[str, str]] = Counter()
    recipe_kernel_durations: dict[str, Counter[str]] = defaultdict(Counter)
    recipe_kernel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    recipe_event_durations: dict[str, Counter[str]] = defaultdict(Counter)
    recipe_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    intervals_all: list[tuple[float, float]] = []
    intervals_by_engine: dict[str, list[tuple[float, float]]] = defaultdict(list)
    intervals_by_operation: dict[str, list[tuple[float, float]]] = defaultdict(list)
    intervals_by_component: dict[str, list[tuple[float, float]]] = defaultdict(list)
    intervals_by_engine_operation: dict[
        tuple[str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    intervals_by_recipe: dict[str, list[tuple[float, float]]] = defaultdict(list)
    intervals_by_recipe_operation: dict[
        tuple[str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    intervals_by_recipe_component: dict[
        tuple[str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    engine_steps: list[dict] = []
    launch_recipe_count = 0
    compiled_region_count = 0
    event_count = 0

    with gzip.open(args.trace, "rb") as trace_file:
        for event in ijson.items(trace_file, "traceEvents.item"):
            event_count += 1
            category = str(event.get("cat", ""))
            category_counts[category] += 1
            pid_counts[str(event.get("pid", ""))] += 1
            if event.get("ph") != "X":
                continue

            name = str(event.get("name", ""))
            duration = float(event.get("dur", 0.0))
            event_args = event.get("args") or {}
            recipe_handle = str(event_args.get("recipeHandle", ""))
            recipe_name = str(event_args.get("recipeName", ""))
            if recipe_handle and recipe_name:
                recipe_names[recipe_handle] = recipe_name
            if category in {"cpu_op", "python_function", "user_annotation"}:
                cpu_durations[name] += duration
                cpu_counts[name] += 1
            if name.endswith("_process_engine_step"):
                engine_steps.append(event)
            if category == "privateuse1_runtime" and name == "LaunchRecipe":
                launch_recipe_count += 1
            if category == "cpu_op" and name.startswith("Torch-Compiled Region"):
                compiled_region_count += 1

            engine = engine_name(event)
            if engine is None:
                continue
            start = float(event["ts"])
            end = start + duration
            interval = (start, end)
            model_event = str(event_args.get("EventName", "")) or name
            normalized_event = normalize_layers(model_event)
            operation = operation_name(model_event)
            component = component_name(model_event)

            intervals_all.append(interval)
            intervals_by_engine[engine].append(interval)
            intervals_by_operation[operation].append(interval)
            intervals_by_component[component].append(interval)
            intervals_by_engine_operation[(engine, operation)].append(interval)
            engine_cumulative_us[engine] += duration
            kernel_durations[f"{engine}\t{name}"] += duration
            kernel_counts[f"{engine}\t{name}"] += 1
            event_durations[f"{engine}\t{normalized_event}"] += duration
            event_counts[f"{engine}\t{normalized_event}"] += 1
            if recipe_handle:
                recipe_counts[recipe_handle] += 1
                intervals_by_recipe[recipe_handle].append(interval)
                recipe_engine_cumulative_us[(recipe_handle, engine)] += duration
                recipe_kernel_durations[recipe_handle][f"{engine}\t{name}"] += duration
                recipe_kernel_counts[recipe_handle][f"{engine}\t{name}"] += 1
                recipe_event_durations[recipe_handle][
                    f"{engine}\t{normalized_event}"
                ] += duration
                recipe_event_counts[recipe_handle][
                    f"{engine}\t{normalized_event}"
                ] += 1
                intervals_by_recipe_operation[(recipe_handle, operation)].append(interval)
                intervals_by_recipe_component[(recipe_handle, component)].append(interval)

    device_union_us = interval_union_us(intervals_all)
    device_start = min((start for start, _ in intervals_all), default=0.0)
    device_end = max((end for _, end in intervals_all), default=0.0)
    device_span_us = max(0.0, device_end - device_start)
    engine_summary = {}
    for engine, cumulative_us in engine_cumulative_us.items():
        capacity = ENGINE_CAPACITY.get(engine, 1)
        engine_summary[engine] = {
            "cumulative_ms": cumulative_us / 1000.0,
            "union_ms": interval_union_us(intervals_by_engine[engine]) / 1000.0,
            "capacity": capacity,
            "average_busy_ms_per_engine": cumulative_us / capacity / 1000.0,
            "normalized_occupancy_of_device_span": (
                cumulative_us / capacity / device_span_us if device_span_us else 0.0
            ),
        }

    def union_rows(groups: dict) -> list[dict]:
        rows = [
            {
                "name": str(name),
                "union_ms": interval_union_us(intervals) / 1000.0,
            }
            for name, intervals in groups.items()
        ]
        return sorted(rows, key=lambda row: row["union_ms"], reverse=True)

    recipe_summary = []
    for handle, intervals in intervals_by_recipe.items():
        union_us = interval_union_us(intervals)
        row = {
            "handle": handle,
            "name": recipe_names.get(handle, ""),
            "device_event_count": recipe_counts[handle],
            "union_ms": union_us / 1000.0,
            "engines": {},
            "top_kernels": counter_rows(
                recipe_kernel_durations[handle],
                recipe_kernel_counts[handle],
                15,
            ),
            "top_model_events": counter_rows(
                recipe_event_durations[handle],
                recipe_event_counts[handle],
                20,
            ),
            "operation_union": [],
            "component_union": [],
        }
        for (recipe_handle, operation), operation_intervals in (
            intervals_by_recipe_operation.items()
        ):
            if recipe_handle == handle:
                row["operation_union"].append({
                    "name": operation,
                    "union_ms": interval_union_us(operation_intervals) / 1000.0,
                })
        row["operation_union"].sort(
            key=lambda value: value["union_ms"], reverse=True
        )
        for (recipe_handle, component), component_intervals in (
            intervals_by_recipe_component.items()
        ):
            if recipe_handle == handle:
                row["component_union"].append({
                    "name": component,
                    "union_ms": interval_union_us(component_intervals) / 1000.0,
                })
        row["component_union"].sort(
            key=lambda value: value["union_ms"], reverse=True
        )
        for engine in ENGINE_CAPACITY:
            cumulative_us = recipe_engine_cumulative_us[(handle, engine)]
            if cumulative_us:
                row["engines"][engine] = {
                    "cumulative_ms": cumulative_us / 1000.0,
                    "average_busy_ms_per_engine": (
                        cumulative_us / ENGINE_CAPACITY[engine] / 1000.0
                    ),
                }
        recipe_summary.append(row)
    recipe_summary.sort(key=lambda row: row["union_ms"], reverse=True)

    artifact = {
        "trace": str(args.trace),
        "trace_event_count": event_count,
        "device": {
            "span_ms": device_span_us / 1000.0,
            "union_ms": device_union_us / 1000.0,
            "active_fraction": device_union_us / device_span_us if device_span_us else 0.0,
            "engines": engine_summary,
            "operation_union": union_rows(intervals_by_operation),
            "component_union": union_rows(intervals_by_component),
            "engine_operation_union": union_rows(intervals_by_engine_operation),
        },
        "host": {
            "engine_step_count": len(engine_steps),
            "engine_step_ms": [float(event["dur"]) / 1000.0 for event in engine_steps],
            "launch_recipe_count": launch_recipe_count,
            "compiled_region_count": compiled_region_count,
        },
        "top_kernels": counter_rows(kernel_durations, kernel_counts, args.top),
        "top_model_events": counter_rows(event_durations, event_counts, args.top),
        "top_cpu_events": counter_rows(cpu_durations, cpu_counts, args.top),
        "recipes": recipe_summary,
        "category_counts": category_counts,
        "pid_counts": pid_counts,
        "recipe_counts": recipe_counts,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(json.dumps(artifact["device"], indent=2))
    print(json.dumps(artifact["host"], indent=2))
    print("top_kernels")
    print(json.dumps(artifact["top_kernels"][:20], indent=2))
    print("top_model_events")
    print(json.dumps(artifact["top_model_events"][:20], indent=2))
    print("top_recipes")
    print(json.dumps(artifact["recipes"][:10], indent=2))


if __name__ == "__main__":
    main()
