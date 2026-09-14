#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense H3 FusedSDPA softmax modes at the production token shape.

The archived production benchmark is the timing baseline.  The current
``fp32`` mode is executed once only to provide a paired numerical reference;
only candidate modes are timed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


def _device_wave(function, iterations: int) -> tuple[float, float]:
    begin = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    begin.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    end.record()
    torch.hpu.synchronize()
    return begin.elapsed_time(end) / iterations, (time.perf_counter_ns() - started) / 1e6 / iterations


def _forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    used_rows: int,
    softmax_mode: str,
) -> torch.Tensor:
    """Replay the complete BSHD-to-BHSD H3 attention path."""
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    original_rows = query.shape[1]
    query = query[:, :used_rows].permute(0, 2, 1, 3).contiguous()
    key = key[:, :used_rows].permute(0, 2, 1, 3).contiguous()
    value = value[:, :used_rows].permute(0, 2, 1, 3).contiguous()
    output = FusedSDPA.apply(
        query,
        key,
        value,
        None,
        0.0,
        False,
        query.shape[-1]**-0.5,
        softmax_mode,
        True,
    ).permute(0, 2, 1, 3)
    if used_rows < original_rows:
        output = torch.cat(
            (
                output,
                output.new_zeros(
                    output.shape[0],
                    original_rows - used_rows,
                    output.shape[2],
                    output.shape[3],
                ),
            ),
            dim=1,
        )
    return output


def _sampled_stats(candidate: torch.Tensor, reference: torch.Tensor, used_rows: int) -> dict[str, Any]:
    stride = max(used_rows // 1024, 1)
    candidate_sample = candidate[:, :used_rows:stride].float()
    reference_sample = reference[:, :used_rows:stride].float()
    difference = candidate_sample - reference_sample
    reference_norm = torch.linalg.vector_norm(reference_sample).clamp_min(1e-12)
    reference_peak = reference_sample.abs().max().clamp_min(1e-12)
    result = {
        "sample_row_stride": stride,
        "sample_shape": list(candidate_sample.shape),
        "finite_full_output": bool(torch.isfinite(candidate[:, :used_rows]).all().cpu()),
        "suffix_zero": bool((candidate[:, used_rows:] == 0).all().cpu()),
        "max_abs_error": float(difference.abs().max().cpu()),
        "mean_abs_error": float(difference.abs().mean().cpu()),
        "relative_l2": float((torch.linalg.vector_norm(difference) / reference_norm).cpu()),
        "max_error_over_reference_peak": float((difference.abs().max() / reference_peak).cpu()),
    }
    return result


def _profile(function, output_path: Path) -> dict[str, Any]:
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            record_shapes=True,
    ) as profiler:
        function()
        torch.hpu.synchronize()
    profiler.export_chrome_trace(str(output_path))
    events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]
    names = sorted({
        str(event.get("name"))
        for event in events
        if "sdpa" in str(event.get("name", "")).lower() or "softmax" in str(event.get("name", "")).lower()
    })
    return {"trace": output_path.name, "attention_event_names": names}


def _hl_smi() -> str:
    result = subprocess.run(
        [
            "hl-smi",
            "--query-aip=module_id,memory.used,utilization.aip,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="archived benchmark_hpu_chain result.json")
    parser.add_argument("--rows", type=int, default=38272)
    parser.add_argument("--used-rows", type=int, default=38222)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    archived = json.loads(args.baseline.read_text(encoding="utf-8"))
    archived_case = next(case for case in archived["attention"]["cases"] if case["case"] == "production")

    report: dict[str, Any] = {
        "schema_version":
        1,
        "status":
        "running",
        "question":
        "Can H3 keep dense FusedSDPA semantics while avoiding FP32 softmax at the production shape?",
        "archived_baseline":
        str(args.baseline.resolve()),
        "archived_fp32_recompute_device_ms":
        archived_case["device_ms"],
        "shape_bshd": [1, args.rows, args.heads, args.head_size],
        "used_rows":
        args.used_rows,
        "seed":
        args.seed,
        "visible_modules":
        os.environ.get("HABANA_VISIBLE_MODULES"),
        "hls_module_id":
        os.environ.get("HLS_MODULE_ID"),
        "cpu_affinity":
        sorted(os.sched_getaffinity(0)),
        "torch_version":
        torch.__version__,
        "argv":
        sys.argv,
        "working_directory":
        str(Path.cwd()),
        "source_commit":
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "pt_hpu_lazy_mode":
        os.environ.get("PT_HPU_LAZY_MODE"),
        "hl_smi_before":
        _hl_smi(),
    }
    try:
        import habana_frameworks.torch  # noqa: F401

        torch.hpu.init()
        torch.manual_seed(args.seed)
        torch.hpu.reset_peak_memory_stats()
        query = torch.randn(
            1,
            args.rows,
            args.heads,
            args.head_size,
            dtype=torch.bfloat16,
            device="hpu",
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        reference = _forward(query, key, value, used_rows=args.used_rows, softmax_mode="fp32")
        torch.hpu.synchronize()
        candidates: dict[str, Any] = {}
        for mode in ("None", "fast"):
            function = lambda mode=mode: _forward(
                query,
                key,
                value,
                used_rows=args.used_rows,
                softmax_mode=mode,
            )
            for _ in range(args.warmups):
                function()
            torch.hpu.synchronize()
            samples = []
            for _ in range(args.waves):
                device_ms, host_ms = _device_wave(function, args.iterations)
                samples.append({"device_ms": device_ms, "synchronized_host_ms": host_ms})
            candidate = function()
            torch.hpu.synchronize()
            correctness = _sampled_stats(candidate, reference, args.used_rows)
            if not correctness["finite_full_output"] or not correctness["suffix_zero"]:
                raise AssertionError(f"invalid {mode} output: {correctness}")
            numerical_gate_pass = (correctness["relative_l2"] <= 0.01
                                   and correctness["max_error_over_reference_peak"] <= 0.02)
            median_device_ms = statistics.median(sample["device_ms"] for sample in samples)
            candidates[mode] = {
                "softmax_mode": mode,
                "recompute_mode": True,
                "samples": samples,
                "median_device_ms": median_device_ms,
                "median_synchronized_host_ms": statistics.median(sample["synchronized_host_ms"] for sample in samples),
                "over_archived_fp32_ratio": median_device_ms / archived_case["device_ms"],
                "correctness_vs_fp32": correctness,
                "numerical_gate": {
                    "relative_l2_max": 0.01,
                    "max_error_over_reference_peak_max": 0.02,
                    "pass": numerical_gate_pass,
                },
            }
            del candidate

        qualified = [label for label, result in candidates.items() if result["numerical_gate"]["pass"]]
        if not qualified:
            raise AssertionError(f"no attention candidate passed the numerical gate: {candidates}")
        selected = min(qualified, key=lambda label: candidates[label]["median_device_ms"])
        selected_function = lambda: _forward(
            query,
            key,
            value,
            used_rows=args.used_rows,
            softmax_mode=selected,
        )
        candidates[selected]["execution_evidence"] = _profile(
            selected_function,
            args.output_dir / f"{selected.lower()}-attention.trace.json",
        )
        original = selected_function()
        changed = _forward(
            query + 0.125,
            key,
            value,
            used_rows=args.used_rows,
            softmax_mode=selected,
        )
        original_checksum = float(original[:, :args.used_rows:251].float().mean().cpu())
        changed_checksum = float(changed[:, :args.used_rows:251].float().mean().cpu())
        if original_checksum == changed_checksum:
            raise AssertionError("changing the query did not change the selected attention output")
        del original, changed
        report.update(
            candidates=candidates,
            selected_qualified_candidate=selected,
            changing_input_checksums=[original_checksum, changed_checksum],
            peak_allocated_bytes=int(torch.hpu.max_memory_allocated()),
            hl_smi_after=_hl_smi(),
            status="pass",
        )
    except BaseException as exc:
        report.update(status="fail", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
