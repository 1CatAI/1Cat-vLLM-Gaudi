# SPDX-License-Identifier: Apache-2.0

"""Gaudi2 performance gate for Triton fused add+RMSNorm."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable

from triton.backends.gaudi.driver import prepare_environment

prepare_environment()
os.environ["VLLM_HPU_TRITON_MODE"] = "strict"

import torch  # noqa: E402
from habana_frameworks.torch.hpex.normalization.FusedRMSNorm import FusedRMSNorm  # noqa: E402

from vllm_gaudi.ops.triton_gaudi import (  # noqa: E402
    fused_add_rms_norm,
    prepare_if_enabled,
)


DEFAULT_SHAPES = (
    (1, 768),
    (8, 768),
    (32, 768),
    (128, 768),
    (7, 769),
    (32, 769),
    (1, 5120),
    (8, 5120),
    (32, 5120),
    (128, 5120),
    (1, 8192),
    (8, 8192),
    (32, 8192),
    (128, 8192),
)


def _vendor_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_sum = residual + hidden
    return FusedRMSNorm.apply(residual_sum, weight, epsilon), residual_sum


def _triton_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = fused_add_rms_norm(hidden, residual, weight, epsilon)
    if result is None:
        raise RuntimeError(
            "strict Gaudi Triton benchmark unexpectedly selected the vendor path")
    return result


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        rows, columns = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shapes must use ROWSxCOLUMNS") from exc
    if rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, columns


def _timed(fn: Callable[[], object], repetitions: int) -> tuple[float, float]:
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter_ns()
    outputs = []
    for _ in range(repetitions):
        outputs.append(fn())
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000 / repetitions
    return start.elapsed_time(end) / repetitions, wall_ms


def _benchmark_pair(
    vendor: Callable[[], object],
    triton: Callable[[], object],
    warmup: int,
    repetitions: int,
    rounds: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    for _ in range(warmup):
        vendor()
        triton()
    torch.hpu.synchronize()
    samples: dict[str, list[tuple[float, float]]] = {"vendor": [], "triton": []}
    functions = {"vendor": vendor, "triton": triton}
    for round_index in range(rounds):
        order = ("vendor", "triton") if round_index % 2 == 0 else ("triton", "vendor")
        for name in order:
            samples[name].append(_timed(functions[name], repetitions))
    medians = {
        name: (
            statistics.median(sample[0] for sample in values),
            statistics.median(sample[1] for sample in values),
        )
        for name, values in samples.items()
    }
    return medians["vendor"], medians["triton"]


def _check_torch_compile_fullgraph() -> None:
    rows, columns = 32, 769
    hidden = torch.randn(rows, columns, dtype=torch.bfloat16, device="hpu")
    residual = torch.randn_like(hidden)
    weight = torch.randn(columns, dtype=torch.bfloat16, device="hpu") * 0.1 + 1.0
    compiled = torch.compile(fused_add_rms_norm, backend="hpu_backend", fullgraph=True, dynamic=False)

    actual_output, actual_residual = compiled(hidden, residual, weight, 1.0e-6)
    expected_residual = residual + hidden
    expected_output = FusedRMSNorm.apply(expected_residual, weight, 1.0e-6)
    torch.hpu.synchronize()
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_output, expected_output, rtol=0.02, atol=0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", dest="shapes", action="append", type=_parse_shape)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-geomean-speedup", type=float, default=1.20)
    parser.add_argument("--min-shape-speedup", type=float, default=0.95)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action="store_true",
        help="benchmark both implementations through hpu_backend fullgraph compilation (default)",
    )
    execution.add_argument(
        "--eager",
        dest="torch_compile",
        action="store_false",
        help="run the diagnostic eager operator comparison",
    )
    parser.set_defaults(torch_compile=True)
    parser.add_argument("--skip-torch-compile-check", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1 or args.rounds < 1:
        parser.error("warmup, repetitions, and rounds must be positive")

    prepare_if_enabled()
    torch.set_grad_enabled(False)
    results = []
    for rows, columns in args.shapes or DEFAULT_SHAPES:
        hidden = torch.randn(rows, columns, dtype=torch.bfloat16, device="hpu")
        residual = torch.randn_like(hidden)
        weight = torch.randn(columns, dtype=torch.bfloat16, device="hpu") * 0.1 + 1.0
        vendor_impl = _vendor_impl
        triton_impl = _triton_impl
        if args.torch_compile:
            # Keep each static shape in an independent Dynamo wrapper. Reusing
            # one wrapper across the matrix measures recompilation/dispatch
            # state rather than the warm recipe selected for that shape.
            torch._dynamo.reset()
            vendor_impl = torch.compile(
                _vendor_impl,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )
            triton_impl = torch.compile(
                _triton_impl,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )

        def vendor():
            return vendor_impl(hidden, residual, weight, 1.0e-6)

        def triton_kernel():
            return triton_impl(hidden, residual, weight, 1.0e-6)

        expected_output, expected_residual = vendor()
        actual_output, actual_residual = triton_kernel()
        torch.hpu.synchronize()
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual_output, expected_output, rtol=0.02, atol=0.02)

        vendor_times, triton_times = _benchmark_pair(
            vendor,
            triton_kernel,
            args.warmup,
            args.repetitions,
            args.rounds,
        )
        result = {
            "rows": rows,
            "columns": columns,
            "torch_compile_fullgraph": args.torch_compile,
            "device_speedup": vendor_times[0] / triton_times[0],
            "wall_speedup": vendor_times[1] / triton_times[1],
            "vendor_device_ms": vendor_times[0],
            "triton_device_ms": triton_times[0],
            "vendor_wall_ms": vendor_times[1],
            "triton_wall_ms": triton_times[1],
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "device_geomean_speedup": statistics.geometric_mean(result["device_speedup"] for result in results),
        "wall_geomean_speedup": statistics.geometric_mean(result["wall_speedup"] for result in results),
        "minimum_device_speedup": min(result["device_speedup"] for result in results),
        "minimum_wall_speedup": min(result["wall_speedup"] for result in results),
    }
    if not args.skip_torch_compile_check:
        _check_torch_compile_fullgraph()
        summary["torch_compile_fullgraph"] = True
    passed = (summary["device_geomean_speedup"] >= args.min_geomean_speedup
              and summary["wall_geomean_speedup"] >= args.min_geomean_speedup
              and summary["minimum_device_speedup"] >= args.min_shape_speedup
              and summary["minimum_wall_speedup"] >= args.min_shape_speedup)
    print(json.dumps({"passed": passed, "summary": summary}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
