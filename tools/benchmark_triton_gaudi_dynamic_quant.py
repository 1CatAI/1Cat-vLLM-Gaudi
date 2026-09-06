# SPDX-License-Identifier: Apache-2.0
"""Gaudi2 performance gate for Triton row-wise dynamic FP8 quantization."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import functools
import json
import os
import statistics
import time

from triton.backends.gaudi.driver import prepare_environment

prepare_environment()
os.environ["VLLM_HPU_TRITON_MODE"] = "strict"

import torch  # noqa: E402

from vllm_gaudi.ops.triton_gaudi import (  # noqa: E402
    dynamic_quant, prepare_if_enabled,
)

FP8_DTYPE = torch.float8_e4m3fn
GAUDI2_FP8_MAX = 240.0
DEFAULT_SHAPES = (
    (1, 3584),
    (8, 3584),
    (16, 3584),
    (32, 3584),
    (1, 5120),
    (8, 5120),
    (16, 5120),
    (32, 5120),
    (1, 11008),
    (8, 11008),
    (16, 11008),
    (32, 11008),
)


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        rows, columns = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shapes must use ROWSxCOLUMNS") from exc
    if rows <= 0 or rows > 32 or columns <= 0 or columns > 16384:
        raise argparse.ArgumentTypeError("dynamic-quant shapes require rows in [1, 32] and columns in [1, 16384]")
    return rows, columns


def _vendor(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (input_tensor.abs().amax(dim=-1, keepdim=True) + 1.0e-8) / GAUDI2_FP8_MAX
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        input_tensor,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float()


def _triton(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = dynamic_quant(input_tensor)
    if result is None:
        raise RuntimeError("strict Gaudi Triton benchmark unexpectedly selected the vendor path")
    return result


def _timed(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    repetitions: int,
) -> tuple[float, float]:
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter_ns()
    output = None
    for _ in range(repetitions):
        output = fn()
    end.record()
    end.synchronize()
    if output is None:
        raise AssertionError("benchmark did not execute")
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000 / repetitions
    return start.elapsed_time(end) / repetitions, wall_ms


def _benchmark_pair(
    vendor: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    triton_kernel: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    repetitions: int,
    rounds: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    for _ in range(warmup):
        vendor()
        triton_kernel()
    torch.hpu.synchronize()
    samples: dict[str, list[tuple[float, float]]] = {
        "vendor": [],
        "triton": [],
    }
    functions = {"vendor": vendor, "triton": triton_kernel}
    for round_index in range(rounds):
        order = (("vendor", "triton") if round_index % 2 == 0 else ("triton", "vendor"))
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


def _validate(
    expected: tuple[torch.Tensor, torch.Tensor],
    actual: tuple[torch.Tensor, torch.Tensor],
) -> float:
    expected_quantized, expected_scale = expected
    actual_quantized, actual_scale = actual
    torch.testing.assert_close(
        actual_scale,
        expected_scale,
        rtol=0.005,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        actual_quantized.float() * actual_scale,
        expected_quantized.float() * expected_scale,
        rtol=0.08,
        atol=0.02,
    )
    return float((actual_quantized == expected_quantized).float().mean().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", dest="shapes", action="append", type=_parse_shape)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-geomean-speedup", type=float, default=1.02)
    parser.add_argument("--min-shape-speedup", type=float, default=0.99)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="measure eager execution instead of hpu_backend fullgraph compilation",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1 or args.rounds < 1:
        parser.error("warmup, repetitions, and rounds must be positive")

    prepare_if_enabled()
    torch.set_grad_enabled(False)
    results = []
    for rows, columns in args.shapes or DEFAULT_SHAPES:
        input_tensor = (torch.randn(rows, columns, dtype=torch.bfloat16, device="hpu") * 0.5).contiguous()
        vendor_impl = _vendor
        triton_impl = _triton
        if not args.eager:
            torch._dynamo.reset()
            vendor_impl = torch.compile(
                _vendor,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )
            triton_impl = torch.compile(
                _triton,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )

        vendor = functools.partial(vendor_impl, input_tensor)
        triton_kernel = functools.partial(triton_impl, input_tensor)

        expected = vendor()
        actual = triton_kernel()
        torch.hpu.synchronize()
        quantized_equal_fraction = _validate(expected, actual)

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
            "torch_compile_fullgraph": not args.eager,
            "quantized_equal_fraction": quantized_equal_fraction,
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
        "torch_compile_fullgraph": not args.eager,
        "device_geomean_speedup": statistics.geometric_mean(result["device_speedup"] for result in results),
        "wall_geomean_speedup": statistics.geometric_mean(result["wall_speedup"] for result in results),
        "minimum_device_speedup": min(result["device_speedup"] for result in results),
        "minimum_wall_speedup": min(result["wall_speedup"] for result in results),
        "minimum_quantized_equal_fraction": min(result["quantized_equal_fraction"] for result in results),
    }
    passed = (summary["device_geomean_speedup"] >= args.min_geomean_speedup
              and summary["wall_geomean_speedup"] >= args.min_geomean_speedup
              and summary["minimum_device_speedup"] >= args.min_shape_speedup
              and summary["minimum_wall_speedup"] >= args.min_shape_speedup)
    print(json.dumps({"passed": passed, "summary": summary}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
