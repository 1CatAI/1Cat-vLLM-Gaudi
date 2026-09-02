# SPDX-License-Identifier: Apache-2.0

"""Gaudi2 performance gate for Triton BF16 SiLU-and-mul."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable

from triton.backends.gaudi.driver import prepare_environment


prepare_environment()
os.environ.setdefault("VLLM_HPU_TRITON_MODE", "strict")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm_gaudi.ops.triton_gaudi import prepare_if_enabled, silu_and_mul  # noqa: E402


DEFAULT_SHAPES = (
    (1, 2048),
    (8, 2048),
    (32, 2048),
    (1, 3584),
    (8, 3584),
    (32, 3584),
    (1, 11008),
    (8, 11008),
    (32, 11008),
    (1, 17408),
    (8, 17408),
    (32, 17408),
    (128, 17408),
)


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        rows, columns = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shapes must use ROWSxOUTPUT_COLUMNS") from exc
    if rows <= 0 or columns <= 128:
        raise argparse.ArgumentTypeError("rows must be positive and output columns must exceed 128")
    return rows, columns


def _vendor(input_tensor: torch.Tensor) -> torch.Tensor:
    columns = input_tensor.shape[-1] // 2
    return F.silu(input_tensor[..., :columns]) * input_tensor[..., columns:]


def _triton(input_tensor: torch.Tensor) -> torch.Tensor:
    output = silu_and_mul(input_tensor)
    if output is None:
        raise RuntimeError("strict Gaudi Triton benchmark unexpectedly selected the vendor path")
    return output


def _timed(fn: Callable[[], torch.Tensor], repetitions: int) -> tuple[float, float]:
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
    vendor: Callable[[], torch.Tensor],
    triton_kernel: Callable[[], torch.Tensor],
    warmup: int,
    repetitions: int,
    rounds: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    for _ in range(warmup):
        vendor()
        triton_kernel()
    torch.hpu.synchronize()
    samples: dict[str, list[tuple[float, float]]] = {"vendor": [], "triton": []}
    functions = {"vendor": vendor, "triton": triton_kernel}
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", dest="shapes", action="append", type=_parse_shape)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-geomean-speedup", type=float, default=1.20)
    parser.add_argument("--min-shape-speedup", type=float, default=0.95)
    parser.add_argument("--eager", action="store_true", help="measure eager execution instead of fullgraph compile")
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1 or args.rounds < 1:
        parser.error("warmup, repetitions, and rounds must be positive")
    if args.block_size < 128 or args.block_size > 1024 or args.block_size & (args.block_size - 1):
        parser.error("block-size must be a power of two in [128, 1024]")

    os.environ["VLLM_HPU_TRITON_SILU_BLOCK_SIZE"] = str(args.block_size)
    prepare_if_enabled()
    torch.set_grad_enabled(False)
    vendor_impl = _vendor
    triton_impl = _triton
    if not args.eager:
        vendor_impl = torch.compile(_vendor, backend="hpu_backend", fullgraph=True, dynamic=False)
        triton_impl = torch.compile(_triton, backend="hpu_backend", fullgraph=True, dynamic=False)

    results = []
    for rows, columns in args.shapes or DEFAULT_SHAPES:
        input_tensor = (torch.randn(rows, 2 * columns, dtype=torch.bfloat16, device="hpu") * 0.25).contiguous()
        expected = _vendor(input_tensor)
        actual = triton_impl(input_tensor)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.02)

        vendor_times, triton_times = _benchmark_pair(
            lambda: vendor_impl(input_tensor),
            lambda: triton_impl(input_tensor),
            args.warmup,
            args.repetitions,
            args.rounds,
        )
        result = {
            "rows": rows,
            "columns": columns,
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
        "block_size": args.block_size,
        "torch_compile_fullgraph": not args.eager,
        "device_geomean_speedup": statistics.geometric_mean(result["device_speedup"] for result in results),
        "wall_geomean_speedup": statistics.geometric_mean(result["wall_speedup"] for result in results),
        "minimum_device_speedup": min(result["device_speedup"] for result in results),
        "minimum_wall_speedup": min(result["wall_speedup"] for result in results),
    }
    passed = (summary["device_geomean_speedup"] >= args.min_geomean_speedup
              and summary["wall_geomean_speedup"] >= args.min_geomean_speedup
              and summary["minimum_device_speedup"] >= args.min_shape_speedup
              and summary["minimum_wall_speedup"] >= args.min_shape_speedup)
    print(json.dumps({"passed": passed, "summary": summary}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
