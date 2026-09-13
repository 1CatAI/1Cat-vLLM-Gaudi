# SPDX-License-Identifier: Apache-2.0
"""Benchmark the single-launch Gaudi2 DFlash2 greedy lattice walk."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flashinfer_gaudi._native import native_dflash2_select_path_op
from flashinfer_gaudi.dflash2 import select_path_reference

_STEPS = 7
_TOP_K = 16
_VOCAB_SIZE = 248320


def _make_inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1709 + batch)
    candidate_ids = torch.randint(
        _VOCAB_SIZE,
        (batch, _STEPS, _TOP_K),
        dtype=torch.int32,
        generator=generator,
    )
    scores = torch.randn(
        batch,
        _STEPS,
        _TOP_K,
        _TOP_K,
        dtype=torch.float32,
        generator=generator,
    )
    # Force a first-step tie while keeping later predecessor-dependent rows
    # random, so the benchmark checks both tie and chained-walk semantics.
    tie_score = scores[:, 0, 0].amax(dim=-1) + 1
    scores[:, 0, 0, 0] = tie_score
    scores[:, 0, 0, 1] = tie_score
    return candidate_ids.to("hpu"), scores.to("hpu")


def _compile_reference():
    return torch.compile(select_path_reference, backend="hpu_backend", fullgraph=True, dynamic=False)


def _compile_native(op):

    def run(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return op(candidate_ids, scores)

    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


def _measure(function, inputs: tuple[torch.Tensor, ...], iterations: int) -> tuple[float, float]:
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function(*inputs)
    end_event.record()
    torch.hpu.synchronize()
    host_ms = (time.perf_counter_ns() - started) / 1e6 / iterations
    return host_ms, start_event.elapsed_time(end_event) / iterations


def _stats(samples: list[tuple[float, float]]) -> dict[str, float]:
    return {
        "host_median_ms": statistics.median(item[0] for item in samples),
        "device_median_ms": statistics.median(item[1] for item in samples),
        "device_min_ms": min(item[1] for item in samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--wave-iterations", type=int, default=100)
    parser.add_argument("--waves", type=int, default=7)
    args = parser.parse_args()

    native_op = native_dflash2_select_path_op()
    if native_op is None:
        raise RuntimeError("The native DFlash2 selector op is not loaded. Rebuild with "
                           "`python3 tools/build_flashinfer_gaudi.py`.")

    report: dict[str, object] = {"batches": {}}
    for batch in (int(value) for value in args.batches.split(",") if value):
        if not 1 <= batch <= 16:
            raise ValueError(f"Native DFlash2 selector supports batches 1 through 16, got {batch}.")
        torch._dynamo.reset()
        inputs = _make_inputs(batch)
        reference = _compile_reference()
        native = _compile_native(native_op)
        expected = reference(*inputs)
        actual = native(*inputs)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

        for _ in range(args.warmups):
            reference(*inputs)
            native(*inputs)
        torch.hpu.synchronize()

        samples: dict[str, list[tuple[float, float]]] = {"pytorch": [], "native": []}
        order = (("pytorch", reference), ("native", native))
        for wave in range(args.waves):
            wave_order = order if wave % 2 == 0 else tuple(reversed(order))
            for name, function in wave_order:
                samples[name].append(_measure(function, inputs, args.wave_iterations))

        reference_stats = _stats(samples["pytorch"])
        native_stats = _stats(samples["native"])
        report["batches"][str(batch)] = {
            "pytorch": reference_stats,
            "native": native_stats,
            "device_speedup": reference_stats["device_median_ms"] / native_stats["device_median_ms"],
            "host_speedup": reference_stats["host_median_ms"] / native_stats["host_median_ms"],
            "tokens_exact": True,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
