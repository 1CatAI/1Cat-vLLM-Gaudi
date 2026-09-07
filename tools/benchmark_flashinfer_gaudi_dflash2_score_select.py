# SPDX-License-Identifier: Apache-2.0
"""Benchmark fused selected-row DFlash2 scoring and path selection."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flashinfer_gaudi._native import native_dflash2_score_select_op
from flashinfer_gaudi.dflash2 import score_and_select_path_reference

_STEPS = 7
_TOP_K = 16
_VOCAB_SIZE = 248320
_RANK = 256


def _make_tables() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3011)
    predecessor = torch.randn(_VOCAB_SIZE, _RANK, dtype=torch.bfloat16, generator=generator) * 0.05
    successor = torch.randn(_VOCAB_SIZE, _RANK, dtype=torch.bfloat16, generator=generator) * 0.05
    return predecessor.to("hpu"), successor.to("hpu")


def _make_request_inputs(batch: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(3203 + batch)
    candidates = torch.randint(
        _VOCAB_SIZE,
        (batch, _STEPS, _TOP_K),
        dtype=torch.int32,
        generator=generator,
    )
    unary = torch.randn(batch, _STEPS, _TOP_K, dtype=torch.float32, generator=generator)
    hidden = torch.randn(batch, _STEPS, _RANK, dtype=torch.bfloat16, generator=generator) * 0.1
    anchors = torch.randint(_VOCAB_SIZE, (batch, ), dtype=torch.int32, generator=generator)
    return tuple(tensor.to("hpu") for tensor in (candidates, unary, hidden, anchors))


def _compile_reference():
    return torch.compile(score_and_select_path_reference, backend="hpu_backend", fullgraph=True, dynamic=False)


def _compile_native(op):

    def run(
        predecessor: torch.Tensor,
        successor: torch.Tensor,
        candidates: torch.Tensor,
        unary: torch.Tensor,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        return op(predecessor, successor, candidates, unary, hidden, anchors)

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

    native_op = native_dflash2_score_select_op()
    if native_op is None:
        raise RuntimeError("The native DFlash2 score-select op is not loaded. Rebuild with "
                           "`python3 tools/build_flashinfer_gaudi.py`.")
    predecessor, successor = _make_tables()

    report: dict[str, object] = {"batches": {}}
    for batch in (int(value) for value in args.batches.split(",") if value):
        if not 1 <= batch <= 16:
            raise ValueError(f"Native DFlash2 score-select supports batches 1 through 16, got {batch}.")
        torch._dynamo.reset()
        request_inputs = _make_request_inputs(batch)
        inputs = (predecessor, successor, *request_inputs)
        reference = _compile_reference()
        native = _compile_native(native_op)
        expected = reference(*inputs)
        actual = native(*inputs)
        torch.hpu.synchronize()
        expected_cpu = expected.cpu()
        actual_cpu = actual.cpu()
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0)

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
            "draft_tokens_exact": True,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
