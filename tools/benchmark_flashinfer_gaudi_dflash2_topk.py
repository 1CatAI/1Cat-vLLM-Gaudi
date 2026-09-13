# SPDX-License-Identifier: Apache-2.0
"""Benchmark direct Gaudi TopK CGUID use for DFlash2 candidate generation."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flashinfer_gaudi.dflash2 import _top_k_vendor_cguid, top_k_reference

_STEPS = 7
_TOP_K = 16
_VOCAB_SIZE = 248320


def _make_scores(batch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2017 + batch)
    scores = torch.randn(
        batch * _STEPS,
        _VOCAB_SIZE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    # Production BF16 logits can tie. Guarantee an interior tie so the report
    # exposes ordering differences while candidate-set equality stays strict.
    scores[:, 0] = 100
    scores[:, 1] = 100
    return scores.to("hpu")


def _compile_reference():

    def run(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return top_k_reference(scores, _TOP_K, sorted=True, deterministic=True)

    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


def _compile_candidate():

    def run(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _top_k_vendor_cguid(scores, _TOP_K)

    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


def _measure(function, scores: torch.Tensor, iterations: int) -> tuple[float, float]:
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function(scores)
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
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--wave-iterations", type=int, default=20)
    parser.add_argument("--waves", type=int, default=7)
    args = parser.parse_args()

    report: dict[str, object] = {"batches": {}}
    for batch in (int(value) for value in args.batches.split(",") if value):
        if not 1 <= batch <= 16:
            raise ValueError(f"DFlash2 TopK qualification supports batches 1 through 16, got {batch}.")
        torch._dynamo.reset()
        scores = _make_scores(batch)
        reference = _compile_reference()
        candidate = _compile_candidate()
        reference_values, reference_ids = reference(scores)
        candidate_values, candidate_ids = candidate(scores)
        torch.hpu.synchronize()

        reference_values_cpu = reference_values.cpu()
        candidate_values_cpu = candidate_values.cpu()
        reference_ids_cpu = reference_ids.cpu()
        candidate_ids_cpu = candidate_ids.cpu()
        torch.testing.assert_close(candidate_values_cpu, reference_values_cpu, rtol=0, atol=0)
        torch.testing.assert_close(
            candidate_ids_cpu.to(reference_ids_cpu.dtype).sort(dim=-1).values,
            reference_ids_cpu.sort(dim=-1).values,
            rtol=0,
            atol=0,
        )
        ids_order_exact = bool(torch.equal(candidate_ids_cpu, reference_ids_cpu))

        for _ in range(args.warmups):
            reference(scores)
            candidate(scores)
        torch.hpu.synchronize()

        samples: dict[str, list[tuple[float, float]]] = {"pytorch": [], "vendor_cguid": []}
        order = (("pytorch", reference), ("vendor_cguid", candidate))
        for wave in range(args.waves):
            wave_order = order if wave % 2 == 0 else tuple(reversed(order))
            for name, function in wave_order:
                samples[name].append(_measure(function, scores, args.wave_iterations))

        reference_stats = _stats(samples["pytorch"])
        candidate_stats = _stats(samples["vendor_cguid"])
        report["batches"][str(batch)] = {
            "pytorch": reference_stats,
            "vendor_cguid": candidate_stats,
            "device_speedup": reference_stats["device_median_ms"] / candidate_stats["device_median_ms"],
            "host_speedup": reference_stats["host_median_ms"] / candidate_stats["host_median_ms"],
            "candidate_set_exact": True,
            "candidate_order_exact": ids_order_exact,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
