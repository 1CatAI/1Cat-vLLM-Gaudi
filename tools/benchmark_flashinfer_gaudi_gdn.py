# SPDX-License-Identifier: Apache-2.0
"""Compare FlashInfer-Gaudi GDN tactics on production Qwen decode shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flashinfer_gaudi import get_capabilities
from flashinfer_gaudi._native import bridge_packed_gdn_op, public_packed_gdn_op
from flashinfer_gaudi._reference import packed_recurrent_decode
from flashinfer_gaudi.gdn_decode import BackendUnavailableError, _call_native_packed


def _synchronize() -> None:
    torch.hpu.synchronize()


def _make_inputs(batch: int):
    generator = torch.Generator().manual_seed(31 + batch)
    key_heads, value_heads, dim = 16, 48, 128
    width = 2 * key_heads * dim + value_heads * dim
    packed = torch.randn(batch, width, dtype=torch.bfloat16, generator=generator).to("hpu")
    log_decay = (-torch.rand(batch, value_heads, dtype=torch.float32, generator=generator)).to("hpu")
    beta = torch.sigmoid(torch.randn(batch, value_heads, dtype=torch.float32,
                                     generator=generator)).to(torch.bfloat16).to("hpu")
    state = torch.randn(batch + 1, value_heads, dim, dim, dtype=torch.float32, generator=generator).to("hpu")
    indices = torch.arange(1, batch + 1, dtype=torch.int32).to("hpu")
    return packed, log_decay, beta, state, indices


def _compile(policy: str):
    if policy == "pytorch":

        def run(packed, log_decay, beta, state, indices):
            return packed_recurrent_decode(
                packed,
                log_decay,
                beta,
                state,
                indices,
                indices,
                None,
                True,
            )[0]

    else:
        op = public_packed_gdn_op() if policy == "public" else bridge_packed_gdn_op()
        if op is None:
            raise BackendUnavailableError(f"The {policy} backend is not loaded.")

        def run(packed, log_decay, beta, state, indices):
            return _call_native_packed(op, packed, log_decay, beta, state, indices)

    return torch.compile(run, backend="hpu_backend", fullgraph=True)


def _measure(fn, inputs, warmups: int, iterations: int) -> dict[str, float]:
    for _ in range(warmups):
        fn(*inputs)
    _synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs)
        _synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,16,32")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--backend", choices=("public", "bridge"), default="public")
    args = parser.parse_args()

    results: dict[str, object] = {"capabilities": get_capabilities(), "batches": {}}
    for batch in (int(item) for item in args.batches.split(",") if item):
        reference_inputs = _make_inputs(batch)
        candidate_inputs = tuple(tensor.clone() for tensor in reference_inputs)
        reference = _compile("pytorch")
        candidate = _compile(args.backend)

        reference_output = reference(*reference_inputs)
        candidate_output = candidate(*candidate_inputs)
        _synchronize()
        candidate_output_cpu = candidate_output.cpu()
        reference_output_cpu = reference_output.cpu()
        candidate_state_cpu = candidate_inputs[3].cpu()
        reference_state_cpu = reference_inputs[3].cpu()
        output_max_abs = (candidate_output_cpu.float() - reference_output_cpu.float()).abs().max().item()
        state_max_abs = (candidate_state_cpu - reference_state_cpu).abs().max().item()
        # The TPC reduction tree is not bit-identical to the graph-compiler
        # reduction. Keep a tight state gate and a BF16-appropriate output
        # gate; model-level token validation remains required for promotion.
        torch.testing.assert_close(candidate_output_cpu, reference_output_cpu, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(candidate_state_cpu, reference_state_cpu, atol=2e-5, rtol=2e-4)

        reference_stats = _measure(reference, reference_inputs, args.warmups, args.iterations)
        candidate_stats = _measure(candidate, candidate_inputs, args.warmups, args.iterations)
        results["batches"][str(batch)] = {
            "pytorch": reference_stats,
            args.backend: candidate_stats,
            "median_speedup": reference_stats["median_ms"] / candidate_stats["median_ms"],
            "output_max_abs": output_max_abs,
            "state_max_abs": state_max_abs,
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
