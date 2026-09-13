# SPDX-License-Identifier: Apache-2.0
"""Candidate-only complete MoE timing for prepared Q16/S16 weights."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from deepseek_v4_mxfp4_prepared_common import (
    comparison,
    compile_prepared_moe,
    make_prepared_inputs,
    native_reference,
    selected_native_inputs,
)

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-proof", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--launches", type=int, default=10)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.launches) < 1:
        parser.error("warmup, samples and launches must be positive")
    root = Path(__file__).resolve().parents[1]
    build = json.loads((root / "vllm_gaudi/lib/deepseek_v4_build.json").read_text())
    proof = json.loads(args.placement_proof.read_text())
    if not proof.get("qualified") or proof["native_build"]["binaries"] != build["binaries"]:
        parser.error("placement proof must cover the exact prepared candidate libraries")
    torch.ops.load_library(__import__("os").environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    candidate = compile_prepared_moe()
    native = torch.compile(native_reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    result = {
        "qualified": False,
        "torch": torch.__version__,
        "native_build": build,
        "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "placement_proof": str(args.placement_proof),
        "correctness": {},
        "timer": "host perf_counter + device drain after each group of launches",
        "baseline_timed": False,
        "runtime_injection": False,
        "arm": "prepared_candidate",
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    timed_inputs = None
    for label, sample in (("synthetic", None), ("loaded_sample", args.sample)):
        prepared, standard, ids, _ = make_prepared_inputs(sample)
        expected = native(*selected_native_inputs(standard, ids)).cpu()
        actual = candidate(*prepared, True).cpu()
        detail = comparison(actual, expected)
        result["correctness"][label] = detail
        torch.save(
            {"input": prepared[0].cpu(), "ids": ids, "router": prepared[2].cpu(),
             "candidate": actual, "reference": expected},
            args.output.parent / f"{label}-outputs.pt",
        )
        print(label, json.dumps(detail), flush=True)
        save()
        if not detail["exact"]:
            result["status"] = "arithmetic mismatch; stop before timing and E2E"
            save()
            raise SystemExit(2)
        if label == "synthetic":
            timed_inputs = prepared
        del standard
        if label != "synthetic":
            del prepared
        torch.hpu.synchronize()
    for _ in range(args.warmup):
        for _ in range(args.launches):
            candidate(*timed_inputs, True)
        torch.hpu.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        for _ in range(args.launches):
            candidate(*timed_inputs, True)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / args.launches)
    result["measurement"] = {
        "warmup_groups": args.warmup,
        "samples": args.samples,
        "launches_per_sample": args.launches,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    result["historical_baselines"] = {
        "indexed_build07_median_ms": 0.5322106000676285,
        "gathered_native_median_ms": 0.5318429000908509,
        "notice": "Reused archived comparable results; neither baseline was rerun.",
    }
    result["status"] = "prepared arithmetic and placement gates passed; candidate measured; E2E unqualified"
    save()
    print(json.dumps(result["measurement"], indent=2), flush=True)


if __name__ == "__main__":
    main()
