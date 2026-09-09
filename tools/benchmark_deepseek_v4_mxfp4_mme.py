# SPDX-License-Identifier: Apache-2.0
"""Candidate-only complete MoE timing after placement and arithmetic gates.

The default measures only the changed candidate. Explicit baseline timing
requires a recorded reason and saved correctness output. Historical timings
remain descriptive until their complete comparison fingerprint is verified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from check_deepseek_v4_mxfp4_mme import make_inputs
from vllm_gaudi.ops.deepseek_v4_mxfp4 import normal_e8m0_scales

import torch


def native_reference(x, ids, router, w13, w2, s13, s2):
    return torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights(
        x, ids, router, w13, w2, s13, s2, block_size=32, permuted_weights=True,
        activation="silu", experts_min=0, experts_max=5, is_fp4=True, chunk_size=0, total_experts=0)


def gathered_native(x, ids, router, w13, w2, s13, s2):
    gathered = [t.index_select(0, ids.reshape(-1)) for t in (w13, w2, s13, s2)]
    selected = [tuple(t[i] for i in range(6)) for t in gathered]
    local_ids = torch.arange(6, dtype=torch.int32, device=x.device).view(1, 6)
    return native_reference(x, local_ids, router, *selected)


def comparison(actual, reference):
    a, b = actual.float(), reference.float()
    return {"exact": torch.equal(actual, reference),
            "nonzero": int(torch.count_nonzero(a - b)),
            "max_abs": float((a - b).abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)),
            "cosine": float(torch.nn.functional.cosine_similarity(a, b).item())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-proof", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--arm", choices=("candidate", "baseline"), default="candidate")
    parser.add_argument("--baseline-reason-file", type=Path)
    parser.add_argument("--saved-reference", type=Path)
    parser.add_argument("--normal-scales", action="store_true")
    args = parser.parse_args()
    if args.arm == "baseline" and (not args.baseline_reason_file or not args.saved_reference):
        parser.error("baseline measurement requires a recorded reason and saved correctness reference")
    if min(args.warmup, args.samples, args.launches) < 1:
        parser.error("warmup, samples and launches must be positive")
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select free physical modules explicitly")
    root = Path(__file__).resolve().parents[1]
    build = json.loads((root / "vllm_gaudi/lib/deepseek_v4_build.json").read_text())
    proof = json.loads(args.placement_proof.read_text())
    if not proof.get("qualified") or proof["native_build"]["binaries"] != build["binaries"]:
        parser.error("placement proof must cover the exact candidate native libraries")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    op = torch.ops.custom_op.custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2
    candidate = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    native = torch.compile(native_reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    result = {"qualified": False, "torch": torch.__version__, "native_build": build,
              "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
              "placement_proof": str(args.placement_proof), "correctness": {},
              "timer": "host perf_counter + device drain after each group of launches",
              "baseline_timed": args.arm == "baseline", "runtime_injection": False, "arm": args.arm}

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    synthetic = None
    synthetic_normal = False
    cases = (("synthetic", None), ("loaded_sample", args.sample)) if args.arm == "candidate" else ()
    for label, sample in cases:
        inputs, ids, _ = make_inputs(sample)
        normal = args.normal_scales and normal_e8m0_scales(*inputs[5:7])
        if sample is None:
            synthetic = inputs
            synthetic_normal = normal
        selected = [tuple(t[expert] for expert in ids.flatten().tolist()) for t in inputs[3:]]
        local_ids = torch.arange(6, dtype=torch.int32, device="hpu").view(1, 6)
        expected = native(inputs[0], local_ids, inputs[2], *selected).cpu()
        actual = candidate(*inputs, normal).cpu()
        detail = comparison(actual, expected)
        detail["normal_scales"] = normal
        result["correctness"][label] = detail
        torch.save({"input": inputs[0].cpu(), "ids": ids, "router": inputs[2].cpu(),
                    "candidate": actual, "reference": expected}, args.output.parent / (label + "-outputs.pt"))
        print(label, json.dumps(detail), flush=True)
        save()
        if not detail["exact"]:
            result["status"] = "arithmetic mismatch; stop before timing and E2E"
            save()
            raise SystemExit(2)
    measured = candidate
    if args.arm == "baseline":
        result["baseline_reason"] = args.baseline_reason_file.read_text()
        synthetic, _, _ = make_inputs()
        measured = torch.compile(gathered_native, backend="hpu_backend", fullgraph=True, dynamic=False)
        expected = torch.load(args.saved_reference, map_location="cpu", weights_only=True)["reference"]
        detail = comparison(measured(*synthetic).cpu(), expected)
        result["correctness"]["saved_reference"] = detail
        save()
        if not detail["exact"]:
            raise SystemExit("Baseline correctness failed; no timing")
    timed_inputs = [*synthetic, synthetic_normal] if args.arm == "candidate" else synthetic
    for _ in range(args.warmup):
        for _ in range(args.launches):
            measured(*timed_inputs)
        torch.hpu.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        for _ in range(args.launches):
            measured(*timed_inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / args.launches)
    result["measurement"] = {"warmup_groups": args.warmup, "samples": args.samples,
                             "launches_per_sample": args.launches, "samples_ms": samples,
                             "median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
                             "min_ms": min(samples), "max_ms": max(samples)}
    if args.history:
        history = json.loads(args.history.read_text())
        result["historical_reference"] = {"path": str(args.history),
            "gathered_native_median_ms": history["gathered_native_moe_single_graph"]["median_ms"],
            "notice": "No baseline remeasurement; verify external fingerprint before claiming a speedup."}
    result["status"] = "local arithmetic gates passed and selected arm measured; E2E unqualified"
    save()
    print(json.dumps(result["measurement"], indent=2), flush=True)


if __name__ == "__main__":
    main()
