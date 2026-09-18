# SPDX-License-Identifier: Apache-2.0
"""Qualify grouped prompt MoE followed by the unchanged N256 C1 operator."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--changing-routes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert
    from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_grouped_prefill import run_grouped_prefill

    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.set_num_threads(2)
    shard = PreparedV41Shard(args.prepared, 0, 0)

    def load(projection):
        qs, ss, cs = [], [], []
        for index in range(args.experts):
            q = read_expert(shard.catalog[f"layers.0.ffn.experts.{projection}_q16"], index)
            s = read_expert(shard.catalog[f"layers.0.ffn.experts.{projection}_s16"], index)
            pq, ps, pc, _ = prepare_expert(q, s)
            qs.append(pq)
            ss.append(ps)
            cs.append(pc)
        q = torch.from_numpy(np.stack(qs)).to("hpu")
        s = torch.from_numpy(np.stack(ss)).to("hpu")
        c = torch.from_numpy(np.stack(cs).view(np.int16)).view(torch.bfloat16).to("hpu")
        return q, s, c

    q13, s13, c13 = load("w13")
    q2, s2, c2 = load("w2")
    lookup = mxfp4_bf16_lut("hpu")
    torch.manual_seed(41)
    x = torch.randn(args.tokens, 5120, dtype=torch.bfloat16).to("hpu")
    ids_cpu = torch.stack([torch.randperm(args.experts)[:6] for _ in range(args.tokens)]).int()
    ids = ids_cpu.to("hpu")
    routing = (torch.rand(args.tokens, 6).softmax(-1) * 1.5).to("hpu")

    def c1(value, indices, route):
        return torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2(
            value, indices, route, q13, q2, s13, s2, lookup, c13, c2, True)

    decode = torch.compile(c1, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected_c1 = decode(x[:1], ids[:1], routing[:1]).cpu()
    print("native C1 compiled", flush=True)

    def candidate():
        return run_grouped_prefill(x, ids, routing, q13, q2, s13, s2, lookup, True)

    actual = candidate()
    torch.hpu.synchronize()
    print("grouped prefill compiled", flush=True)
    # Compare the same algorithm's grouped GEMM with the existing BF16 native
    # compound at the same real inputs, preserving clamp and routed W2 input.
    ref = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2(x[:1].contiguous(), ids[:1].contiguous(),
                                                                              routing[:1].contiguous(), q13, q2, s13,
                                                                              s2, lookup, True).cpu().float()
    test = actual[:1].cpu().float()
    error = (test - ref).abs()
    samples, devices = [], []
    resident_bytes = torch.hpu.memory_allocated()
    torch.hpu.reset_peak_memory_stats()
    for _ in range(args.repeat):
        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        start = time.perf_counter_ns()
        begin.record()
        actual = candidate()
        end.record()
        end.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        devices.append(begin.elapsed_time(end))
    actual_c1 = decode(x[:1], ids[:1], routing[:1]).cpu()
    report = {
        "experts": args.experts,
        "tokens": args.tokens,
        "host_ms": samples,
        "device_interval_ms": devices,
        "c1_unchanged_after_prefill": torch.equal(expected_c1, actual_c1),
        "bf16_reference_max_abs": error.max().item(),
        "bf16_reference_relative_l2": (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(ref)).item(),
        "finite": bool(torch.isfinite(actual.float()).all().cpu()),
        "peak_allocated_bytes": torch.hpu.max_memory_allocated(),
        "additional_peak_bytes": torch.hpu.max_memory_allocated() - resident_bytes,
    }
    if args.changing_routes:
        changed = []
        for skew in (False, True):
            next_ids = torch.zeros_like(ids) if skew else (ids + 1) % args.experts
            next_x = x * 0.75
            next_routing = routing.flip(-1).contiguous()
            output = run_grouped_prefill(next_x, next_ids, next_routing, q13, q2, s13, s2, lookup, True)
            reference = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2(
                next_x[:1].contiguous(), next_ids[:1].contiguous(), next_routing[:1].contiguous(), q13, q2, s13, s2,
                lookup, True).cpu().float()
            difference = output[:1].cpu().float() - reference
            relative = (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)).item()
            changed.append({"all_routes_one_expert": skew, "relative_l2": relative})
            if relative > 0.01:
                raise RuntimeError(f"Changed routing failed: {changed[-1]}")
        report["changing_routes"] = changed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["finite"] or not report["c1_unchanged_after_prefill"] or report["bf16_reference_relative_l2"] > 0.01:
        raise RuntimeError("Grouped prefill failed its component numerical gate")


if __name__ == "__main__":
    main()
