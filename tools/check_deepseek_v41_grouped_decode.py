# SPDX-License-Identifier: Apache-2.0
"""Real-weight local routed-MoE chain gate for 4/8/16-row grouping.

Includes device route grouping, existing quantization, both matrices, restore,
ordered reduction and mHC consumption. This first gate has no TP exchange and
is not an end-to-end or production-stage qualification.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, choices=(1, 4, 8, 32, 64), default=8)
    p.add_argument("--rows", type=int, choices=(1, 4, 8, 16), default=4)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--compiled-only",
                   action="store_true",
                   help="Use the same ordinary compiled submission for both arms, without duplicate capture workspace")
    p.add_argument("--capture-candidate",
                   action="store_true",
                   help="After comparison, qualify the candidate's native capture workspace and changing inputs")
    p.add_argument("--capture-only", action="store_true", help="Reuse archived timing and check only native ownership")
    args = p.parse_args()
    if args.capture_only:
        args.compiled_only = args.capture_candidate = True
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_grouped_decode import grouped_decode
    from vllm_gaudi.ops.deepseek_v41_math import hc_post
    from vllm_gaudi.ops.deepseek_v41_route_blocks import device_route_blocks
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    torch.set_num_threads(1)
    torch.manual_seed(8321)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    report = dict(scope=__doc__,
                  batch=args.batch,
                  rows=args.rows,
                  cases=[],
                  qualified=False,
                  execution="ordinary_compiled" if args.compiled_only else "native_component")
    save = lambda: args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    shard = PreparedV41Shard(args.prepared, 0, 0)
    specs = {k: v for k, v in shard.specs.items() if k.startswith("layers.0.ffn.experts.")}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs)
    weights = tree.layers.get_submodule("0").ffn.experts
    operands = (weights.w13_q16, weights.w2_q16, weights.w13_s16, weights.w2_s16, mxfp4_bf16_lut("hpu"),
                weights.w13_fp8_channel, weights.w2_fp8_channel)
    b = args.batch
    value = torch.randn(b, 5120).bfloat16().to("hpu")
    ids = torch.randint(0, 384, (b, 6), dtype=torch.int32, device="hpu")
    route = torch.softmax(torch.randn(b, 6), -1).to("hpu") * 1.5
    residual = torch.randn(b, 4, 5120).bfloat16().to("hpu")
    post, comb = torch.rand(b, 4).to("hpu"), torch.rand(b, 4, 4).to("hpu")

    def reference(x, expert_ids, routing, old, p, c, q13, q2, s13, s2, lut, c13, c2):
        routed = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2(
            x, expert_ids, routing, q13, q2, s13, s2, lut, c13, c2, True)
        return hc_post(routed, old, p, c)

    def candidate(x, expert_ids, routing, old, p, c, q13, q2, s13, s2, lut, c13, c2):
        routed = grouped_decode(x, expert_ids, routing, q13, q2, s13, s2, lut, c13, c2, rows=args.rows)
        return hc_post(routed, old, p, c)

    recorder = RecipeRecorder(args.output.parent)
    params = (ids, route, residual, post, comb, *operands)
    plans, programs = {}, {}

    class Ordinary:

        def __init__(self, fn):
            self.fn, self.outputs, self.recipes = fn, [], None

        def __call__(self):
            self.outputs = [self.fn(value, *params)]

        def close(self):
            pass

    for name, fn in ((("grouped", candidate), ) if args.capture_only else
                     (("direct", reference), ("grouped", candidate))):
        renamed = FunctionType(fn.__code__.replace(co_name=f"grouped_gate_{name}"),
                               fn.__globals__,
                               argdefs=fn.__defaults__,
                               closure=fn.__closure__)
        programs[name] = torch.compile(renamed, backend="hpu_backend", fullgraph=True, dynamic=False)
        programs[name](value, *params)
        plans[name] = (Ordinary(programs[name]) if args.compiled_only else recorder.prepare(
            programs[name], [value], [params]))
    report["allocated_after_prepare_bytes"] = torch.hpu.memory_allocated()
    report["recipes"] = {name: plan.recipes for name, plan in plans.items()}
    save()
    for distribution in (() if args.capture_only else ("uniform", "concentrated", "all_same", "unique", "uniform")):
        value.copy_(torch.randn(b, 5120).bfloat16())
        host_ids = torch.stack([torch.randperm(384)[:6] for _ in range(b)]).int()
        if distribution == "concentrated":
            host_ids = host_ids.remainder(12)
        elif distribution == "all_same":
            host_ids.zero_()
        elif distribution == "unique":
            host_ids = torch.arange(b * 6, dtype=torch.int32).reshape(b, 6)
        ids.copy_(host_ids)
        route.copy_(torch.softmax(torch.randn(b, 6), -1) * 1.5)
        desc = (None,
                torch.arange(b * 6).reshape(-1, 1)) if args.rows == 1 else device_route_blocks(host_ids, rows=args.rows)
        case = dict(distribution=distribution,
                    real_routes=b * 6,
                    active_groups=int((desc[1][:, 0] >= 0).sum()),
                    capacity_groups=desc[1].shape[0],
                    padded_rows=desc[1].numel() - b * 6,
                    mme_empty_groups_skipped=False,
                    real_router_distribution=False)
        # Synthetic routes deliberately stress ownership with real weights;
        # a frozen production router distribution remains a later gate.
        expected = programs["direct"](value, *params).cpu()
        results = {}
        for name, plan in plans.items():
            plan()
            # Diagnostic Compute intentionally has no serving completion
            # registration. Join before a DMA read of its retained outputs.
            torch.hpu.synchronize()
            actual = plan.outputs[0].cpu()
            difference = actual.float() - expected.float()
            results[name] = dict(exact=torch.equal(expected, actual),
                                 max_abs=float(difference.abs().max()),
                                 relative_l2=float(difference.norm() / expected.float().norm()))
        case["checks"] = results
        report["cases"].append(case)
        save()
        if not all(x["exact"] for x in results.values()):
            torch.save(
                dict(expected=expected,
                     grouped=plans["grouped"].outputs[0].cpu(),
                     input=value.cpu(),
                     ids=host_ids,
                     route=route.cpu()), args.output.with_suffix(".failure.pt"))
            raise RuntimeError("Grouped N256 changed outputs; investigate before performance qualification")
        for name, plan in plans.items():
            for _ in range(3):
                plan()
            torch.hpu.synchronize()
            samples = []
            for _ in range(args.samples):
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                wall = time.perf_counter()
                begin.record()
                plan()
                end.record()
                end.synchronize()
                samples.append(dict(device_ms=begin.elapsed_time(end), wall_ms=(time.perf_counter() - wall) * 1000))
            case[name] = dict(samples=samples,
                              median_device_ms=statistics.median(x["device_ms"] for x in samples),
                              median_wall_ms=statistics.median(x["wall_ms"] for x in samples))
        save()
        print(json.dumps({
            k: v
            for k, v in case.items() if k not in ("direct", "grouped")
        } | {
            "direct_ms": case["direct"]["median_device_ms"],
            "grouped_ms": case["grouped"]["median_device_ms"]
        }),
              flush=True)
    report["component_exact"] = None if args.capture_only else True
    if args.capture_candidate:
        native = recorder.prepare(programs["grouped"], [value], [params])
        workspace = native.native.workspace_bytes()
        checks = []
        report["native_candidate"] = dict(workspace_bytes=workspace, checks=checks, info=native.native.info())
        save()
        if workspace > 2 * 2**30:
            raise RuntimeError("Candidate native workspace exceeds the 2 GiB per-rank budget")
        for change in range(3):
            value.copy_(torch.randn(b, 5120).bfloat16())
            ids.copy_(torch.stack([torch.randperm(384)[:6] for _ in range(b)]).int())
            expected = programs["grouped"](value, *params).cpu()
            native()
            torch.hpu.synchronize()
            same = torch.equal(native.outputs[0].cpu(), expected)
            checks.append(dict(change=change, exact=same))
            save()
            if not same:
                raise RuntimeError("Candidate native replay differs from its ordinary compiled algorithm")
        native.close()
    report["peak_allocated_bytes"] = torch.hpu.max_memory_allocated()
    for plan in plans.values():
        plan.close()
    torch.distributed.destroy_process_group()
    save()


if __name__ == "__main__":
    main()
