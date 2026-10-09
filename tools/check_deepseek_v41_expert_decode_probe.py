# SPDX-License-Identifier: Apache-2.0
"""Real one-layer FP4 read/ALU/store diagnosis; never a model speed claim."""
import argparse
import json
import os
from pathlib import Path
import time

from benchmark_deepseek_v41_projection_chains import summarize
from deepseek_v41_micro_replay import RecipeRecorder
import numpy as np
import torch

from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, saturated_decode_eligible
from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
from vllm_gaudi.ops.deepseek_v41_native_trace import NativeTrace, configure_post_graph_directory, scope
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def checksum(q, scales, table, decoded):
    blocks, stream = q.shape
    k = stream // 64
    packed = q.view(np.uint8).reshape(blocks, k, 128)
    values = np.empty((blocks, k, 256), np.uint8)
    values[..., 0::2], values[..., 1::2] = packed & 15, packed >> 4
    if decoded:
        groups = scales.view(np.uint8).reshape(blocks, k // 32 + 1, 256).astype(np.int16)
        delta = (groups[:, :-1] - groups[:, -1:]) * 8
        outside = (delta < -40) | (delta > 48)
        assert not np.any(np.repeat(outside, 32, axis=1) & (values != 0) & (values != 8))
        biased = np.where((table[:16] & 127) == 0, 0, table[:16].astype(np.int16) + 48)
        subtract = np.repeat(48 - delta, 32, axis=1)
        values = np.maximum(biased[values] - subtract, 0).astype(np.uint8)
    return np.bitwise_xor.reduce(values.reshape(blocks, k // 128, 128, 256), axis=2).transpose(1, 0, 2).reshape(
        k // 128, blocks * 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--modes", type=int, choices=range(4), nargs="+", default=list(range(4)))
    parser.add_argument("--trace", action="store_true", help="One missing phase trace, outside scored device timings")
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(20261004)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    logical_intermediate = config["moe_intermediate_size"] // shard.manifest["tensor_parallel_size"]
    recorder = RecipeRecorder(output)
    configure_post_graph_directory(output / "graphs/rank0")
    lookup = mxfp4_bf16_lut("hpu")
    table = lookup.cpu().view(torch.uint8).numpy()
    op = torch.ops.custom_op.custom_deepseek_v41_expert_decode_probe_gaudi2
    result = dict(status="running", layer=args.layer, routes=36, selected_real_experts=list(range(36)),
                  scope="read/checksum and decode/checksum share XOR cost; store+MME versus SAT+MME use real layout",
                  diagnostic_only=True, performance_qualified=False, results=[], checks=[])
    trace_plans = []
    with torch.inference_mode():
        for projection, pack in (("w13", 2), ("w2", 1)):
            prefix = f"layers.{args.layer}.ffn.experts.{projection}"
            qs, ss = (shard.catalog[prefix + suffix] for suffix in ("_q16", "_s16"))
            cpu = {}
            for expert in range(36):
                q, s = read_expert(qs, expert), read_expert(ss, expert)
                pq, ps, _, _ = prepare_expert(q, s, compact_scales=True)
                active_k = 5120 if projection == "w13" else logical_intermediate
                assert not np.any(pq[:, active_k * 64:])
                assert saturated_decode_eligible(ps, active_k=active_k)
                cpu[expert] = (pq, ps)
            pq, ps = cpu[0]
            q = torch.empty((qs.shape[0], *pq.shape), dtype=torch.int16, device="hpu")
            scales = torch.empty((qs.shape[0], *ps.shape), dtype=torch.int16, device="hpu")
            for expert, (pq, ps) in cpu.items():
                q[expert].copy_(torch.from_numpy(pq))
                scales[expert].copy_(torch.from_numpy(ps))
            ids = torch.arange(36, dtype=torch.int32, device="hpu")
            x = torch.randint(-4, 5, (36 // pack, 1, q.shape[2] // 64), dtype=torch.int32).float().div_(4).to(
                device="hpu", dtype=torch.float8_e4m3fn)
            compiled, replay = {}, {}
            for mode in (range(4) if args.trace else args.modes):
                def entry(ids, q, scales, lookup, x, mode=mode, pack=pack):
                    return op(ids, q, scales, lookup, x, mode, pack)
                compiled[mode] = torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
                arguments = (ids, q, scales, lookup, x)
                first = compiled[mode](*arguments)
                torch.hpu.synchronize()
                replay[mode] = recorder.prepare(compiled[mode], [ids], [(q, scales, lookup, x)])
                expected = first.cpu()
                replay[mode]()
                torch.hpu.synchronize()
                assert torch.equal(replay[mode].outputs[0].cpu(), expected)
            for mode in [m for m in args.modes if m < 2]:
                arrays = [checksum(*cpu[e], table, mode == 1) for e in range(36)]
                reference = np.stack([np.concatenate(arrays[a:a + pack], axis=1) for a in range(0, 36, pack)])
                actual = replay[mode].outputs[0].cpu().numpy()
                if mode == 0:
                    # The packed-load instruction supplies the useful nibble
                    # in bits 0..3; upper bits are not part of the FP4 code.
                    # Production's repeated-16 LUT likewise ignores them.
                    actual = actual & 15
                if not np.array_equal(actual, reference):
                    np.savez(output / f"{projection}-mode{mode}-mismatch.npz", actual=actual, reference=reference,
                             packed_q=cpu[0][0], scale_planes=cpu[0][1])
                    result.update(status="failed checksum contract; timings invalid")
                    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                    raise ValueError(
                        f"{projection}/mode{mode}: checksum mismatch, {np.count_nonzero(actual != reference)}")
                result["checks"].append(dict(projection=projection, mode=mode, checksum_exact=True))
            if 2 in replay:
                expected = x.float().sum(-1, keepdim=True).cpu().expand_as(replay[2].outputs[0].cpu())
                assert torch.equal(expected, replay[2].outputs[0].cpu()), "Constant-write MME result differs"
                result["checks"].append(dict(projection=projection, store_mme_exact=True))
            for round_id in range(args.rounds):
                order = args.modes if round_id % 2 == 0 else list(reversed(args.modes))
                route = torch.randperm(36, dtype=torch.int64).to(torch.int32).to("hpu")
                ids.copy_(route)
                x.copy_((x.float() + 0.25).to(torch.float8_e4m3fn))
                torch.hpu.synchronize()
                for mode in order:
                    for _ in range(16):
                        replay[mode]()
                    torch.hpu.synchronize()
                    samples, host_samples = [], []
                    for _ in range(args.samples):
                        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        started = time.perf_counter_ns()
                        begin.record()
                        replay[mode]()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end))
                        host_samples.append((time.perf_counter_ns() - started) / 1e6)
                    result["results"].append(dict(projection=projection, mode=mode, round=round_id,
                                                  timing=summarize(samples), device_samples_ms=samples,
                                                  synchronized_host_ms=summarize(host_samples),
                                                  fp4_read_bytes_per_layer=36 * cpu[0][0].nbytes,
                                                  fp8_output_bytes_per_layer=36 * cpu[0][0].nbytes * 2))
                    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            if args.trace:
                trace_plans.extend((projection, mode, plan) for mode, plan in replay.items())
            else:
                for plan in replay.values():
                    plan.close()
            del replay, compiled, q, scales
        if args.trace:
            trace = NativeTrace(output / "traces", scope_only=True)
            trace.start()
            for projection, mode, plan in trace_plans:
                with scope(f"expert_probe::{projection}::mode{mode}"):
                    for _ in range(8):
                        plan()
                    torch.hpu.synchronize()
            trace.stop()
            result["raw_trace_metadata"] = trace.metadata
            for _, _, plan in trace_plans:
                plan.close()
        result["status"] = "passed diagnostic; SRAM placement must be checked separately"
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(status=result["status"], checks=result["checks"], output=str(output))), flush=True)


if __name__ == "__main__":
    main()
