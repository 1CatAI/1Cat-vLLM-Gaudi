# SPDX-License-Identifier: Apache-2.0
"""Real-weight Q/KV and shared-expert FP8 screen before stage integration."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 9, 14])
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--native-scaled-gemm", action="store_true")
    parser.add_argument("--block-scaled-gemm", action="store_true")
    parser.add_argument("--check-tokens", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--reuse-ffn-prequant", action="store_true")
    parser.add_argument("--parts", choices=("qkv", "shared"), nargs="+", default=["qkv", "shared"])
    parser.add_argument("--fused-shared-silu", action="store_true")
    parser.add_argument("--direct-rne-input", action="store_true")
    parser.add_argument("--tail-check-only", action="store_true")
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(args.prepared)
    import habana_frameworks.torch.core  # noqa: F401
    import numpy as np
    import torch
    import torch.nn.functional as F
    from deepseek_v41_micro_replay import RecipeRecorder
    from prepare_deepseek_v41_woa_fp8 import read_bytes
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import (
        prepare_block32_rows, covering_scale, encode_gaudi2, decode_e4m3fn, decode_gaudi2)

    torch.set_num_threads(1)
    torch.hpu.set_device(0)
    bind_worker_cpu(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    out = Path(os.environ["DSV41_RUN_EVIDENCE"])
    recorder = RecipeRecorder(out)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    report = {"scope": "single-operator/local shared-expert screen; no stage or end-to-end gain",
              "tp_size": shard.tensor_parallel_size, "layers": args.layers, "weights": [], "numerics": [],
              "timings": [], "native_scaled_gemm": args.native_scaled_gemm,
              "block_scaled_gemm": args.block_scaled_gemm, "reuse_ffn_prequant": args.reuse_ffn_prequant,
              "fused_shared_silu": args.fused_shared_silu, "direct_rne_input": args.direct_rne_input}

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def metrics(x, reference):
        x, reference = x.float().cpu(), reference.float().cpu()
        delta = x - reference
        return {"finite": bool(torch.isfinite(x).all()), "max_abs": float(delta.abs().max()),
                "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-30)),
                "cosine": float(F.cosine_similarity(x.flatten(), reference.flatten(), dim=0))}

    def load(prefix):
        name = prefix + ".weight"
        weight, scale = shard.catalog[name], shard.catalog[prefix + ".scale"]
        n, k = weight.shape
        assert n % 32 == k % 32 == 0 and scale.shape == (n // 32, k // 32)
        codes = read_bytes(weight, 0, n * k).reshape(n, k)
        powers = read_bytes(scale, 0, n // 32 * (k // 32)).reshape(n // 32, k // 32)
        q, s, audit = prepare_block32_rows(codes, powers)
        if args.block_scaled_gemm:
            # Preserve every block's scale. Halve checkpoint codes so its
            # 448-range encoding fits Gaudi2's normal finite FP8 range.
            values = decode_e4m3fn(codes)
            q = encode_gaudi2(values * .5)
            restored = decode_gaudi2(q) * 2
            error = restored.astype(np.float64) - values
            audit.update(block_scale_preserved=True, changed=int(np.count_nonzero(restored != values)),
                         error_energy=float(np.sum(error ** 2)), source_energy=float(np.sum(values ** 2)))
            s = np.ldexp(np.ones_like(powers, dtype=np.float32), powers.astype(np.int32) - 126)
        report["weights"].append({"tensor": name, "shape": [n, k], "scale_shape": list(scale.shape), **audit})
        return (shard.dense(name, "cpu"), torch.from_numpy(q.copy()).view(torch.float8_e4m3fn),
                torch.from_numpy(s.T.copy()))

    with torch.inference_mode():
        # A nonzero tail detects omission of the final 64 columns at TP4 K=576.
        tail_checks = []
        for tokens in args.check_tokens:
            x = torch.zeros(tokens, 576, dtype=torch.bfloat16, device="hpu")
            # Cover both conversion halves and a nonuniform final vector.
            tail = torch.arange(64, dtype=torch.float32).reshape(1, -1) + 1
            rows = torch.arange(1, tokens + 1, dtype=torch.float32).reshape(-1, 1)
            x[:, 512:] = (tail * rows * .625).bfloat16().to("hpu")
            q, scale = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(x)
            cpu = x.cpu().float().numpy()
            s = covering_scale(np.abs(cpu).max(axis=1, keepdims=True))
            expected = encode_gaudi2(cpu / s)
            assert np.array_equal(q.cpu().view(torch.uint8).numpy(), expected)
            assert np.array_equal(scale.cpu().numpy(), s)
            tail_checks.append({"tokens": tokens, "tail_exact": True, "scale_exact": True})
        report["C1_C6_quant_tail_checks"] = tail_checks
        if args.tail_check_only:
            report["status"] = "passed_tail_contract_no_performance_measurement"
            save()
            torch.distributed.destroy_process_group()
            return

        inputs, old_qkv, new_qkv, old_shared, new_shared = [], [], [], [], []
        for layer in args.layers:
            q, kv = [load(f"layers.{layer}.attn.{p}") for p in ("wq_a", "wkv")]
            w1, w3, w2 = [load(f"layers.{layer}.ffn.shared_experts.{p}") for p in ("w1", "w3", "w2")]
            old_qkv.append((torch.cat((q[0], kv[0])).to("hpu"),))
            new_qkv.append((torch.cat((q[1], kv[1])).to("hpu"), torch.cat((q[2], kv[2]), dim=1).to("hpu")))
            old_shared.append((torch.cat((w1[0], w3[0])).to("hpu"), w2[0].to("hpu")))
            new_shared.append((torch.cat((w1[1], w3[1])).to("hpu"),
                               torch.cat((w1[2], w3[2]), dim=1).to("hpu"), w2[1].to("hpu"), w2[2].to("hpu")))
            if args.fused_shared_silu:
                width = w1[1].shape[0]
                padded = (width + 127) // 128 * 128
                assert padded == 640, "The existing diagnostic SiLU entry accepts width640"
                def padded_gate(weight):
                    return torch.cat((weight, torch.zeros(padded - width, weight.shape[1], dtype=weight.dtype)))
                gate = torch.cat((padded_gate(w1[1]), padded_gate(w3[1]))).to("hpu")
                scales = torch.cat((F.pad(w1[2], (0, padded - width), value=1),
                                    F.pad(w3[2], (0, padded - width), value=1)), dim=1)
                channel = scales.bfloat16().reshape(1, padded * 2 // 256, 256).to("hpu")
                down = torch.cat((w2[1], torch.zeros(w2[1].shape[0], padded - width, dtype=w2[1].dtype)), dim=1)
                new_shared[-1] = (gate, channel, down.to("hpu"), w2[2].to("hpu"))
            if args.reuse_ffn_prequant:
                norm = shard.tensor(f"layers.{layer}.ffn_norm.weight", "hpu")
                old_shared[-1] = (*old_shared[-1], norm)
                new_shared[-1] = (*new_shared[-1], norm)
            generator = torch.Generator().manual_seed(81277 + layer)
            inputs.append(torch.randn(1, 5120, generator=generator).bfloat16().to("hpu"))
        save()

        def reference_qkv(x, weight):
            return F.linear(quantize_activation(x), weight)

        def fp8_project(x, weight, scale):
            if not args.direct_rne_input:
                x = quantize_activation(x)
            if args.native_scaled_gemm or args.block_scaled_gemm:
                q, sx = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(x)
                if args.block_scaled_gemm:
                    sx = sx.expand(-1, x.shape[1] // 32).contiguous()
                return torch.ops.hpu.fp8_gemm_v2(q, False, weight, True, None, torch.bfloat16, sx, scale, None, False)
            return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(x, weight, scale)

        def candidate_qkv(x, weight, scale):
            return fp8_project(x, weight, scale)

        def middle(projected, dtype):
            gate, up = projected.float().chunk(2, dim=-1)
            return (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(dtype)

        def reference_shared(x, gate_up, down, norm=None):
            if args.reuse_ffn_prequant:
                x, _, _ = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, norm, 1e-6)
            projected = F.linear(quantize_activation(x), gate_up)
            return F.linear(quantize_activation(middle(projected, x.dtype)), down)

        def candidate_shared(x, gate_up, gate_scale, down, down_scale, norm=None):
            if args.fused_shared_silu:
                if args.reuse_ffn_prequant:
                    x, _, _ = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, norm, 1e-6)
                if not args.direct_rne_input:
                    x = quantize_activation(x)
                q, sx = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(x)
                product = torch.ops.hpu.fp8_gemm_v2(q, False, gate_up, True, None, torch.float32,
                                                   None, None, None, False)
                rows = x.shape[0]
                ids = torch.zeros(1, rows, device=x.device, dtype=torch.int32)
                route = torch.ones(1, rows, device=x.device, dtype=torch.float32)
                qm, sm = torch.ops.custom_op.custom_deepseek_v41_shared_silu_quant_gaudi2(
                    product.reshape(rows, 1, -1), ids, sx, gate_scale, route)
                return torch.ops.hpu.fp8_gemm_v2(qm.reshape(rows, -1), False, down, True, None,
                                                torch.bfloat16, sm.reshape(rows, 1), down_scale, None, False)
            if args.reuse_ffn_prequant:
                x, q, sx = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, norm, 1e-6)
                projected = torch.ops.hpu.fp8_gemm_v2(q, False, gate_up, True, None, torch.bfloat16,
                                                     sx, gate_scale, None, False)
            else:
                projected = fp8_project(x, gate_up, gate_scale)
            return fp8_project(middle(projected, x.dtype), down, down_scale)

        arms = (("qkv", reference_qkv, candidate_qkv, old_qkv, new_qkv),
                ("shared", reference_shared, candidate_shared, old_shared, new_shared))
        for name, reference, candidate, old, new in arms:
            if name not in args.parts:
                continue
            compiled = {"baseline": torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False),
                        "candidate": torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)}
            for tokens in args.check_tokens:
                x = inputs[0].expand(tokens, -1).contiguous()
                x *= torch.arange(1, tokens + 1, device="hpu").reshape(-1, 1).to(x.dtype)
                numerical = metrics(compiled["candidate"](x, *new[0]), compiled["baseline"](x, *old[0]))
                report["numerics"].append({"part": name, "tokens": tokens, **numerical})
                save()
                assert numerical["finite"] and numerical["relative_l2"] <= .03, numerical
            plans = {arm: recorder.prepare(fn, inputs, old if arm == "baseline" else new)
                     for arm, fn in compiled.items()}
            load = subprocess.check_output(["hl-smi", "--query-aip=module_id,memory.used,utilization.aip",
                                            "--format=csv,noheader,nounits"], text=True)
            report.setdefault("machine_load_at_timing", []).append({"part": name, "all_modules_csv": load})
            samples = {arm: [] for arm in plans}
            for trial in range(4):
                order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                for arm in order:
                    for _ in range(16):
                        plans[arm]()
                    torch.hpu.synchronize()
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        plans[arm]()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end) / args.iterations / len(inputs)
                    samples[arm].append(ms)
                    print(json.dumps({"part": name, "arm": arm, "trial": trial, "device_ms_per_layer": ms}), flush=True)
            medians = {arm: statistics.median(ms) for arm, ms in samples.items()}
            report["timings"].append({"part": name, "samples_ms_per_layer": samples, "medians_ms_per_layer": medians,
                                      "difference_ms_per_layer": medians["baseline"] - medians["candidate"]})
            save()
            for plan in plans.values():
                plan.close()
        report["status"] = "passed_single_operator_screen"
        report["estimated_16layer_compute_ms"] = 16 * sum(x["difference_ms_per_layer"] for x in report["timings"])
        save()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
