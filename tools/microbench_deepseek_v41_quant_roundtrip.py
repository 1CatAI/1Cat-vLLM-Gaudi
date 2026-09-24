# SPDX-License-Identifier: Apache-2.0
"""Validate independent group codecs through real Q/KV input consumers."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--compiled", action="store_true", help="Also qualify a combined compiler region")
    parser.add_argument("--skip-reference-timing", action="store_true")
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    import torch
    import torch.nn.functional as F
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.set_num_threads(1)
    torch.manual_seed(220932)
    torch.ops.load_library(str(args.library))
    old = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
    wide = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2
    report = dict(scope="group32 activation codec -> real fused Q/KV projection -> Q and KV RMSNorm",
                  tokens=args.tokens,
                  compiled=args.compiled,
                  checks=[],
                  timings={})

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    def compare(a, b):
        a, b = a.cpu(), b.cpu()
        aa, bb = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        return dict(exact=torch.equal(aa, bb),
                    different=int((aa != bb).sum()),
                    max_abs_finite=float(torch.nan_to_num((a.float() - b.float()).abs(), nan=0).max()))

    # Adjacent groups must not share maxima. All BF16 bit patterns include
    # signed zero, subnormals, infinities and NaN payloads.
    bits = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
    cases = [("all_bf16_codes", bits.reshape(512, 128)),
             ("all_bf16_codes_permuted", bits[torch.randperm(65536)].reshape(512, 128))]
    for width in (32, 64, 96, 128, 160, 384, 416, 1280, 5120):
        x = torch.randn(7, width).reshape(7, -1, 32)
        scales = torch.exp2((torch.arange(width // 32) % 40 - 20).float())
        cases.append((f"tails_and_group_scale_K{width}", (x * scales[None, :, None]).reshape(7, width).bfloat16()))
    # Exercise the exponent increment at mantissa 96/97 and the rounding
    # midpoints on every small FP8 grid, with independent neighboring groups.
    boundaries = []
    for exponent in range(113, 256):
        for mantissa in (0, 96, 97, 127):
            if exponent == 255 and mantissa:
                continue
            maximum = (exponent << 7) | mantissa
            for distance in (15, 16, 17, 18, 19):
                fractions = (0, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 111, 112, 113, 127)
                small = [((exponent - distance) << 7) | fraction for fraction in fractions]
                boundaries.extend([maximum, *small, *(value | 0x8000 for value in small[:10]), 0x8000])
    boundary_bits = torch.tensor(boundaries, dtype=torch.int32).short().view(torch.bfloat16)
    cases.append(("scale_and_tiny_rounding_boundaries", boundary_bits.reshape(-1, 32)))
    with torch.inference_mode():
        for name, source in cases:
            x = source.to("hpu")
            check = compare(wide(x), old(x))
            report["checks"].append(dict(name=name, **check))
            save()
            if not check["exact"]:
                raise RuntimeError(f"Wide group codec changed {name}: {check}")
        shard = PreparedV41Shard(args.prepared, args.layer // 20, 0)
        prefix = f"layers.{args.layer}.attn."
        q = shard.dense(prefix + "wq_a.weight", "cpu")
        kv = shard.dense(prefix + "wkv.weight", "cpu")
        q_width = q.shape[0]
        weight_cpu = torch.cat((q, kv)).contiguous()
        weight = weight_cpu.to("hpu")
        q_norm = shard.tensor(prefix + "q_norm.weight", "hpu")
        kv_norm = shard.tensor(prefix + "kv_norm.weight", "hpu")
        epsilon = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
        report["weight_sha256"] = hashlib.sha256(weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()

        def reference(x, w, qn, kn):
            result = F.linear(old(x), w)
            return rms_norm(result[:, :q_width], qn, epsilon), rms_norm(result[:, q_width:], kn, epsilon)

        def candidate(x, w, qn, kn):
            result = F.linear(wide(x), w)
            return rms_norm(result[:, :q_width], qn, epsilon), rms_norm(result[:, q_width:], kn, epsilon)

        programs = {"reference": reference, "candidate": candidate}
        if args.compiled:
            programs = {
                name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                for name, fn in programs.items()
            }
        inputs = []
        for change in range(2):
            x = torch.randn(args.tokens, weight.shape[1]).bfloat16()
            # Distinct contiguous request segments, different values and order.
            if change:
                x = x.roll(args.tokens // 2, 0)
            x = x.to("hpu")
            inputs.append(x)
            expected = programs["reference"](x, weight, q_norm, kv_norm)
            observed = programs["candidate"](x, weight, q_norm, kv_norm)
            checks = [compare(a, b) for a, b in zip(observed, expected)]
            report["checks"].append(dict(name=f"complete_chain_change{change}", outputs=checks))
            save()
            if not all(check["exact"] for check in checks):
                raise RuntimeError("Wide codec changed the Q/KV consumer")
        if args.checks_only:
            return
        for fn in programs.values():
            for x in inputs:
                fn(x, weight, q_norm, kv_norm)
        torch.hpu.synchronize()
        before = dict(metric_global("graph_compilation").stats())
        for name in (("candidate", ) if args.skip_reference_timing else programs):
            samples = []
            for iteration in range(args.repeats):
                start, end = (torch.hpu.Event(enable_timing=True) for _ in range(2))
                torch.hpu.reset_peak_memory_stats()
                resident = torch.hpu.memory_allocated()
                wall = time.perf_counter_ns()
                start.record()
                result = programs[name](inputs[iteration % 2], weight, q_norm, kv_norm)
                end.record()
                end.synchronize()
                samples.append(
                    dict(device_ms=start.elapsed_time(end),
                         wall_ms=(time.perf_counter_ns() - wall) / 1e6,
                         additional_peak_bytes=torch.hpu.max_memory_allocated() - resident))
                del result
            report["timings"][name] = dict(samples=samples,
                                           median_device_ms=statistics.median(s["device_ms"] for s in samples),
                                           median_wall_ms=statistics.median(s["wall_ms"] for s in samples))
        report["compilation_before"] = before
        report["compilation_after"] = dict(metric_global("graph_compilation").stats())
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
