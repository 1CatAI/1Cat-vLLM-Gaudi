# SPDX-License-Identifier: Apache-2.0
"""Measure group-parallel wo_a emission through the real wo_b projection."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--woa-sidecar", type=Path, required=True)
    p.add_argument("--dense-sidecar", type=Path, required=True)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--skip-reference-timing", action="store_true")
    p.add_argument("--checks-only", action="store_true")
    p.add_argument("--compiled", action="store_true")
    args = p.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    import torch
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar

    torch.set_num_threads(1)
    torch.manual_seed(220934)
    torch.ops.load_library(str(args.library))
    shard = PreparedV41Shard(args.prepared, args.layer // 20, 0)
    a = WoaFP8Sidecar(args.woa_sidecar, shard)
    b = DenseFP8Sidecar(args.dense_sidecar, shard)
    prefix = f"layers.{args.layer}.attn."
    tensors = [a.tensor(prefix + "wo_a." + name, "hpu") for name in ("weight", "channel_scale")]
    tensors += [b.tensor(prefix + "wo_b." + name, "hpu") for name in ("weight", "channel_scale")]
    report = dict(scope="BF16 inverse-RoPE boundary -> wo_a FP8 BMM/scaling/group32 codec -> wo_b FP8 projection",
                  tokens=args.tokens,
                  compiled=args.compiled,
                  checks=[],
                  timings={},
                  weights=[hashlib.sha256(t.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for t in tensors])
    old = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2
    wide = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_wide_gaudi2
    consumer = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2

    def reference(x, wa, sa, wb, sb):
        return consumer(old(x, wa, sa), wb, sb)

    def candidate(x, wa, sa, wb, sb):
        return consumer(wide(x, wa, sa), wb, sb)

    programs = {"reference": reference, "candidate": candidate}
    if args.compiled:
        programs = {
            name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
            for name, fn in programs.items()
        }

    def check(x, name):
        expected = programs["reference"](x, *tensors).cpu()
        observed = programs["candidate"](x, *tensors).cpu()
        delta = (expected.float() - observed.float()).abs()
        result = dict(name=name,
                      exact=torch.equal(expected.view(torch.int16), observed.view(torch.int16)),
                      different=int((expected != observed).sum()),
                      max_abs=float(delta.max()),
                      finite=bool(observed.isfinite().all()))
        report["checks"].append(result)
        args.output.write_text(json.dumps(report, indent=2))
        if not result["exact"]:
            raise RuntimeError("Wide wo_a emission changed its consumer")

    inputs = [torch.randn(args.tokens, 4, 4096).bfloat16().to("hpu") for _ in range(2)]
    with torch.inference_mode():
        for i, x in enumerate(inputs):
            check(x, f"changed_input{i}")
        if args.checks_only:
            return
        for fn in programs.values():
            for x in inputs:
                fn(x, *tensors)
        torch.hpu.synchronize()
        before = dict(metric_global("graph_compilation").stats())
        for name in (("candidate", ) if args.skip_reference_timing else programs):
            samples = []
            for i in range(args.repeats):
                start, end = (torch.hpu.Event(enable_timing=True) for _ in range(2))
                torch.hpu.reset_peak_memory_stats()
                allocated = torch.hpu.memory_allocated()
                wall = time.perf_counter_ns()
                start.record()
                output = programs[name](inputs[i % 2], *tensors)
                end.record()
                end.synchronize()
                samples.append(
                    dict(device_ms=start.elapsed_time(end),
                         wall_ms=(time.perf_counter_ns() - wall) / 1e6,
                         additional_peak_bytes=torch.hpu.max_memory_allocated() - allocated))
                del output
            report["timings"][name] = dict(samples=samples,
                                           median_device_ms=statistics.median(x["device_ms"] for x in samples),
                                           median_wall_ms=statistics.median(x["wall_ms"] for x in samples))
        report["compilation_before"] = before
        report["compilation_after"] = dict(metric_global("graph_compilation").stats())
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
