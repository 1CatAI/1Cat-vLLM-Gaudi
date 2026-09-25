# SPDX-License-Identifier: Apache-2.0
"""Qualify batched routing/expert/shared projection through mHC consumption."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--batch", type=int, choices=(2, 4, 8, 16, 32, 64), default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-timing", action="store_true")
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_math import hc_post
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.set_num_threads(1)
    torch.manual_seed(7472)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    shard = PreparedV41Shard(args.prepared, 0, 0)
    specs = {name: spec for name, spec in shard.specs.items() if name.startswith("layers.0.ffn.")}
    weights = _weight_tree(specs)
    load_weight_tree(shard, weights, "hpu", specs)
    moe = PreparedMoE(weights.layers.get_submodule("0").ffn, 6, True, mxfp4_bf16_lut("hpu"), lambda x: x)
    moe.prepare_shared_gate_up_weight()
    b = args.batch
    x = torch.randn(b, 5120).bfloat16().to("hpu")
    mask = (torch.arange(b) % 3 == 0).to("hpu")
    residual = torch.randn(b, 4, 5120).bfloat16().to("hpu")
    post, comb = torch.rand(b, 4).to("hpu"), torch.rand(b, 4, 4).to("hpu")

    def chain(value, image, old, p, c):
        return hc_post(moe(value, image, ordinary_decode=True), old, p, c)

    reference = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    candidate = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)

    def serial():
        return torch.cat(
            [reference(x[i:i + 1], mask[i:i + 1], residual[i:i + 1], post[i:i + 1], comb[i:i + 1]) for i in range(b)])

    def batch():
        return candidate(x, mask, residual, post, comb)

    report = {"batch": b, "scope": __doc__, "checks": [], "timings": {}}
    with torch.inference_mode():
        for change in range(2):
            if change:
                x.copy_(torch.randn(b, 5120).bfloat16())
                mask.logical_not_()
            expected, actual = serial().cpu(), batch().cpu()
            delta = actual.float() - expected.float()
            check = dict(change=change,
                         exact=torch.equal(expected, actual),
                         max_abs=float(delta.abs().max()),
                         relative_l2=float(delta.norm() / expected.float().norm()))
            report["checks"].append(check)
            args.output.write_text(json.dumps(report, indent=2))
            torch.save(dict(input=x.cpu(), mask=mask.cpu(), reference=expected, actual=actual),
                       args.output.parent / f"output-{change}.pt")
            if not check["exact"]:
                raise RuntimeError(f"Batch FFN changed the existing precision contract: {check}")
        for name, fn in (("serial", serial), ("batch", batch)):
            if name == "serial" and not args.reference_timing:
                continue
            for _ in range(2):
                fn()
            torch.hpu.synchronize()
            samples = []
            for _ in range(5):
                x.copy_(torch.randn(b, 5120).bfloat16())
                torch.hpu.synchronize()
                resident = torch.hpu.memory_allocated()
                torch.hpu.reset_peak_memory_stats()
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start = time.perf_counter()
                begin.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(
                    dict(device_ms=begin.elapsed_time(end),
                         wall_ms=(time.perf_counter() - start) * 1000,
                         extra_peak_bytes=torch.hpu.max_memory_allocated() - resident))
            report["timings"][name] = samples
            args.output.write_text(json.dumps(report, indent=2))
            print(name, statistics.median(x["device_ms"] for x in samples), flush=True)


if __name__ == "__main__":
    main()
