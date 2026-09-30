# SPDX-License-Identifier: Apache-2.0
"""Compare gate-library revisions in fresh processes through real mHC consumers.

The reference arm saves exact tensors, not just a pass flag. Both arms use the
same maintained hc_pre/hc_post and fused control projection. No model timing is
performed here; the identity sublayer isolates eight mHC boundaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from safetensors import safe_open
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, hc_post
    bind_worker_cpu(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    out = Path(os.environ["DSV41_RUN_EVIDENCE"])
    reference = torch.load(args.reference, weights_only=True) if args.reference else None
    saved, checks = {}, []
    result = {"status": "running", "checks": checks, "timings": {},
              "boundary": "8 real-weight hc_pre -> identity sublayer -> hc_post -> next hc_pre",
              "reference": str(args.reference) if args.reference else None,
              "math_module_sha256": hashlib.sha256(Path(hc_pre.__code__.co_filename).read_bytes()).hexdigest()}

    def write():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def check(name, tensors):
        values = tuple(value.cpu() for value in tensors)
        saved[name] = values
        item = {"name": name, "finite": all(bool(torch.isfinite(value).all()) for value in values)}
        if reference is not None:
            expected = reference[name]
            item["exact_bits"] = all(torch.equal(a.contiguous().view(torch.uint8),
                                                  b.contiguous().view(torch.uint8))
                                     for a, b in zip(values, expected, strict=True))
            item["max_abs"] = [float((a.float() - b.float()).abs().max())
                               for a, b in zip(values, expected, strict=True)]
        checks.append(item)
        if not item["finite"] or not item.get("exact_bits", True):
            result["status"] = "numerical_failure"
            write()
            torch.save(saved, out / "outputs.pt")
            raise RuntimeError(f"mHC exact-output gate failed: {item}")

    with torch.inference_mode():
        weights, scales, bases = [], [], []
        with safe_open(args.prepared / "pp0-tp0.safetensors", framework="pt", device="cpu") as handle:
            for layer in range(40):
                for kind in ("attn", "ffn"):
                    prefix = f"layers.{layer}.hc_{kind}_"
                    if layer < 4:
                        weights.append(handle.get_tensor(prefix + "fn").float())
                    scales.append(handle.get_tensor(prefix + "scale").float())
                    bases.append(handle.get_tensor(prefix + "base").float())
        weights = tuple(value.to("hpu") for value in weights)
        cpu_scales, cpu_bases = scales, bases
        scales = tuple(value.to("hpu") for value in scales)
        bases = tuple(value.to("hpu") for value in bases)

        def gate(raw, rrms, scale, base):
            return torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(raw, rrms, scale, base)

        gate = torch.compile(gate, backend="hpu_backend", fullgraph=True, dynamic=False)
        for count in (1, 2, 6, 128, 1024, 2048):
            generator = torch.Generator().manual_seed(9300 + count)
            raw = torch.randn(count, 24, generator=generator)
            raw *= torch.logspace(-4, 3, count).reshape(-1, 1)
            rrms = torch.logspace(-2, 2, count).reshape(-1, 1)
            raw[0] = torch.tensor([0., -0., 1., -1., 80., -80., 1e-6, -1e-6] * 3)
            if count > 1:
                raw[1] = 0  # Exact ties in row softmax and sigmoid boundaries.
            raw, rrms = raw.to("hpu"), rrms.to("hpu")
            for index in range(80):
                check(f"gate/{count}/{index}", (gate(raw, rrms, scales[index], bases[index]),))
            print(f"gate C{count}: all 80 real scales/bases exact", flush=True)

        def chain(residual, previous):
            for index in range(8):
                value, previous, post, comb = hc_pre(residual, previous, weights[index],
                                                     scales[index], bases[index], packed_fn=weights[index])
                residual = hc_post(value, residual, post, comb)
            return residual, previous

        compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
        fixtures = {}
        for count in (1, 2, 6):
            bucket = []
            for seed in range(4):
                generator = torch.Generator().manual_seed(9300 + seed)
                residual = (torch.randn(count, 4, 5120, dtype=torch.bfloat16, generator=generator) * 0.25).to("hpu")
                previous = torch.softmax(torch.randn(count, 4, generator=generator), -1).to("hpu")
                bucket.append((residual, previous))
                check(f"chain/{count}/{seed}", compiled(residual, previous))
            # Reorder and reuse a bucket with different rows, retaining the consumer.
            check(f"reorder/{count}", compiled(bucket[-1][0].flip(0), bucket[-1][1].flip(0)))
            fixtures[count] = bucket
            print(f"chain C{count}: consumers and reordered inputs exact", flush=True)
        check("return_to_c1", compiled(*fixtures[1][0]))
        torch.save(saved, out / "outputs.pt")
        result["fixture_scales_sha256"] = hashlib.sha256(
            b"".join(v.numpy().tobytes() for v in cpu_scales + cpu_bases)).hexdigest()
        bind_worker_helpers(0)
        for count in (1, 2):
            bucket = fixtures[count]
            device, wall = [], []
            residual, previous = (torch.empty_like(value) for value in bucket[0])
            for iteration in range(args.iterations + 8):
                residual.copy_(bucket[iteration % len(bucket)][0])
                previous.copy_(bucket[iteration % len(bucket)][1])
                torch.hpu.synchronize()
                begin, end = (torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                started = time.perf_counter_ns()
                begin.record()
                output = compiled(residual, previous)
                end.record()
                # Include the completion of both real outputs, not only enqueue.
                output[0].cpu()
                output[1].cpu()
                torch.hpu.synchronize()
                if iteration >= 8:
                    device.append(begin.elapsed_time(end))
                    wall.append((time.perf_counter_ns() - started) / 1e6)
            result["timings"][str(count)] = {"device_ms": device, "wall_ms": wall,
                                             "device_median_ms": statistics.median(device),
                                             "wall_median_ms": statistics.median(wall)}
            write()
            print(f"C{count} device={statistics.median(device):.6f} wall={statistics.median(wall):.6f} ms", flush=True)
        result["status"] = "exact_and_component_measured" if reference is not None else "reference_saved_and_measured"
        result["memory_allocated"] = torch.hpu.memory_allocated()
        result["memory_reserved"] = torch.hpu.memory_reserved()
        write()


if __name__ == "__main__":
    main()
