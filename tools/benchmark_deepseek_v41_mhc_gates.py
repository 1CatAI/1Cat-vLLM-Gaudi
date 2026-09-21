# SPDX-License-Identifier: Apache-2.0
"""Measure all 80 V4.1 mHC gate/Sinkhorn calls in one compiled graph."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch.core  # noqa: F401
from safetensors import safe_open


def load_constants(prepared):
    scales, bases = [], []
    for file_name, layers in (("pp0-tp0.safetensors", range(20)),
                              ("pp1-tp0.safetensors", range(20, 40))):
        with safe_open(prepared / file_name, framework="pt", device="cpu") as handle:
            for layer in layers:
                for kind in ("attn", "ffn"):
                    prefix = f"layers.{layer}.hc_{kind}_"
                    scales.append(handle.get_tensor(prefix + "scale").float())
                    bases.append(handle.get_tensor(prefix + "base").float())
    return torch.stack(scales).to("hpu"), torch.stack(bases).to("hpu")


def full_sweep(mixes, rrms, scales, bases):
    outputs = []
    for index in range(80):
        outputs.append(
            torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
                mixes[index].contiguous(), rrms[index].contiguous(),
                scales[index].contiguous(), bases[index].contiguous()))
    return torch.stack(outputs)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    extension = next(args.native_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    torch.ops.load_library(str(extension))
    scales, bases = load_constants(args.prepared)

    generator = torch.Generator().manual_seed(4164)
    mixes = (torch.randn(80, 1, 24, generator=generator) * 3.0).to("hpu")
    rrms = (torch.rand(80, 1, 1, generator=generator) * 0.5 + 0.75).to("hpu")
    compiled = torch.compile(lambda m, r: full_sweep(m, r, scales, bases),
                             backend="hpu_backend", fullgraph=True,
                             dynamic=False)
    output = compiled(mixes, rrms)
    output_cpu = output.cpu().contiguous()
    torch.hpu.synchronize()
    raw = output_cpu.numpy().tobytes()

    device, host = [], []
    for iteration in range(args.iterations + 8):
        started = time.perf_counter_ns()
        begin = torch.hpu.Event(enable_timing=True)
        end = torch.hpu.Event(enable_timing=True)
        begin.record()
        value = compiled(mixes, rrms)
        end.record()
        value.cpu()
        torch.hpu.synchronize()
        if iteration >= 8:
            device.append(begin.elapsed_time(end))
            host.append((time.perf_counter_ns() - started) / 1e6)

    result = {
        "scope": "40 real layers / 80 mHC gate and Sinkhorn calls",
        "iterations": args.iterations,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "output_shape": list(output_cpu.shape),
        "finite": torch.isfinite(output_cpu).all().item(),
        "device_ms": device,
        "device_median_ms": statistics.median(device),
        "device_mean_ms": statistics.fmean(device),
        "host_median_ms": statistics.median(host),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(output_cpu, args.output.with_suffix(".pt"))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
