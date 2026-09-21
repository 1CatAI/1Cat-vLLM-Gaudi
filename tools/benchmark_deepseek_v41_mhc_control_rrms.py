# SPDX-License-Identifier: Apache-2.0
"""Benchmark fused mHC control projection/RRMS through all gate consumers.

The chain uses four real V4.1 layers (attention and FFN mHC boundaries), real
weights, changing BF16 residuals and the production fused gate operator.  The
identity sublayer between mHC boundaries keeps the real dependency topology
without mixing attention or expert time into this component decision.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch.core  # noqa: F401
from safetensors import safe_open


def pack_control_weight(weight):
    """The linear-lane kernel consumes the original checkpoint K order."""
    return weight.contiguous()


def gate(projection, rrms, scale, base):
    value = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
        projection.contiguous(), rrms.contiguous(), scale.contiguous(),
        base.contiguous())
    return value[:, :4], value[:, 4:8], value[:, 8:].reshape(-1, 4, 4)


def update(residual, previous_pre, projection, rrms, scale, base):
    pre, post, comb = gate(projection, rrms, scale, base)
    collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).to(
        residual.dtype)
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)
    # An identity sublayer output preserves the production hc_post dataflow.
    residual = (collapsed.float().unsqueeze(1) * post.unsqueeze(-1) +
                mixed).to(residual.dtype)
    return residual, pre


def reference(residual, previous_pre, weights, scales, bases):
    for index in range(8):
        flat = residual.flatten(1).float()
        projection = torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2(
            flat, weights[index])
        rrms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
        residual, previous_pre = update(residual, previous_pre, projection,
                                        rrms, scales[index], bases[index])
    return residual, previous_pre


def candidate(residual, previous_pre, weights, scales, bases):
    for index in range(8):
        control = (
            torch.ops.custom_op.
            custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
                residual.flatten(1).contiguous(), weights[index], 1e-20))
        projection, rrms = control[:, :24], control[:, 24:]
        residual, previous_pre = update(residual, previous_pre, projection,
                                        rrms, scales[index], bases[index])
    return residual, previous_pre


def measure(function, fixtures, iterations):
    residual = torch.empty_like(fixtures[0][0])
    previous = torch.empty_like(fixtures[0][1])
    device, host = [], []
    for iteration in range(iterations + 8):
        source, source_pre = fixtures[iteration % len(fixtures)]
        residual.copy_(source)
        previous.copy_(source_pre)
        torch.hpu.synchronize()
        begin, end = (torch.hpu.Event(enable_timing=True),
                      torch.hpu.Event(enable_timing=True))
        started = time.perf_counter_ns()
        begin.record()
        output = function(residual, previous)
        end.record()
        # Drain both real consumers instead of timing enqueue only.
        output[0].cpu()
        output[1].cpu()
        torch.hpu.synchronize()
        if iteration >= 8:
            device.append(begin.elapsed_time(end))
            host.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "iterations": iterations,
        "device_ms": device,
        "device_median_ms": statistics.median(device),
        "device_mean_ms": statistics.fmean(device),
        "host_ms": host,
        "host_median_ms": statistics.median(host),
    }


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

    rank_file = args.prepared / "pp0-tp0.safetensors"
    weights, scales, bases = [], [], []
    with safe_open(rank_file, framework="pt", device="cpu") as handle:
        for layer in range(4):
            for kind in ("attn", "ffn"):
                prefix = f"layers.{layer}.hc_{kind}_"
                weights.append(handle.get_tensor(prefix + "fn").float())
                scales.append(handle.get_tensor(prefix + "scale").float())
                bases.append(handle.get_tensor(prefix + "base").float())
    original = torch.stack(weights).to("hpu")
    packed = torch.stack([pack_control_weight(value) for value in weights]).to(
        "hpu")
    scales = torch.stack(scales).to("hpu")
    bases = torch.stack(bases).to("hpu")

    compiled_reference = torch.compile(
        lambda residual, previous: reference(residual, previous, original,
                                             scales, bases),
        backend="hpu_backend", fullgraph=True, dynamic=False)
    compiled_candidate = torch.compile(
        lambda residual, previous: candidate(residual, previous, packed,
                                             scales, bases),
        backend="hpu_backend", fullgraph=True, dynamic=False)
    fixtures = []
    for seed in range(4):
        generator = torch.Generator().manual_seed(4100 + seed)
        residual = (torch.randn(1, 4, 5120, generator=generator,
                                dtype=torch.bfloat16) * 0.25).to("hpu")
        previous = torch.tensor([[1.0 - 0.01 * seed, 0.01 * seed, 0, 0]],
                                dtype=torch.float32, device="hpu")
        fixtures.append((residual, previous))

    checks = []
    for residual, previous in fixtures:
        expected = compiled_reference(residual, previous)
        actual = compiled_candidate(residual, previous)
        torch.hpu.synchronize()
        expected = tuple(value.cpu() for value in expected)
        actual = tuple(value.cpu() for value in actual)
        checks.append({
            "residual_bitwise": torch.equal(expected[0].view(torch.int16),
                                             actual[0].view(torch.int16)),
            "residual_max_abs": (expected[0].float() -
                                 actual[0].float()).abs().max().item(),
            "pre_max_abs": (expected[1] - actual[1]).abs().max().item(),
            "finite": all(torch.isfinite(value).all().item()
                          for value in actual),
        })
    if not all(item["finite"] for item in checks):
        raise RuntimeError("fused control/RRMS produced non-finite output")

    reference_result = measure(compiled_reference, fixtures, args.iterations)
    candidate_result = measure(compiled_candidate, fixtures, args.iterations)
    result = {
        "status": "microbench_complete",
        "boundary": "four real layers / eight mHC boundaries through fused gates and hc_post consumer",
        "real_weight_bytes": int(original.numel() * original.element_size()),
        "extra_prepared_weight_bytes": int(packed.numel() * packed.element_size()),
        "checks": checks,
        "reference": reference_result,
        "candidate": candidate_result,
    }
    result["device_saved_ms"] = (reference_result["device_median_ms"] -
                                 candidate_result["device_median_ms"])
    result["device_saved_percent"] = (
        100 * result["device_saved_ms"] /
        reference_result["device_median_ms"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
