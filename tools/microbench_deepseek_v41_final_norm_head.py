# SPDX-License-Identifier: Apache-2.0
"""Measure final RMSNorm through the real BF16/FP32 vocabulary projection."""
import argparse
import json
from pathlib import Path
import statistics

import habana_frameworks.torch.core  # noqa: F401
import torch

from vllm_gaudi.ops.deepseek_v41_math import rms_norm
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def measure(functions, activation, iterations):
    samples = {name: [] for name in functions}
    for function in functions.values():
        for _ in range(8):
            function(activation)
    torch.hpu.synchronize()
    events = {
        name: [(torch.hpu.Event(enable_timing=True),
                torch.hpu.Event(enable_timing=True))
               for _ in range(iterations)]
        for name in functions
    }
    for wave in range(iterations):
        activation.mul_(1.0001 if wave % 2 else 0.9999)
        order = ("parent", "candidate") if wave % 2 == 0 else (
            "candidate", "parent")
        for name in order:
            begin, end = events[name][wave]
            begin.record()
            functions[name](activation)
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end))
    return {
        name: {
            "device_ms": values,
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
        }
        for name, values in samples.items()
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    torch.ops.load_library(str(next(
        args.native_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))))
    shard = PreparedV41Shard(args.prepared, 1, 0)
    norm_weight = shard.tensor("norm.weight", "hpu")
    head_weight = shard.tensor("head.weight", "hpu")
    epsilon = 1e-20
    result = {
        "scope": "real PP1 final RMSNorm -> BF16xBF16 MME -> FP32 local logits",
        "head_shape": list(head_weight.shape),
        "batches": {},
    }

    for batch in (1, 2, 32):
        generator = torch.Generator().manual_seed(5120 + batch)
        source = (torch.randn(batch, 5120, generator=generator,
                              dtype=torch.bfloat16) * 0.25).to("hpu")

        def parent(value):
            normalized = rms_norm(value, norm_weight, epsilon)
            return torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                normalized.contiguous(), head_weight)

        def candidate(value):
            normalized = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(
                value.contiguous(), norm_weight, epsilon)
            return torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                normalized.contiguous(), head_weight)

        compiled = {
            "parent": torch.compile(parent, backend="hpu_backend",
                                    fullgraph=True, dynamic=False),
            "candidate": torch.compile(candidate, backend="hpu_backend",
                                       fullgraph=True, dynamic=False),
        }
        checks = []
        for factor in (0.0, 1e-4, 0.25, 1.0):
            value = (source * factor).to(torch.bfloat16)
            reference_norm = rms_norm(value, norm_weight, epsilon)
            actual_norm = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(
                value.contiguous(), norm_weight, epsilon)
            expected = compiled["parent"](value).cpu()
            actual = compiled["candidate"](value).cpu()
            checks.append({
                "factor": factor,
                "norm_bitwise": torch.equal(reference_norm.cpu(),
                                             actual_norm.cpu()),
                "norm_max_abs": float((reference_norm.float() -
                                       actual_norm.float()).abs().max().cpu()),
                "logits_max_abs": float((expected - actual).abs().max()),
                "logits_mean_abs": float((expected - actual).abs().mean()),
                "top1_equal": torch.equal(expected.argmax(-1),
                                           actual.argmax(-1)),
                "finite": bool(torch.isfinite(actual).all()),
            })
        timing = measure(compiled, source.clone(), args.iterations)
        parent_ms = timing["parent"]["median_ms"]
        candidate_ms = timing["candidate"]["median_ms"]
        result["batches"][str(batch)] = {
            "checks": checks,
            "timing": timing,
            "saved_ms": parent_ms - candidate_ms,
            "saved_percent": 100.0 * (parent_ms - candidate_ms) / parent_ms,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
