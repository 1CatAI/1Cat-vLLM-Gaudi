# SPDX-License-Identifier: Apache-2.0
"""Real-weight wq_a+wkv BF16 versus channel-scaled FP8 producer/consumer chain.

The benchmark ends after the two production RMSNorm consumers.  It prepares
the candidate directly from immutable checkpoint FP8/block32 bytes, so neither
arm uses synthetic weights or omits the source block scales.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_deepseek_v41_projection_chains import benchmark
from deepseek_v41_micro_replay import RecipeRecorder
from prepare_deepseek_v41_woa_fp8 import read_bytes
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rms_norm
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import prepare_block32_rows


def prepare_projection(shard, name, shape):
    n, k = shape
    weight, scale = shard.catalog[name + ".weight"], shard.catalog[name + ".scale"]
    if weight.dtype != "F8_E4M3" or weight.shape != shape or scale.shape != (n // 32, k // 32):
        raise ValueError(f"checkpoint projection contract changed: {name}")
    packed = np.empty(shape, dtype=np.uint8)
    channels = np.empty((n, ), dtype=np.float32)
    error_energy = source_energy = 0.0
    for row in range(0, n, 256):
        rows = min(256, n - row)
        if rows % 32:
            raise ValueError("projection rows must remain block32 aligned")
        codes = read_bytes(weight, row * k, rows * k).reshape(rows, k)
        powers = read_bytes(scale, row // 32 * (k // 32), rows // 32 * (k // 32)).reshape(rows // 32,
                                                                                          k // 32)
        q, s, record = prepare_block32_rows(codes, powers)
        packed[row:row + rows] = q
        channels[row:row + rows] = s[:, 0]
        error_energy += record["error_energy"]
        source_energy += record["source_energy"]
    return packed, channels, (error_energy / source_energy)**0.5 if source_energy else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--pp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--include-reference", action="store_true")
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(190923)
    recorder = RecipeRecorder(output)
    shard = PreparedV41Shard(args.prepared, args.pp_rank, 0)
    inputs, old, new, audit = [], [], [], []
    first = args.pp_rank * 20
    with torch.inference_mode():
        for layer in range(first, first + 20):
            prefix = f"layers.{layer}.attn"
            q = shard.dense(prefix + ".wq_a.weight", "hpu")
            kv = shard.dense(prefix + ".wkv.weight", "hpu")
            fused = torch.cat((q.cpu(), kv.cpu()), 0).contiguous().to("hpu")
            q8, sq, eq = prepare_projection(shard, prefix + ".wq_a", (1280, 5120))
            kv8, skv, ekv = prepare_projection(shard, prefix + ".wkv", (512, 5120))
            packed = torch.from_numpy(np.concatenate((q8, kv8), 0)).view(torch.float8_e4m3fn).to("hpu")
            channels = torch.from_numpy(np.concatenate((sq, skv))[None]).to("hpu")
            qnorm = shard.tensor(prefix + ".q_norm.weight", "hpu")
            kvnorm = shard.tensor(prefix + ".kv_norm.weight", "hpu")
            old.append((fused, qnorm, kvnorm))
            new.append((packed, channels, qnorm, kvnorm))
            inputs.append(torch.randn(1, 5120, dtype=torch.bfloat16, device="hpu"))
            audit.append({"layer": layer, "wq_a_relative_l2": eq, "wkv_relative_l2": ekv})

        def reference(x, weight, qnorm, kvnorm):
            value = F.linear(quantize_activation(x), weight)
            return (rms_norm(value[:, :1280], qnorm, 1e-20),
                    rms_norm(value[:, 1280:], kvnorm, 1e-20))

        def candidate(x, weight, scale, qnorm, kvnorm):
            value = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
                quantize_activation(x), weight, scale)
            return (rms_norm(value[:, :1280], qnorm, 1e-20),
                    rms_norm(value[:, 1280:], kvnorm, 1e-20))

        # Quantization is intentionally not bitwise equal to retained BF16.
        # Record its error at identical real inputs before timing each new
        # algorithm against its own ordinary compiled execution.
        differences = []
        for x, before, after in zip(inputs, old, new, strict=True):
            left, right = reference(x, *before), candidate(x, *after)
            for name, a, b in zip(("query", "kv"), left, right, strict=True):
                delta = a.float() - b.float()
                differences.append({
                    "layer": len(differences) // 2 + first,
                    "output": name,
                    "different": int((a != b).sum().cpu()),
                    "maximum_absolute": float(delta.abs().max().cpu()),
                    "relative_l2": float((delta.square().sum().sqrt() /
                                           a.float().square().sum().sqrt().clamp_min(1e-20)).cpu()),
                })
        (output / "qkv-input-fp8-audit.json").write_text(
            json.dumps({"weight_audit": audit, "consumer_differences": differences}, indent=2) + "\n")
        benchmark("qkv-input-fp8", reference, candidate, inputs, old, new, output, args.rounds, recorder,
                  not args.include_reference)


if __name__ == "__main__":
    main()
