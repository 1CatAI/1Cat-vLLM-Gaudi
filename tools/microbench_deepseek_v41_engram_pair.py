# SPDX-License-Identifier: Apache-2.0
"""Compare two Engram projection submissions with one paired BMM plan."""
import argparse
import json
from pathlib import Path
import statistics

import habana_frameworks.torch.core  # noqa: F401
import torch

from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
from vllm_gaudi.ops.deepseek_v41_math import engram_update
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def metrics(reference, candidate):
    delta = reference.float() - candidate.float()
    return {
        "bitwise": torch.equal(reference, candidate),
        "maximum_absolute": float(delta.abs().max().cpu()),
        "mean_absolute": float(delta.abs().mean().cpu()),
        "finite": bool(torch.isfinite(candidate).all().cpu()),
    }


def measure(functions, rows, residual, active, iterations):
    samples = {name: [] for name in functions}
    for function in functions.values():
        for _ in range(8):
            function(rows, residual, active)
    torch.hpu.synchronize()
    events = {
        name: [(torch.hpu.Event(enable_timing=True),
                torch.hpu.Event(enable_timing=True))
               for _ in range(iterations)]
        for name in functions
    }
    for wave in range(iterations):
        # Replace the inputs instead of multiplying BF16 in place by a value
        # that may round to one. Both arms consume changing production shapes.
        current = (rows + (0.001953125 if wave % 2 else -0.001953125)).to(
            torch.bfloat16)
        order = ("parent", "candidate") if wave % 2 == 0 else (
            "candidate", "parent")
        for name in order:
            begin, end = events[name][wave]
            begin.record()
            functions[name](current, residual, active)
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
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    torch.ops.load_library(str(next(
        args.native_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))))
    shard = PreparedV41Shard(args.prepared, 0, 0)
    sidecar = EngramFP8Sidecar(args.sidecar, shard)
    layers = (1, 14)
    weights = torch.stack([
        sidecar.tensor(f"layers.{layer}.engram.wkv.weight", "hpu")
        for layer in layers
    ])
    scales = torch.stack([
        sidecar.tensor(f"layers.{layer}.engram.wkv.channel_scale", "hpu")
        for layer in layers
    ])
    q_weight = torch.stack([
        shard.tensor(f"layers.{layer}.engram.q_weight", "hpu")
        for layer in layers
    ])
    k_weight = torch.stack([
        shard.tensor(f"layers.{layer}.engram.k_weight", "hpu")
        for layer in layers
    ])
    result = {
        "scope": "two real Engram FP8 projections through both residual consumers",
        "weight_shape": list(weights.shape),
        "batches": {},
    }

    for tokens in (1, 2, 32):
        generator = torch.Generator().manual_seed(6144 + tokens)
        rows = (torch.randn(2, tokens, 6144, generator=generator,
                            dtype=torch.bfloat16) * 0.25).to("hpu")
        residual = (torch.randn(2, tokens, 4, 5120,
                                generator=generator,
                                dtype=torch.bfloat16) * 0.25).to("hpu")
        active = torch.ones(2, tokens, dtype=torch.bool, device="hpu")

        def consume(projected, source_residual, mask):
            first = engram_update(source_residual[0], projected[0],
                                  q_weight[0], k_weight[0], mask[0])
            second = engram_update(source_residual[1], projected[1],
                                   q_weight[1], k_weight[1], mask[1])
            return torch.stack((first, second))

        def parent(source, source_residual, mask):
            first = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
                source[0].contiguous(), weights[0], scales[0])
            second = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
                source[1].contiguous(), weights[1], scales[1])
            return consume(torch.stack((first, second)), source_residual,
                           mask)

        def candidate(source, source_residual, mask):
            projected = torch.ops.custom_op.custom_deepseek_v41_dense_pair_fp8_gaudi2(
                source.contiguous(), weights, scales)
            return consume(projected, source_residual, mask)

        compiled = {
            "parent": torch.compile(parent, backend="hpu_backend",
                                    fullgraph=True, dynamic=False),
            "candidate": torch.compile(candidate, backend="hpu_backend",
                                       fullgraph=True, dynamic=False),
        }
        checks = []
        for factor in (0.0, 0.25, 1.0):
            value = (rows * factor).to(torch.bfloat16)
            expected = compiled["parent"](value, residual, active).cpu()
            actual = compiled["candidate"](value, residual, active).cpu()
            checks.append({"factor": factor, **metrics(expected, actual)})
        timing = measure(compiled, rows, residual, active, args.iterations)
        before = timing["parent"]["median_ms"]
        after = timing["candidate"]["median_ms"]
        result["batches"][str(tokens)] = {
            "checks": checks,
            "timing": timing,
            "saved_ms": before - after,
            "saved_percent": 100.0 * (before - after) / before,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
