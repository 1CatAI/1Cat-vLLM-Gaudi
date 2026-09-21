# SPDX-License-Identifier: Apache-2.0
"""Benchmark real Engram projection through its first residual consumer.

This is an isolated candidate screen.  Both arms include the production
activation roundtrip and ``engram_update``; only the resident projection
weight representation and MME path differ.
"""
import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmark_deepseek_v41_projection_chains import benchmark  # noqa: E402
from deepseek_v41_micro_replay import RecipeRecorder  # noqa: E402
from prepare_deepseek_v41_woa_fp8 import read_bytes  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import engram_update, quantize_activation  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import prepare_block32_rows  # noqa: E402


def prepare_projection(shard, name):
    shape = (25600, 6144)
    weight = shard.catalog[name + ".weight"]
    scale = shard.catalog[name + ".scale"]
    if (weight.dtype != "F8_E4M3" or weight.shape != shape or scale.dtype != "U8"
            or scale.shape != (800, 192)):
        raise ValueError(f"Engram projection contract changed: {name}")
    packed = np.empty(shape, dtype=np.uint8)
    channels = np.empty((25600, ), dtype=np.float32)
    error_energy = source_energy = 0.0
    maximum_absolute_error = 0.0
    temporary_upper_bound_bytes = 0
    for row in range(0, shape[0], 256):
        rows = min(256, shape[0] - row)
        codes = read_bytes(weight, row * shape[1], rows * shape[1]).reshape(rows, shape[1])
        powers = read_bytes(scale, row // 32 * scale.shape[1], rows // 32 * scale.shape[1]).reshape(
            rows // 32, scale.shape[1])
        q, s, record = prepare_block32_rows(codes, powers)
        packed[row:row + rows] = q
        channels[row:row + rows] = s[:, 0]
        error_energy += record["error_energy"]
        source_energy += record["source_energy"]
        maximum_absolute_error = max(maximum_absolute_error, record["maximum_absolute_error"])
        temporary_upper_bound_bytes = max(temporary_upper_bound_bytes,
                                          record["temporary_upper_bound_bytes"])
    return packed, channels, {
        "relative_l2": (error_energy / source_energy)**0.5 if source_energy else 0.0,
        "maximum_absolute_error": maximum_absolute_error,
        "source_weight_bytes": int(packed.nbytes),
        "prepared_weight_bytes": int(packed.nbytes + channels.nbytes),
        "temporary_upper_bound_bytes": int(temporary_upper_bound_bytes + packed.nbytes + channels.nbytes),
    }


def tensor_metrics(reference, candidate):
    delta = reference.float() - candidate.float()
    denominator = reference.float().square().sum().sqrt().clamp_min(1e-20)
    return {
        "different": int((reference != candidate).sum().cpu()),
        "maximum_absolute": float(delta.abs().max().cpu()),
        "mean_absolute": float(delta.abs().mean().cpu()),
        "relative_l2": float((delta.square().sum().sqrt() / denominator).cpu()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(210921)
    recorder = RecipeRecorder(output)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    inputs, old, new, weight_audit = [], [], [], []

    with torch.inference_mode():
        for layer in (1, 14):
            prefix = f"layers.{layer}.engram"
            bf16 = shard.dense(prefix + ".wkv.weight", "hpu")
            packed, channels, audit = prepare_projection(shard, prefix + ".wkv")
            fp8 = torch.from_numpy(packed).view(torch.float8_e4m3fn).to("hpu")
            channel = torch.from_numpy(channels[None]).to("hpu")
            q_weight = shard.tensor(prefix + ".q_weight", "hpu")
            k_weight = shard.tensor(prefix + ".k_weight", "hpu")
            rows = torch.randn(1, 6144, dtype=torch.bfloat16, device="hpu")
            residual = torch.randn(1, 4, 5120, dtype=torch.bfloat16, device="hpu")
            active = torch.ones(1, dtype=torch.bool, device="hpu")
            inputs.append(rows)
            old.append((residual, bf16, q_weight, k_weight, active))
            new.append((residual, fp8, channel, q_weight, k_weight, active))
            weight_audit.append({"layer": layer, **audit})
            print(f"prepared Engram layer {layer}: weight relative_l2={audit['relative_l2']:.9g}", flush=True)

        def reference(rows, residual, weight, q_weight, k_weight, active):
            kv = F.linear(quantize_activation(rows), weight)
            return engram_update(residual, kv, q_weight, k_weight, active)

        def candidate(rows, residual, weight, scale, q_weight, k_weight, active):
            kv = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
                quantize_activation(rows), weight, scale)
            return engram_update(residual, kv, q_weight, k_weight, active)

        cross_arm = []
        for layer, rows, before, after in zip((1, 14), inputs, old, new, strict=True):
            wanted = reference(rows, *before)
            actual = candidate(rows, *after)
            cross_arm.append({"layer": layer, "residual": tensor_metrics(wanted, actual)})
        torch.hpu.synchronize()
        (output / "engram-projection-audit.json").write_text(json.dumps({
            "scope": "real PP0 Engram layers 1 and 14; projection through residual update",
            "checkpoint": str(args.prepared),
            "weight_audit": weight_audit,
            "cross_arm": cross_arm,
        }, indent=2) + "\n")

        def ordinary_validator(arm, _rows, _weights, compiled, ordinary):
            # The retained BF16/F32 Engram math is already not bitwise equal
            # between eager and compiled execution.  Preserve the discrepancy
            # as evidence while still requiring changing-input replay below.
            return {"arm": arm, "compiled_vs_ordinary": tensor_metrics(ordinary[0], compiled[0])}

        benchmark("engram-projection", reference, candidate, inputs, old, new, output,
                  args.rounds, recorder, candidate_only=False, ordinary_validator=ordinary_validator)


if __name__ == "__main__":
    main()
