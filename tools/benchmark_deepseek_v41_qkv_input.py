# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the V4.1 wq_a/wkv fused input projection.

This is deliberately a local producer benchmark.  It times the same real
weights and activation quantization used by the stage, but does not claim an
end-to-end decode result; that claim requires a subsequent four-rank run.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_math import quantize_activation  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


def _separate(value, q_weight, kv_weight):
    q = quantize_activation(value)
    # Reusing the same quantized activation is not the baseline: this function
    # intentionally models the two current calls, each with its own helper.
    return F.linear(quantize_activation(value), q_weight), F.linear(q, kv_weight)


def _fused(value, weight):
    return F.linear(quantize_activation(value), weight)


def _summary(values):
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "samples": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--pp-rank", type=int, default=0, choices=(0, 1))
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(141001)
    shard = PreparedV41Shard(args.prepared, args.pp_rank, 0)
    layer_ids = args.layers if args.layers is not None else list(range(args.pp_rank * 20, (args.pp_rank + 1) * 20))
    inputs, separate_weights, fused_weights = [], [], []
    with torch.inference_mode():
        for layer in layer_ids:
            prefix = f"layers.{layer}.attn."
            q = shard.dense(prefix + "wq_a.weight", "hpu")
            kv = shard.dense(prefix + "wkv.weight", "hpu")
            inputs.append(torch.randn(1, q.shape[1], dtype=torch.bfloat16, device="hpu"))
            separate_weights.append((q, kv))
            fused_weights.append(torch.cat((q, kv), dim=0).contiguous())

        separate = torch.compile(_separate, backend="hpu_backend", fullgraph=True, dynamic=False)
        fused = torch.compile(_fused, backend="hpu_backend", fullgraph=True, dynamic=False)
        # Warm up and check the actual changed producer on every layer.
        for value, (q, kv), weight in zip(inputs, separate_weights, fused_weights, strict=True):
            left = separate(value, q, kv)
            right = fused(value, weight)
            assert torch.equal(left[0].cpu(), right[:, :q.shape[0]].cpu())
            assert torch.equal(left[1].cpu(), right[:, q.shape[0]:].cpu())
        torch.hpu.synchronize()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                  for _ in range(args.samples)]
        record = {"scope": "wq_a+wkv input producer only; real prepared weights; no TP/PP",
                  "layers": layer_ids, "layers_per_sweep": len(inputs), "rounds": [],
                  "weight_bytes": {"separate": sum(a.numel() + b.numel() for a, b in separate_weights) * 2,
                                   "fused": sum(w.numel() for w in fused_weights) * 2}}
        for round_id in range(args.rounds):
            for value in inputs:
                value.mul_(1.001 + round_id * .001)
            for name, fn, weights in (("separate", separate, separate_weights), ("fused", fused, fused_weights)):
                torch.hpu.synchronize()
                device = []
                for begin, end in events:
                    begin.record()
                    for value, weight in zip(inputs, weights, strict=True):
                        fn(value, *weight) if name == "separate" else fn(value, weight)
                    end.record()
                    end.synchronize()
                    device.append(begin.elapsed_time(end))
                item = {"round": round_id, "arm": name, "device_sweep": _summary(device),
                        "mean_ms_per_layer": float(np.mean(device)) / len(inputs)}
                record["rounds"].append(item)
                print(name, round_id, item["mean_ms_per_layer"], "ms/layer", flush=True)
        separate_mean = np.mean([x["mean_ms_per_layer"] for x in record["rounds"] if x["arm"] == "separate"])
        fused_mean = np.mean([x["mean_ms_per_layer"] for x in record["rounds"] if x["arm"] == "fused"])
        record["speedup_vs_separate"] = separate_mean / fused_mean
    (output / "qkv-input.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
