# SPDX-License-Identifier: Apache-2.0
"""Compare the complete production V4.1 gate/router chain with fused scores.

This is a component qualification only.  It uses every real PP0 gate matrix,
changes activations and image masks, and times through materialized top-6 IDs
and weights.  A numerically different candidate is archived and rejected
before performance timing.
"""
import argparse
import json
import os
from pathlib import Path
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def summarize(values):
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "samples": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    prepared = Path(os.environ["DSV41_PREPARED_DIR"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(410920)
    shard = PreparedV41Shard(prepared, args.pp_rank, args.tp_rank)
    layer_start = args.pp_rank * 20
    rows = []
    with torch.inference_mode():
        for layer in range(layer_start, layer_start + 20):
            prefix = f"layers.{layer}.ffn.gate."
            rows.append((
                torch.randn(1, 5120, dtype=torch.bfloat16, device="hpu"),
                shard.tensor(prefix + "weight", "hpu"),
                shard.tensor(prefix + "bias", "hpu"),
                shard.tensor(prefix + "bias_vl", "hpu"),
                torch.tensor([layer % 7 == 0], dtype=torch.bool, device="hpu"),
            ))

        def parent(x, weight, text, image, mask):
            logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                x.contiguous(), weight)
            scores = F.softplus(logits).sqrt()
            return torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2(
                scores, text, image, mask)

        def candidate(x, weight, text, image, mask):
            logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                x.contiguous(), weight)
            return torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                logits, text, image, mask)

        compiled = {
            "parent": torch.compile(parent, backend="hpu_backend", fullgraph=True,
                                    dynamic=False),
            "candidate": torch.compile(candidate, backend="hpu_backend",
                                       fullgraph=True, dynamic=False),
        }
        for fn in compiled.values():
            for _ in range(3):
                for row in rows:
                    fn(*row)
            torch.hpu.synchronize()

        checks = []
        all_exact = True
        for change in (1.0, 1.03125, 0.96875):
            for row in rows:
                row[0].mul_(change)
            for layer, row in enumerate(rows):
                expected = tuple(value.cpu() for value in compiled["parent"](*row))
                actual = tuple(value.cpu() for value in compiled["candidate"](*row))
                ids_equal = torch.equal(expected[0], actual[0])
                weights_equal = torch.equal(expected[1], actual[1])
                delta = (expected[1] - actual[1]).abs()
                checks.append({
                    "change": change,
                    "layer": layer,
                    "ids_equal": ids_equal,
                    "weights_bitwise_equal": weights_equal,
                    "weight_max_abs": float(delta.max()),
                    "weight_mean_abs": float(delta.mean()),
                })
                all_exact &= ids_equal and weights_equal
        torch.hpu.synchronize()
        record = {
            "scope": (f"20 real PP{args.pp_rank}/TP{args.tp_rank} gate matrices; "
                      "BF16xBF16/F32 gate MME through top-6 outputs"),
            "candidate": "production FP32 softplus followed by fused sqrt and exact top-6 TPC",
            "input_changes": [1.0, 1.03125, 0.96875],
            "checks": checks,
            "bitwise_exact": all_exact,
            "decision": "pending" if all_exact else "rejected_numeric_difference",
            "rounds": [],
        }
        (output / "router-logits.json").write_text(json.dumps(record, indent=2) + "\n")
        if not all_exact:
            print("router logits fusion rejected: candidate differs from production score chain", flush=True)
            return

        events = [(torch.hpu.Event(enable_timing=True),
                   torch.hpu.Event(enable_timing=True)) for _ in range(24)]
        for round_id in range(3):
            order = ("parent", "candidate") if round_id % 2 == 0 else (
                "candidate", "parent")
            for arm in order:
                device, host = [], []
                fn = compiled[arm]
                for begin, end in events:
                    started = time.perf_counter_ns()
                    begin.record()
                    for row in rows:
                        fn(*row)
                    end.record()
                    end.synchronize()
                    host.append((time.perf_counter_ns() - started) / 1e6 / len(rows))
                    device.append(begin.elapsed_time(end) / len(rows))
                record["rounds"].append({
                    "round": round_id,
                    "arm": arm,
                    "device_per_layer": summarize(device),
                    "synchronized_host_per_layer": summarize(host),
                })
                (output / "router-logits.json").write_text(
                    json.dumps(record, indent=2) + "\n")
                print(arm, round_id, np.mean(device), flush=True)
        parents = [r["device_per_layer"]["mean_ms"] for r in record["rounds"]
                   if r["arm"] == "parent"]
        candidates = [r["device_per_layer"]["mean_ms"] for r in record["rounds"]
                      if r["arm"] == "candidate"]
        record["parent_mean_ms_per_layer"] = float(np.mean(parents))
        record["candidate_mean_ms_per_layer"] = float(np.mean(candidates))
        record["saved_ms_per_layer"] = record["parent_mean_ms_per_layer"] - record[
            "candidate_mean_ms_per_layer"]
        record["decision"] = "accepted_component" if record[
            "saved_ms_per_layer"] > 0 else "rejected_performance"
        (output / "router-logits.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
