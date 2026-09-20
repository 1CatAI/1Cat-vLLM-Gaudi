# SPDX-License-Identifier: Apache-2.0
"""Check and time the exact, batch-generic V4.1 KV norm/RoPE fusion."""

import argparse
import json
import os
from pathlib import Path
import time

import torch


def _measure(fn, args, repeats):
    for _ in range(8):
        fn(*args)
    torch.hpu.synchronize()
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    for _ in range(repeats):
        fn(*args)
    end.record()
    torch.hpu.synchronize()
    host_ms = (time.perf_counter_ns() - host_start) / 1.0e6
    return {
        "device_ms_per_call": start.elapsed_time(end) / repeats,
        "host_ms_per_call": host_ms / repeats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 6, 32, 512])
    parser.add_argument("--repeats", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(20260920)
    phase = torch.randn(8192, 64, dtype=torch.float32, device="hpu")
    weight = torch.randn(512, dtype=torch.bfloat16, device="hpu")
    epsilon = 1.0e-6

    def reference(value, norm_weight, positions, rope_phase):
        normalized = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(
            value, norm_weight, epsilon)
        if normalized.shape[0] <= 6:
            return torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                normalized.reshape(normalized.shape[0], 1, 512),
                positions, rope_phase).reshape(normalized.shape)
        pairs = normalized[..., -64:].float().unflatten(-1, (-1, 2))
        native_phase = rope_phase.index_select(0, positions.long())
        real, imag = pairs[..., 0], pairs[..., 1]
        cosine, sine = native_phase[..., :32], native_phase[..., 32:]
        rotated = torch.stack((real * cosine - imag * sine,
                               real * sine + imag * cosine), -1)
        return torch.cat((normalized[..., :-64],
                          rotated.flatten(-2).to(normalized.dtype)), -1)

    def candidate(value, norm_weight, positions, rope_phase):
        return torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(
            value, norm_weight, positions, rope_phase, epsilon)

    result = {
        "scope": "exact KV RMSNorm BF16 boundary plus forward RoPE",
        "batches": [],
        "repeats": args.repeats,
    }
    with torch.inference_mode():
        for batch in args.batches:
            value = torch.randn(batch, 512, dtype=torch.bfloat16, device="hpu")
            positions = torch.arange(511, 511 + batch, dtype=torch.int32, device="hpu")
            ref = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
            cand = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
            expected = ref(value, weight, positions, phase).cpu()
            actual = cand(value, weight, positions, phase).cpu()
            changed = value.clone()
            changed[:, 0].add_(torch.tensor(0.5, dtype=torch.bfloat16, device="hpu"))
            expected_changed = ref(changed, weight, positions, phase).cpu()
            actual_changed = cand(changed, weight, positions, phase).cpu()
            row = {
                "batch": batch,
                "bitwise_equal": bool(torch.equal(expected.view(torch.int16), actual.view(torch.int16))),
                "changed_bitwise_equal": bool(torch.equal(expected_changed.view(torch.int16),
                                                           actual_changed.view(torch.int16))),
                "changed_input_consumed": bool(not torch.equal(actual.view(torch.int16),
                                                                actual_changed.view(torch.int16))),
                "different": int((expected.view(torch.int16) != actual.view(torch.int16)).sum()),
                "reference": _measure(ref, (value, weight, positions, phase), args.repeats),
                "candidate": _measure(cand, (value, weight, positions, phase), args.repeats),
            }
            row["device_gain_percent"] = 100.0 * (
                row["reference"]["device_ms_per_call"]
                - row["candidate"]["device_ms_per_call"]
            ) / row["reference"]["device_ms_per_call"]
            result["batches"].append(row)
            print(json.dumps(row), flush=True)
    if not all(row["bitwise_equal"] and row["changed_bitwise_equal"]
               and row["changed_input_consumed"] for row in result["batches"]):
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise AssertionError("KV norm/RoPE fusion contract failed")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
