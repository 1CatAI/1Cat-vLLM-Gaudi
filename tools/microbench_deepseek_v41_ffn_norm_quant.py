# SPDX-License-Identifier: Apache-2.0
"""Diagnostic for fused V4.1 FFN RMSNorm and exact expert quantization.

This first-stage diagnostic verifies the numerical boundary and local compiled
chain.  It does not qualify the candidate for production: promotion also
requires the real N256 W13 consumer in the timed chain.
"""
import argparse
import json
from pathlib import Path
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def cpu_outputs(outputs):
    normalized, quantized, scale = outputs
    return (normalized.cpu(), quantized.view(torch.uint8).cpu(), scale.cpu())


def timing(function, inputs, weights, repeats):
    for _ in range(4):
        for x, weight in zip(inputs, weights, strict=True):
            function(x, weight)
    torch.hpu.synchronize()
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    wall = time.perf_counter_ns()
    start.record()
    for index in range(repeats):
        function(inputs[index % len(inputs)], weights[index % len(weights)])
    end.record()
    end.synchronize()
    host_ms = (time.perf_counter_ns() - wall) / 1e6 / repeats
    return {
        "device_ms_per_call": start.elapsed_time(end) / repeats,
        "host_ms_per_call": host_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 32, 128, 512])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(str(args.native_library))
    shard = PreparedV41Shard(args.prepared, 0, 0)
    epsilon = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
    layers = [0, 5, 10, 15]
    weights = [shard.tensor(f"layers.{layer}.ffn_norm.weight", "hpu") for layer in layers]

    def reference(x, weight):
        value = x.float()
        normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
                      * weight.float()).to(torch.bfloat16).contiguous()
        quantized, scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(normalized)
        return normalized, quantized, scale

    def candidate(x, weight):
        return torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, weight, epsilon)

    compiled_reference = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    compiled_candidate = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    report = {
        "scope": "FFN RMSNorm BF16 output plus exact N256 activation quantization; W13 consumer pending",
        "production_width": 5120,
        "epsilon": epsilon,
        "real_weight_layers": layers,
        "repeats": args.repeats,
        "batches": [],
    }
    torch.manual_seed(9471)
    all_exact = True
    for batch in args.batches:
        inputs = [torch.randn(batch, 5120, dtype=torch.bfloat16).to("hpu")
                  for _ in layers]
        reference_output = cpu_outputs(compiled_reference(inputs[0], weights[0]))
        candidate_output = cpu_outputs(compiled_candidate(inputs[0], weights[0]))
        equality = [torch.equal(a, b) for a, b in zip(reference_output, candidate_output, strict=True)]
        changed = inputs[0] + torch.full_like(inputs[0], 0.03125)
        changed_output = cpu_outputs(compiled_candidate(changed, weights[0]))
        changed_consumed = any(not torch.equal(a, b)
                               for a, b in zip(candidate_output, changed_output, strict=True))
        reference_time = timing(compiled_reference, inputs, weights, args.repeats)
        candidate_time = timing(compiled_candidate, inputs, weights, args.repeats)
        result = {
            "batch": batch,
            "normalized_bitwise_equal": equality[0],
            "quantized_bitwise_equal": equality[1],
            "scale_bitwise_equal": equality[2],
            "changed_input_consumed": changed_consumed,
            "reference": reference_time,
            "candidate": candidate_time,
            "device_gain_percent": 100.0 * (reference_time["device_ms_per_call"]
                                              - candidate_time["device_ms_per_call"])
                                      / reference_time["device_ms_per_call"],
        }
        report["batches"].append(result)
        print(json.dumps(result), flush=True)
        all_exact &= all(equality) and changed_consumed
    report["all_bitwise_exact"] = all_exact
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all_exact:
        raise AssertionError("FFN norm/quant numerical contract failed")


if __name__ == "__main__":
    main()
