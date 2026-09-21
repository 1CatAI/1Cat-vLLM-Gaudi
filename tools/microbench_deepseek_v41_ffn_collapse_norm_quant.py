# SPDX-License-Identifier: Apache-2.0
"""Qualify mHC collapse -> BF16 -> FFN norm -> exact N256 quant fusion."""
import argparse
import json
from pathlib import Path
import statistics
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def cpu(outputs):
    normalized, quantized, scale = outputs
    return normalized.cpu(), quantized.view(torch.uint8).cpu(), scale.cpu()


def measure(function, fixtures, repeats):
    for _ in range(8):
        for residual, pre, weight in fixtures:
            function(residual, pre, weight)
    torch.hpu.synchronize()
    samples, host = [], []
    for index in range(repeats):
        residual, pre, weight = fixtures[index % len(fixtures)]
        begin = torch.hpu.Event(enable_timing=True)
        end = torch.hpu.Event(enable_timing=True)
        wall = time.perf_counter_ns()
        begin.record()
        function(residual, pre, weight)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
        host.append((time.perf_counter_ns() - wall) / 1e6)
    return {
        "device_median_ms": statistics.median(samples),
        "device_mean_ms": statistics.mean(samples),
        "host_median_ms": statistics.median(host),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=64)
    args = parser.parse_args()
    torch.ops.load_library(str(next(args.native_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))))
    shard = PreparedV41Shard(args.prepared, 0, 0)
    epsilon = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
    layers = (0, 5, 10, 15)
    weights = [shard.tensor(f"layers.{layer}.ffn_norm.weight", "hpu") for layer in layers]

    def reference(residual, previous_pre, weight):
        collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).to(torch.bfloat16)
        value = collapsed.float()
        normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
                      * weight.float()).to(torch.bfloat16).contiguous()
        quantized, scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(normalized)
        return normalized, quantized, scale

    def candidate(residual, previous_pre, weight):
        return torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            residual, previous_pre, weight, epsilon)

    reference = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    candidate = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(9407)
    report = {"scope": "mHC collapse through exact N256 activation operands",
              "layers": list(layers), "batches": []}
    qualified = True
    for batch in (1, 2):
        fixtures = []
        for weight in weights:
            residual = torch.randn(batch, 4, 5120, dtype=torch.bfloat16, device="hpu")
            previous_pre = torch.softmax(torch.randn(batch, 4, dtype=torch.float32, device="hpu"), -1)
            fixtures.append((residual, previous_pre, weight))
        expected = cpu(reference(*fixtures[0]))
        actual = cpu(candidate(*fixtures[0]))
        equal = [torch.equal(a, b) for a, b in zip(expected, actual, strict=True)]
        changed = list(fixtures[0])
        changed[0] = changed[0] + torch.full_like(changed[0], 0.03125)
        changed_consumed = any(not torch.equal(a, b) for a, b in zip(actual, cpu(candidate(*changed)), strict=True))
        reference_time = measure(reference, fixtures, args.repeats)
        candidate_time = measure(candidate, fixtures, args.repeats)
        result = {
            "batch": batch,
            "normalized_bitwise_equal": equal[0],
            "quantized_bitwise_equal": equal[1],
            "scale_bitwise_equal": equal[2],
            "changed_input_consumed": changed_consumed,
            "reference": reference_time,
            "candidate": candidate_time,
            "median_gain_percent": 100.0 * (reference_time["device_median_ms"] - candidate_time["device_median_ms"])
                                   / reference_time["device_median_ms"],
        }
        report["batches"].append(result)
        qualified &= all(equal) and changed_consumed and result["median_gain_percent"] > 0
        print(json.dumps(result), flush=True)
    report["qualified"] = qualified
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not qualified:
        raise AssertionError("FFN collapse/norm/quant candidate did not qualify")


if __name__ == "__main__":
    main()
