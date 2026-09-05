import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


def benchmark(name, function, lhs, rhs, iterations):
    output = function(lhs, rhs)
    torch.hpu.synchronize()
    for _ in range(3):
        output = function(lhs, rhs)
    torch.hpu.synchronize()

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = function(lhs, rhs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    print(
        f"{name} dtype={output.dtype} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--m", type=int, default=192)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()

    torch.ops.load_library(str(args.extension.resolve()))
    torch.manual_seed(739251)
    lhs_bf16 = torch.randn(
        args.batch,
        args.m,
        args.k,
        dtype=torch.bfloat16,
        device="hpu",
    )
    rhs_bf16 = torch.randn(
        args.batch,
        args.k,
        args.n,
        dtype=torch.bfloat16,
        device="hpu",
    )
    lhs_fp32 = lhs_bf16.float()
    rhs_fp32 = rhs_bf16.float()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    bf16 = torch.compile(torch.bmm, **compile_args)
    fp32 = torch.compile(torch.bmm, **compile_args)
    mixed = torch.compile(
        torch.ops.custom_op.flashqla_bf16_bmm_f32,
        **compile_args,
    )

    bf16_output = benchmark(
        "bf16_to_bf16",
        bf16,
        lhs_bf16,
        rhs_bf16,
        args.iterations,
    )
    fp32_output = benchmark(
        "fp32_to_fp32",
        fp32,
        lhs_fp32,
        rhs_fp32,
        args.iterations,
    )
    mixed_output = benchmark(
        "bf16_to_fp32",
        mixed,
        lhs_bf16,
        rhs_bf16,
        args.iterations,
    )

    difference = (mixed_output - fp32_output).abs()
    relative_l2 = torch.linalg.vector_norm(mixed_output - fp32_output)
    relative_l2 /= torch.linalg.vector_norm(fp32_output).clamp_min(1e-12)
    print(
        f"mixed_vs_fp32 max_abs={float(difference.max().cpu()):.9e} "
        f"mean_abs={float(difference.mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e} "
        f"bf16_equal_fraction="
        f"{float((mixed_output.bfloat16() == bf16_output).float().mean().cpu()):.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
