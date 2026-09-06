import argparse
import statistics
import time
from pathlib import Path

import torch
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
HEADS = 48
DIM = 128
EPSILON = 1e-6


def current(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor):
    normalized = FusedRMSNorm.apply(
        x.reshape(1, -1, DIM),
        weight,
        EPSILON,
    ).reshape_as(x)
    gate = torch.nn.functional.silu(z.float())
    return (normalized.float() * gate).to(x.dtype)


def native_tpc(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor):
    return torch.ops.custom_op.qwen38_rmsnorm_gated_bf16_gaudi2(
        x,
        z,
        weight,
    )


def benchmark(name, function, inputs, warmups, iterations):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(warmups):
        output = function(*inputs)
    torch.hpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    median = statistics.median(samples)
    print(
        f"{name} median_ms={median:.6f} min_ms={min(samples):.6f} "
        f"samples_ms={samples}",
        flush=True,
    )
    return output.clone(), median


def report_accuracy(expected, actual):
    expected_float = expected.float()
    actual_float = actual.float()
    difference = actual_float - expected_float
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    print(
        f"max_abs={float(difference.abs().max().cpu()):.9e} "
        f"mean_abs={float(difference.abs().mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e} "
        f"equal_fraction="
        f"{float((actual == expected).float().mean().cpu()):.9f}",
        flush=True,
    )


def capture_trace(name, function, inputs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    handler = torch.profiler.tensorboard_trace_handler(
        str(output_dir),
        worker_name=name,
        use_gzip=True,
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.HPU,
        ],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=False,
    ) as profiler:
        for _ in range(2):
            function(*inputs)
            torch.hpu.synchronize()
            profiler.step()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.ops.load_library(str(args.extension.resolve()))
    torch.manual_seed(739251)
    shape = (TOKENS * HEADS, DIM)
    x = torch.randn(shape, dtype=torch.bfloat16, device="hpu") * 0.05
    z = torch.randn(shape, dtype=torch.bfloat16, device="hpu")
    weight = (
        torch.randn(DIM, dtype=torch.bfloat16, device="hpu") * 0.02 + 1.0
    )
    inputs = (x, z, weight)

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_current = torch.compile(current, **compile_args)
    compiled_native = torch.compile(native_tpc, **compile_args)
    print(f"shape={shape} dtype={x.dtype} compile_args={compile_args}")

    expected, current_ms = benchmark(
        "current_fused_rmsnorm_plus_fp32_silu",
        compiled_current,
        inputs,
        args.warmups,
        args.iterations,
    )
    actual, native_ms = benchmark(
        "native_tpc_rmsnorm_gated",
        compiled_native,
        inputs,
        args.warmups,
        args.iterations,
    )
    report_accuracy(expected, actual)
    print(
        f"speedup={current_ms / native_ms:.6f} "
        f"saved_ms={current_ms - native_ms:.6f} "
        f"projected_48_layer_saved_ms={(current_ms - native_ms) * 48:.6f}",
        flush=True,
    )

    if args.trace_dir is not None:
        capture_trace("current", compiled_current, inputs, args.trace_dir)
        capture_trace("native_tpc", compiled_native, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
