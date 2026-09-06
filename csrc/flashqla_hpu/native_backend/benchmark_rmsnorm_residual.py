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


def baseline(x, residual, gate_input, weight):
    summed = x + residual
    normalized = FusedRMSNorm.apply(
        summed.reshape(1, -1, DIM),
        weight,
        EPSILON,
    ).reshape(-1, DIM)
    gate = torch.nn.functional.silu(gate_input.float())
    return (normalized.float() * gate).to(x.dtype)


def fused_residual(x, residual, gate_input, weight):
    normalized = FusedRMSNorm.apply(
        x.reshape(1, -1, DIM),
        weight,
        EPSILON,
        True,
        0,
        False,
        residual.reshape(1, -1, DIM),
    ).reshape(-1, DIM)
    gate = torch.nn.functional.silu(gate_input.float())
    return (normalized.float() * gate).to(x.dtype)


def benchmark(name, function, inputs, iterations):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(3):
        output = function(*inputs)
    torch.hpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return output


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

    traces = sorted(output_dir.glob(f"{name}*.pt.trace.json.gz"))
    if not traces:
        raise RuntimeError(f"Profiler did not create a trace for {name}")
    trace_path = output_dir / f"{name}.json.gz"
    traces[-1].replace(trace_path)
    print(f"trace={trace_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(739251)
    shape = (TOKENS * HEADS, DIM)
    x = torch.randn(shape, dtype=torch.bfloat16, device="hpu") * 0.01
    residual = torch.randn(shape, dtype=torch.bfloat16, device="hpu") * 0.01
    gate_input = torch.randn(shape, dtype=torch.bfloat16, device="hpu")
    weight = torch.randn(DIM, dtype=torch.bfloat16, device="hpu")
    inputs = x, residual, gate_input, weight

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_baseline = torch.compile(baseline, **compile_args)
    compiled_residual = torch.compile(fused_residual, **compile_args)
    print(f"shape={shape} dtype={x.dtype} compile_args={compile_args}", flush=True)

    expected = benchmark(
        "baseline",
        compiled_baseline,
        inputs,
        args.iterations,
    )
    actual = benchmark(
        "fused_residual",
        compiled_residual,
        inputs,
        args.iterations,
    )
    difference = (actual.float() - expected.float()).abs()
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float())
    relative_l2 /= torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    print(
        f"max_abs={float(difference.max().cpu()):.9e} "
        f"mean_abs={float(difference.mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e} "
        f"equal_fraction={float((actual == expected).float().mean().cpu()):.9f}",
        flush=True,
    )
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-3)
    print("correctness=pass", flush=True)

    if args.trace_dir is not None:
        capture_trace("baseline", compiled_baseline, inputs, args.trace_dir)
        capture_trace("fused_residual", compiled_residual, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
