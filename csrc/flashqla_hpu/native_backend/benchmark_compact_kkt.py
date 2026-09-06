import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


OUTER = 256
HEADS = 16
REPEATS = 3
CHUNK_SIZE = 64


def reference(compact_dot, grouped_beta):
    lower = torch.tril(
        compact_dot.unsqueeze(2) * grouped_beta.unsqueeze(-1),
        diagonal=-1,
    )
    eye = torch.eye(
        CHUNK_SIZE,
        dtype=compact_dot.dtype,
        device=compact_dot.device,
    )
    return lower + eye


def custom(compact_dot, grouped_beta):
    return torch.ops.custom_op.qwen38_compact_kkt_bf16_gaudi2(
        compact_dot,
        grouped_beta,
    )


def make_inputs(outer):
    torch.manual_seed(739251)
    compact_dot = torch.randn(
        outer,
        HEADS,
        CHUNK_SIZE,
        CHUNK_SIZE,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    grouped_beta = torch.rand(
        outer,
        HEADS,
        REPEATS,
        CHUNK_SIZE,
        dtype=torch.bfloat16,
        device="hpu",
    )
    return compact_dot, grouped_beta


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


def report_accuracy(actual, expected):
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
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.ops.load_library(str(args.extension.resolve()))
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_reference = torch.compile(reference, **compile_args)
    compiled_custom = torch.compile(custom, **compile_args)

    small_inputs = make_inputs(2)
    expected_small = compiled_reference(*small_inputs)
    actual_small = compiled_custom(*small_inputs)
    torch.hpu.synchronize()
    report_accuracy(actual_small, expected_small)
    torch.testing.assert_close(
        actual_small.cpu(),
        expected_small.cpu(),
        rtol=0,
        atol=0,
    )

    inputs = make_inputs(OUTER)
    print(
        f"shape=({OUTER},{HEADS},{REPEATS},"
        f"{CHUNK_SIZE},{CHUNK_SIZE})",
        flush=True,
    )
    expected = benchmark(
        "reference",
        compiled_reference,
        inputs,
        args.iterations,
    )
    actual = benchmark(
        "custom",
        compiled_custom,
        inputs,
        args.iterations,
    )
    sample_indices = [0, 1, 31, 127, OUTER - 1]
    report_accuracy(actual[sample_indices], expected[sample_indices])
    torch.testing.assert_close(
        actual[sample_indices].cpu(),
        expected[sample_indices].cpu(),
        rtol=0,
        atol=0,
    )

    if args.trace_dir is not None:
        capture_trace("reference", compiled_reference, inputs, args.trace_dir)
        capture_trace("custom", compiled_custom, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
