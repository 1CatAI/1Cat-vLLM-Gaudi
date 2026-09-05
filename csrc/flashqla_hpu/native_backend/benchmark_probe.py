import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


def baseline(lhs: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor):
    return torch.bmm(lhs, rhs) + bias


def compound(lhs: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor):
    return torch.ops.custom_op.flashqla_compound_probe(lhs, rhs, bias)


def compound_bundled(lhs: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor):
    return torch.ops.custom_op.flashqla_compound_probe_bundled(lhs, rhs, bias)


def compound_scheduled(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    bias: torch.Tensor,
):
    return torch.ops.custom_op.flashqla_compound_probe_scheduled(lhs, rhs, bias)


def synchronize() -> None:
    torch.hpu.synchronize()


def benchmark(name, function, inputs, iterations):
    output = function(*inputs)
    synchronize()
    for _ in range(3):
        output = function(*inputs)
    synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(*inputs)
        synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return output


def capture_trace(name, function, inputs, output_dir):
    trace_path = output_dir / f"{name}.json.gz"
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
        record_shapes=True,
    ) as profiler:
        for _ in range(2):
            function(*inputs)
            synchronize()
            profiler.step()

    traces = sorted(output_dir.glob(f"{name}*.pt.trace.json.gz"))
    if not traces:
        raise RuntimeError(f"Profiler did not create a trace for {name}")
    traces[-1].replace(trace_path)
    print(f"{name} trace={trace_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=12_288)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = next(Path(__file__).parent.glob("flashqla_compound_probe*.so"))
    torch.ops.load_library(str(library.resolve()))

    torch.manual_seed(739251)
    lhs = torch.randn(
        args.batch,
        args.m,
        args.k,
        dtype=torch.bfloat16,
        device="hpu",
    )
    rhs = torch.randn(
        args.batch,
        args.k,
        args.n,
        dtype=torch.bfloat16,
        device="hpu",
    )
    bias = torch.randn(
        args.batch,
        args.m,
        args.n,
        dtype=torch.bfloat16,
        device="hpu",
    )
    inputs = lhs, rhs, bias

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_baseline = torch.compile(baseline, **compile_args)
    compiled_compound = torch.compile(compound, **compile_args)
    compiled_bundled = torch.compile(compound_bundled, **compile_args)
    compiled_scheduled = torch.compile(compound_scheduled, **compile_args)
    print(
        f"shape=({args.batch},{args.m},{args.k})x"
        f"({args.batch},{args.k},{args.n}) compile_args={compile_args}",
        flush=True,
    )

    expected = benchmark(
        "baseline",
        compiled_baseline,
        inputs,
        args.iterations,
    )
    candidates = {
        "compound": compiled_compound,
        "bundled": compiled_bundled,
        "scheduled": compiled_scheduled,
    }
    expected_cpu = expected.cpu()
    for name, function in candidates.items():
        actual = benchmark(name, function, inputs, args.iterations)
        torch.testing.assert_close(actual.cpu(), expected_cpu)
        print(f"{name}_correctness=pass", flush=True)

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        capture_trace("baseline", compiled_baseline, inputs, args.trace_dir)
        for name, function in candidates.items():
            capture_trace(name, function, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
