import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


def reference(a0, scores, gate, beta):
    gate_delta = gate.unsqueeze(-1) - gate.unsqueeze(-2)
    decay = torch.tril(torch.exp(torch.tril(gate_delta))).to(a0.dtype)
    ag = a0 * decay * beta.unsqueeze(-2)
    attention = torch.tril(scores * decay)
    return ag, attention


def custom(a0, scores, gate, beta):
    return torch.ops.custom_op.flashqla_pair_transform_bf16_gaudi2(
        a0,
        scores,
        gate[..., ::2].contiguous(),
        gate[..., 1::2].contiguous(),
        beta,
    )


def make_inputs(outer, heads, extreme=False):
    torch.manual_seed(739251)
    a0 = torch.randn(
        outer,
        heads,
        64,
        64,
        dtype=torch.bfloat16,
        device="hpu",
    )
    scores = torch.randn_like(a0)
    step_scale = 15.0 if extreme else 0.1
    gate_steps = -torch.rand(
        outer,
        heads,
        64,
        dtype=torch.float32,
        device="hpu",
    ) * step_scale
    gate = torch.cumsum(gate_steps, dim=-1)
    beta = torch.rand(
        outer,
        heads,
        64,
        dtype=torch.bfloat16,
        device="hpu",
    )
    return a0, scores, gate, beta


def check_accuracy(extreme):
    inputs = make_inputs(2, 3, extreme=extreme)
    expected = reference(*inputs)
    actual = custom(*inputs)
    torch.hpu.synchronize()
    label = "extreme" if extreme else "ordinary"
    for name, expected_tensor, actual_tensor in zip(
        ("ag", "attention"),
        expected,
        actual,
        strict=True,
    ):
        expected_cpu = expected_tensor.float().cpu()
        actual_cpu = actual_tensor.float().cpu()
        diff = actual_cpu - expected_cpu
        relative_l2 = diff.norm() / expected_cpu.norm()
        print(
            f"{label}_{name}_max_abs={diff.abs().max().item():.9g} "
            f"relative_l2={relative_l2.item():.9g} "
            f"finite={bool(torch.isfinite(actual_cpu).all())}",
            flush=True,
        )
        torch.testing.assert_close(
            actual_cpu,
            expected_cpu,
            rtol=0.02,
            atol=0.02,
        )


def benchmark(name, function, inputs, warmups, repeats, trace_dir):
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled = torch.compile(function, **compile_args)
    started = time.perf_counter()
    output = compiled(*inputs)
    torch.hpu.synchronize()
    compile_s = time.perf_counter() - started
    for _ in range(warmups):
        output = compiled(*inputs)
    torch.hpu.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = compiled(*inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    checksum = sum(tensor.float().sum() for tensor in output)
    torch.hpu.synchronize()
    print(
        f"{name} compile_s={compile_s:.3f} "
        f"median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples} "
        f"checksum={checksum.cpu().item():.9g}",
        flush=True,
    )

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1)
        handler = torch.profiler.tensorboard_trace_handler(
            str(trace_dir),
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
                compiled(*inputs)
                torch.hpu.synchronize()
                profiler.step()
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=256)
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()

    library = Path(__file__).parents[1] / "flashqla_pair_transform_pt2.cpython-312-x86_64-linux-gnu.so"
    torch.ops.load_library(str(library.resolve()))
    check_accuracy(extreme=False)
    check_accuracy(extreme=True)
    inputs = make_inputs(args.outer, args.heads)
    reference_ms = benchmark(
        "pytorch_reference",
        reference,
        inputs,
        args.warmups,
        args.repeats,
        args.trace_dir,
    )
    custom_ms = benchmark(
        "custom_bf16_tpc",
        custom,
        inputs,
        args.warmups,
        args.repeats,
        args.trace_dir,
    )
    print(
        f"speedup={reference_ms / custom_ms:.6f} "
        f"saved_ms={reference_ms - custom_ms:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
