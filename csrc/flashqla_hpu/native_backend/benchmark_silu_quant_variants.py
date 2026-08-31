import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


INTERMEDIATE_SIZE = 17_408
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 240.0


def _quantize(activation: torch.Tensor):
    scale = torch.ops.hpu.calculate_scale_for_cast(
        activation,
        2,
        0,
        -1,
        True,
        FP8_MAX,
        1.0,
    )
    scale = scale + (1.0e-8 / FP8_MAX)
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        activation,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float()


def native_silu(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    return _quantize(torch.nn.functional.silu(gate) * up)


def explicit_sigmoid(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    return _quantize((gate * torch.sigmoid(gate)) * up)


def fp32_sigmoid_bf16_products(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate.float()).to(torch.bfloat16)
    return _quantize((gate * sigmoid) * up)


def fp32_silu_bf16_up(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    gate_fp32 = gate.float()
    silu = (gate_fp32 * torch.sigmoid(gate_fp32)).to(torch.bfloat16)
    return _quantize(silu * up)


def fp32_silu_fp32_up(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    gate_fp32 = gate.float()
    activation = gate_fp32 * torch.sigmoid(gate_fp32) * up.float()
    return _quantize(activation.to(torch.bfloat16))


def tanh_identity(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid = 0.5 * (torch.tanh(0.5 * gate) + 1.0)
    return _quantize((gate * sigmoid) * up)


def positive_exp_identity(gate_up: torch.Tensor):
    gate, up = gate_up.chunk(2, dim=-1)
    silu = gate - gate / (1.0 + torch.exp(gate))
    return _quantize(silu * up)


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
    cloned = tuple(value.clone() for value in output)
    torch.hpu.synchronize()
    return cloned, median


def report_quality(name, reference, candidate):
    reference_q, reference_scale = reference
    candidate_q, candidate_scale = candidate
    reference_dq = reference_q.float() * reference_scale
    candidate_dq = candidate_q.float() * candidate_scale
    difference = candidate_dq - reference_dq
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference_dq).clamp_min(1.0e-12)
    print(
        f"quality {name} "
        f"scale_max_abs="
        f"{float((candidate_scale - reference_scale).abs().max().cpu()):.9e} "
        f"quant_equal_fraction="
        f"{float((candidate_q == reference_q).float().mean().cpu()):.9f} "
        f"dequant_max_abs={float(difference.abs().max().cpu()):.9e} "
        f"dequant_relative_l2={float(relative_l2.cpu()):.9e}",
        flush=True,
    )


def report_fp32_reference_accuracy(name, gate_up, output, rows):
    gate_up_cpu = gate_up[:rows].float().cpu()
    gate, up = gate_up_cpu.chunk(2, dim=-1)
    reference = torch.nn.functional.silu(gate) * up
    quantized, scale = output
    candidate = quantized[:rows].float().cpu() * scale[:rows].cpu()
    difference = candidate - reference
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
    print(
        f"fp32_reference {name} rows={rows} "
        f"max_abs={float(difference.abs().max()):.9e} "
        f"relative_l2={float(relative_l2):.9e}",
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
    parser.add_argument("--tokens", type=int, default=16_384)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--reference-rows", type=int, default=64)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    gate_up = torch.randn(
        (args.tokens, 2 * INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    functions = {
        "native_silu": native_silu,
        "explicit_sigmoid": explicit_sigmoid,
        "fp32_sigmoid_bf16_products": fp32_sigmoid_bf16_products,
        "fp32_silu_bf16_up": fp32_silu_bf16_up,
        "fp32_silu_fp32_up": fp32_silu_fp32_up,
        "tanh_identity": tanh_identity,
        "positive_exp_identity": positive_exp_identity,
    }
    compiled = {
        name: torch.compile(function, **compile_args)
        for name, function in functions.items()
    }
    print(
        f"shape={tuple(gate_up.shape)} compile_args={compile_args}",
        flush=True,
    )

    reference, reference_ms = benchmark(
        "native_silu",
        compiled["native_silu"],
        (gate_up,),
        args.warmups,
        args.iterations,
    )
    report_fp32_reference_accuracy(
        "native_silu", gate_up, reference, args.reference_rows
    )
    for name in (
        "explicit_sigmoid",
        "fp32_sigmoid_bf16_products",
        "fp32_silu_bf16_up",
        "fp32_silu_fp32_up",
        "tanh_identity",
        "positive_exp_identity",
    ):
        candidate, candidate_ms = benchmark(
            name,
            compiled[name],
            (gate_up,),
            args.warmups,
            args.iterations,
        )
        report_quality(name, reference, candidate)
        report_fp32_reference_accuracy(
            name, gate_up, candidate, args.reference_rows
        )
        print(
            f"{name}_speedup={reference_ms / candidate_ms:.6f} "
            f"{name}_saved_ms={reference_ms - candidate_ms:.6f}",
            flush=True,
        )
        del candidate

    if args.trace_dir is not None:
        for name, function in compiled.items():
            capture_trace(name, function, (gate_up,), args.trace_dir)


if __name__ == "__main__":
    main()
