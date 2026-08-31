import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.extension.ops import (
    FP8_MAX,
    apply_fp8_linear_hpu,
    dynamic_quant,
)
from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
HIDDEN_SIZE = 5_120
QKVZ_SIZE = 16_384
BA_SIZE = 96
FP8_DTYPE = torch.float8_e4m3fn


def separate_projection_quant(
    x,
    qkvz_weight,
    qkvz_scale,
    ba_weight,
    ba_scale,
    combined_weight,
    combined_scale,
):
    del combined_weight, combined_scale
    qkvz = apply_fp8_linear_hpu(
        input=x,
        weight=qkvz_weight,
        weight_scale=qkvz_scale,
        trans_B=False,
    )
    ba = apply_fp8_linear_hpu(
        input=x,
        weight=ba_weight,
        weight_scale=ba_scale,
        trans_B=False,
    )
    return qkvz, ba


def _prequantized_linear(x_fp8, x_scale, weight, weight_scale):
    return torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=weight,
        trans_B=False,
        D=None,
        out_dtype=torch.bfloat16,
        A_scale_inv=x_scale,
        B_scale_inv=weight_scale,
        bias=None,
        accumulate=False,
    )


def shared_projection_quant(
    x,
    qkvz_weight,
    qkvz_scale,
    ba_weight,
    ba_scale,
    combined_weight,
    combined_scale,
):
    del combined_weight, combined_scale
    x_fp8, x_scale = dynamic_quant(x)
    qkvz = _prequantized_linear(x_fp8, x_scale, qkvz_weight, qkvz_scale)
    ba = _prequantized_linear(x_fp8, x_scale, ba_weight, ba_scale)
    return qkvz, ba


def combined_projection_quant(
    x,
    qkvz_weight,
    qkvz_scale,
    ba_weight,
    ba_scale,
    combined_weight,
    combined_scale,
):
    del qkvz_weight, qkvz_scale, ba_weight, ba_scale
    projected = apply_fp8_linear_hpu(
        input=x,
        weight=combined_weight,
        weight_scale=combined_scale,
        trans_B=False,
    )
    return projected[:, :QKVZ_SIZE], projected[:, QKVZ_SIZE:]


def make_weight(rows, columns):
    weight_bf16 = torch.randn(
        (rows, columns),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    scale = (weight_bf16.abs().amax(dim=0, keepdim=True) + 1e-8) / FP8_MAX
    weight = torch.ops.hpu.cast_to_fp8_v2(
        weight_bf16,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return weight, scale.squeeze(0).float()


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
    output = tuple(value.clone() for value in output)
    torch.hpu.synchronize()
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return output


def compare(reference, candidate):
    for name, reference_value, candidate_value in zip(
        ("qkvz", "ba"),
        reference,
        candidate,
    ):
        difference = candidate_value.float() - reference_value.float()
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(reference_value.float()).clamp_min(1e-12)
        print(
            f"quality {name} "
            f"max_abs={float(difference.abs().max().cpu()):.9e} "
            f"equal_fraction={float((candidate_value == reference_value).float().mean().cpu()):.9f} "
            f"relative_l2={float(relative_l2.cpu()):.9e}",
            flush=True,
        )


def alternating_benchmark(separate, shared, combined, inputs, iterations):
    samples = {"separate": [], "shared": [], "combined": []}
    for iteration in range(iterations):
        order = (
            (
                ("separate", separate),
                ("shared", shared),
                ("combined", combined),
            )
            if iteration % 2 == 0
            else (
                ("combined", combined),
                ("shared", shared),
                ("separate", separate),
            )
        )
        for name, function in order:
            start = time.perf_counter()
            function(*inputs)
            torch.hpu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1_000)
    for name, values in samples.items():
        print(
            f"alternating_{name} median_ms={statistics.median(values):.6f} "
            f"min_ms={min(values):.6f} samples_ms={values}",
            flush=True,
        )


def capture_trace(separate, shared, combined, inputs, trace_dir):
    trace_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1)
    handler = torch.profiler.tensorboard_trace_handler(
        str(trace_dir),
        worker_name="shared_projection_quant",
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
        with_stack=False,
    ) as profiler:
        for name, function in (
            ("separate", separate),
            ("shared", shared),
            ("combined", combined),
        ):
            with torch.profiler.record_function(name):
                function(*inputs)
                torch.hpu.synchronize()
            profiler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()

    torch.manual_seed(739251)
    x = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    qkvz_weight, qkvz_scale = make_weight(HIDDEN_SIZE, QKVZ_SIZE)
    ba_weight, ba_scale = make_weight(HIDDEN_SIZE, BA_SIZE)
    combined_weight = torch.cat(
        (qkvz_weight, ba_weight),
        dim=1,
    ).contiguous()
    combined_scale = torch.cat((qkvz_scale, ba_scale), dim=0).contiguous()
    inputs = (
        x,
        qkvz_weight,
        qkvz_scale,
        ba_weight,
        ba_scale,
        combined_weight,
        combined_scale,
    )
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    separate = torch.compile(separate_projection_quant, **compile_args)
    shared = torch.compile(shared_projection_quant, **compile_args)
    combined = torch.compile(combined_projection_quant, **compile_args)
    print(
        f"x={tuple(x.shape)} qkvz_weight={tuple(qkvz_weight.shape)} "
        f"ba_weight={tuple(ba_weight.shape)} compile_args={compile_args}",
        flush=True,
    )
    reference = benchmark(
        "separate_projection_quant",
        separate,
        inputs,
        args.warmups,
        args.iterations,
    )
    candidate = benchmark(
        "shared_projection_quant",
        shared,
        inputs,
        args.warmups,
        args.iterations,
    )
    compare(reference, candidate)
    combined_output = benchmark(
        "combined_projection_quant",
        combined,
        inputs,
        args.warmups,
        args.iterations,
    )
    compare(reference, combined_output)
    alternating_benchmark(
        separate,
        shared,
        combined,
        inputs,
        args.iterations,
    )
    if args.trace_dir is not None:
        capture_trace(separate, shared, combined, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
