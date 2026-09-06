import argparse
import statistics
import time
from pathlib import Path

import torch
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
HIDDEN_SIZE = 5_120
INTERMEDIATE_SIZE = 17_408
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
RMS_EPSILON = 1e-6


def swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def swiglu_explicit_sigmoid(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return (gate * torch.sigmoid(gate)) * up


def current_per_token_quant(x: torch.Tensor):
    scale = (x.abs().amax(dim=-1, keepdim=True) + 1e-8) / FP8_MAX
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        x,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float()


def cguid_per_token_quant(x: torch.Tensor):
    scale = torch.ops.hpu.calculate_scale_for_cast(
        x,
        2,  # MAX_ABS_PCS_CALCULATION
        0,  # NO_SCALE_ROUNDING
        -1,
        True,
        float(FP8_MAX),
        1.0,
    )
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        x,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float()


def swiglu_current_quant(x: torch.Tensor):
    return current_per_token_quant(swiglu(x))


def swiglu_cguid_quant(x: torch.Tensor):
    return cguid_per_token_quant(swiglu(x))


def down_current(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
    quantized, scale = swiglu_current_quant(x)
    return torch.ops.hpu.fp8_gemm_v2(
        A=quantized,
        trans_A=False,
        B=weight,
        trans_B=False,
        D=None,
        out_dtype=torch.bfloat16,
        A_scale_inv=scale,
        B_scale_inv=weight_scale,
        bias=None,
        accumulate=False,
    )


def down_cguid(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
    quantized, scale = swiglu_cguid_quant(x)
    return torch.ops.hpu.fp8_gemm_v2(
        A=quantized,
        trans_A=False,
        B=weight,
        trans_B=False,
        D=None,
        out_dtype=torch.bfloat16,
        A_scale_inv=scale,
        B_scale_inv=weight_scale,
        bias=None,
        accumulate=False,
    )


def fp8_linear_current(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    quantized, scale = current_per_token_quant(x)
    return torch.ops.hpu.fp8_gemm_v2(
        A=quantized,
        trans_A=False,
        B=weight,
        trans_B=False,
        D=None,
        out_dtype=torch.bfloat16,
        A_scale_inv=scale,
        B_scale_inv=weight_scale,
        bias=None,
        accumulate=False,
    )


def fp8_linear_cguid(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    quantized, scale = cguid_per_token_quant(x)
    return torch.ops.hpu.fp8_gemm_v2(
        A=quantized,
        trans_A=False,
        B=weight,
        trans_B=False,
        D=None,
        out_dtype=torch.bfloat16,
        A_scale_inv=scale,
        B_scale_inv=weight_scale,
        bias=None,
        accumulate=False,
    )


def mlp_current(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    gate_up = fp8_linear_current(x, gate_up_weight, gate_up_scale)
    return fp8_linear_current(swiglu(gate_up), down_weight, down_scale)


def mlp_cguid_input(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    gate_up = fp8_linear_cguid(x, gate_up_weight, gate_up_scale)
    return fp8_linear_current(swiglu(gate_up), down_weight, down_scale)


def mlp_cguid_down(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    gate_up = fp8_linear_current(x, gate_up_weight, gate_up_scale)
    return fp8_linear_cguid(swiglu(gate_up), down_weight, down_scale)


def mlp_cguid_both(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    gate_up = fp8_linear_cguid(x, gate_up_weight, gate_up_scale)
    return fp8_linear_cguid(swiglu(gate_up), down_weight, down_scale)


def mlp_cguid_explicit_sigmoid(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    gate_up = fp8_linear_cguid(x, gate_up_weight, gate_up_scale)
    return fp8_linear_cguid(
        swiglu_explicit_sigmoid(gate_up),
        down_weight,
        down_scale,
    )


def post_norm_mlp_current(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    residual_out = x + residual
    normalized = FusedRMSNorm.apply(
        residual_out.reshape(1, TOKENS, HIDDEN_SIZE),
        norm_weight,
        RMS_EPSILON,
    ).reshape(TOKENS, HIDDEN_SIZE)
    output = mlp_current(
        normalized,
        gate_up_weight,
        gate_up_scale,
        down_weight,
        down_scale,
    )
    return output, residual_out


def post_norm_mlp_cguid(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
):
    residual_out = x + residual
    normalized = FusedRMSNorm.apply(
        residual_out.reshape(1, TOKENS, HIDDEN_SIZE),
        norm_weight,
        RMS_EPSILON,
    ).reshape(TOKENS, HIDDEN_SIZE)
    output = mlp_cguid_both(
        normalized,
        gate_up_weight,
        gate_up_scale,
        down_weight,
        down_scale,
    )
    return output, residual_out


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


def clone_output(output):
    if isinstance(output, tuple):
        cloned = tuple(value.clone() for value in output)
    else:
        cloned = output.clone()
    torch.hpu.synchronize()
    return cloned


def compare_quant(reference, candidate):
    reference_q, reference_scale = reference
    candidate_q, candidate_scale = candidate
    reference_dq = reference_q.float() * reference_scale.float()
    candidate_dq = candidate_q.float() * candidate_scale.float()
    difference = candidate_dq - reference_dq
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference_dq).clamp_min(1e-12)
    print(
        f"quant_scale_max_abs={float((candidate_scale - reference_scale).abs().max().cpu()):.9e} "
        f"quant_equal_fraction={float((candidate_q == reference_q).float().mean().cpu()):.9f} "
        f"quant_relative_l2={float(relative_l2.cpu()):.9e}",
        flush=True,
    )


def compare_output(reference, candidate):
    difference = candidate.float() - reference.float()
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    print(
        f"output_max_abs={float(difference.abs().max().cpu()):.9e} "
        f"output_equal_fraction={float((candidate == reference).float().mean().cpu()):.9f} "
        f"output_relative_l2={float(relative_l2.cpu()):.9e}",
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
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--explicit-sigmoid-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    gate_up = torch.randn(
        (TOKENS, INTERMEDIATE_SIZE * 2),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    weight_bf16 = torch.randn(
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    weight_scale = (weight_bf16.abs().amax() + 1e-8) / FP8_MAX
    weight = torch.ops.hpu.cast_to_fp8_v2(
        weight_bf16,
        weight_scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    del weight_bf16
    hidden = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    residual = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    norm_weight = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02 + 1.0
    gate_up_weight_bf16 = torch.randn(
        (HIDDEN_SIZE, INTERMEDIATE_SIZE * 2),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    gate_up_weight_scale = (
        gate_up_weight_bf16.abs().amax() + 1e-8
    ) / FP8_MAX
    gate_up_weight = torch.ops.hpu.cast_to_fp8_v2(
        gate_up_weight_bf16,
        gate_up_weight_scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    del gate_up_weight_bf16
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    functions = {
        "current_quant": current_per_token_quant,
        "cguid_quant": cguid_per_token_quant,
        "swiglu_current_quant": swiglu_current_quant,
        "swiglu_cguid_quant": swiglu_cguid_quant,
        "down_current": down_current,
        "down_cguid": down_cguid,
        "mlp_current": mlp_current,
        "mlp_cguid_input": mlp_cguid_input,
        "mlp_cguid_down": mlp_cguid_down,
        "mlp_cguid_both": mlp_cguid_both,
        "mlp_cguid_explicit_sigmoid": mlp_cguid_explicit_sigmoid,
        "post_norm_mlp_current": post_norm_mlp_current,
        "post_norm_mlp_cguid": post_norm_mlp_cguid,
    }
    compiled = {
        name: torch.compile(function, **compile_args)
        for name, function in functions.items()
    }
    print(
        f"gate_up_shape={tuple(gate_up.shape)} weight_shape={tuple(weight.shape)} "
        f"compile_args={compile_args}",
        flush=True,
    )

    if args.explicit_sigmoid_only:
        mlp_inputs = (
            hidden,
            gate_up_weight,
            gate_up_weight_scale,
            weight,
            weight_scale,
        )
        reference = clone_output(benchmark(
            "mlp_cguid_both",
            compiled["mlp_cguid_both"],
            mlp_inputs,
            args.iterations,
        ))
        candidate = benchmark(
            "mlp_cguid_explicit_sigmoid",
            compiled["mlp_cguid_explicit_sigmoid"],
            mlp_inputs,
            args.iterations,
        )
        compare_output(reference, candidate)
        if args.trace_dir is not None:
            capture_trace(
                "mlp_cguid_both",
                compiled["mlp_cguid_both"],
                mlp_inputs,
                args.trace_dir,
            )
            capture_trace(
                "mlp_cguid_explicit_sigmoid",
                compiled["mlp_cguid_explicit_sigmoid"],
                mlp_inputs,
                args.trace_dir,
            )
        return

    activation = swiglu(gate_up)
    torch.hpu.synchronize()
    current_quant = clone_output(benchmark(
        "current_quant",
        compiled["current_quant"],
        (activation,),
        args.iterations,
    ))
    cguid_quant = benchmark(
        "cguid_quant",
        compiled["cguid_quant"],
        (activation,),
        args.iterations,
    )
    compare_quant(current_quant, cguid_quant)
    del activation

    current_swiglu = clone_output(benchmark(
        "swiglu_current_quant",
        compiled["swiglu_current_quant"],
        (gate_up,),
        args.iterations,
    ))
    cguid_swiglu = benchmark(
        "swiglu_cguid_quant",
        compiled["swiglu_cguid_quant"],
        (gate_up,),
        args.iterations,
    )
    compare_quant(current_swiglu, cguid_swiglu)

    current_down = clone_output(benchmark(
        "down_current",
        compiled["down_current"],
        (gate_up, weight, weight_scale),
        args.iterations,
    ))
    cguid_down = benchmark(
        "down_cguid",
        compiled["down_cguid"],
        (gate_up, weight, weight_scale),
        args.iterations,
    )
    compare_output(current_down, cguid_down)

    mlp_inputs = (
        hidden,
        gate_up_weight,
        gate_up_weight_scale,
        weight,
        weight_scale,
    )
    current_mlp = clone_output(benchmark(
        "mlp_current",
        compiled["mlp_current"],
        mlp_inputs,
        args.iterations,
    ))
    mlp_outputs = {}
    for name in ("mlp_cguid_input", "mlp_cguid_down", "mlp_cguid_both"):
        mlp_outputs[name] = benchmark(
            name,
            compiled[name],
            mlp_inputs,
            args.iterations,
        )
        print(f"{name}_difference", flush=True)
        compare_output(current_mlp, mlp_outputs[name])

    post_norm_inputs = (
        hidden,
        residual,
        norm_weight,
        gate_up_weight,
        gate_up_weight_scale,
        weight,
        weight_scale,
    )
    current_post_norm = clone_output(benchmark(
        "post_norm_mlp_current",
        compiled["post_norm_mlp_current"],
        post_norm_inputs,
        args.iterations,
    ))
    cguid_post_norm = benchmark(
        "post_norm_mlp_cguid",
        compiled["post_norm_mlp_cguid"],
        post_norm_inputs,
        args.iterations,
    )
    print("post_norm_mlp_cguid_output_difference", flush=True)
    compare_output(current_post_norm[0], cguid_post_norm[0])
    print("post_norm_mlp_cguid_residual_difference", flush=True)
    compare_output(current_post_norm[1], cguid_post_norm[1])

    if args.trace_dir is not None:
        capture_trace(
            "swiglu_current_quant",
            compiled["swiglu_current_quant"],
            (gate_up,),
            args.trace_dir,
        )
        capture_trace(
            "swiglu_cguid_quant",
            compiled["swiglu_cguid_quant"],
            (gate_up,),
            args.trace_dir,
        )
        capture_trace(
            "down_current",
            compiled["down_current"],
            (gate_up, weight, weight_scale),
            args.trace_dir,
        )
        capture_trace(
            "down_cguid",
            compiled["down_cguid"],
            (gate_up, weight, weight_scale),
            args.trace_dir,
        )
        capture_trace(
            "mlp_current",
            compiled["mlp_current"],
            mlp_inputs,
            args.trace_dir,
        )
        capture_trace(
            "mlp_cguid_both",
            compiled["mlp_cguid_both"],
            mlp_inputs,
            args.trace_dir,
        )
        capture_trace(
            "post_norm_mlp_current",
            compiled["post_norm_mlp_current"],
            post_norm_inputs,
            args.trace_dir,
        )
        capture_trace(
            "post_norm_mlp_cguid",
            compiled["post_norm_mlp_cguid"],
            post_norm_inputs,
            args.trace_dir,
        )


if __name__ == "__main__":
    main()
