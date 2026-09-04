import argparse
import statistics
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
HIDDEN_SIZE = 5_120
INTERMEDIATE_SIZE = 17_408
FP8_DTYPE = torch.float8_e4m3fn
GAUDI2_FP8_MAX = float(torch.finfo(torch.float8_e4m3fnuz).max)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def production_quant(x: torch.Tensor):
    scale = torch.ops.hpu.calculate_scale_for_cast(
        x,
        2,
        0,
        -1,
        True,
        GAUDI2_FP8_MAX,
        1.0,
    )
    scale = scale + (1e-8 / GAUDI2_FP8_MAX)
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        x,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float()


def jit_quant(x: torch.Tensor):
    return torch.ops.hpu.cast_to_fp8_just_in_time(
        x,
        [1, x.shape[-1]],
        out_dtype=FP8_DTYPE,
        scale_dtype=torch.float32,
    )


def swiglu_production_quant(x: torch.Tensor):
    return production_quant(swiglu(x))


def swiglu_jit_quant(x: torch.Tensor):
    return jit_quant(swiglu(x))


def down_production(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    quantized, scale = swiglu_production_quant(x)
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


def down_jit(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    quantized, scale = swiglu_jit_quant(x)
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


def _down_jit_layout(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    clone_quantized: bool,
    clone_scale: bool,
    roundtrip_scale: bool,
):
    quantized, scale = swiglu_jit_quant(x)
    if clone_quantized:
        quantized = quantized.clone(memory_format=torch.contiguous_format)
    if clone_scale:
        scale = scale.clone(memory_format=torch.contiguous_format)
    if roundtrip_scale:
        scale = scale.to(torch.bfloat16).to(torch.float32)
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


def down_jit_clone_quantized(*args):
    return _down_jit_layout(*args, True, False, False)


def down_jit_clone_scale(*args):
    return _down_jit_layout(*args, False, True, False)


def down_jit_clone_both(*args):
    return _down_jit_layout(*args, True, True, False)


def down_jit_roundtrip_scale(*args):
    return _down_jit_layout(*args, False, False, True)


def fp8_gemm_from_quantized(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
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


def make_split_down(compiled_quant, compiled_gemm):
    def run(x, weight, weight_scale):
        quantized, scale = compiled_quant(x)
        return compiled_gemm(quantized, scale, weight, weight_scale)

    return run


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
    if isinstance(output, tuple):
        saved = tuple(value.detach().clone() for value in output)
    else:
        saved = output.detach().clone()
    torch.hpu.synchronize()
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return saved


def compare_quant(reference, candidate, name):
    reference_q, reference_scale = reference
    candidate_q, candidate_scale = candidate
    reference_dq = reference_q.float() * reference_scale.float()
    candidate_dq = candidate_q.float() * candidate_scale.float()
    difference = candidate_dq - reference_dq
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference_dq).clamp_min(1e-12)
    scale_ratio = candidate_scale.float() / reference_scale.float()
    print(
        f"quality {name} "
        f"quant_equal_fraction="
        f"{float((candidate_q == reference_q).float().mean().cpu()):.9f} "
        f"scale_equal_fraction="
        f"{float((candidate_scale == reference_scale).float().mean().cpu()):.9f} "
        f"dequant_relative_l2={float(relative_l2.cpu()):.9e} "
        f"dequant_max_abs={float(difference.abs().max().cpu()):.9e} "
        f"scale_ratio_min={float(scale_ratio.min().cpu()):.9f} "
        f"scale_ratio_max={float(scale_ratio.max().cpu()):.9f}",
        flush=True,
    )


def compare_output(reference, candidate):
    difference = candidate.float() - reference.float()
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    print(
        "quality down "
        f"equal_fraction="
        f"{float((candidate == reference).float().mean().cpu()):.9f} "
        f"relative_l2={float(relative_l2.cpu()):.9e} "
        f"max_abs={float(difference.abs().max().cpu()):.9e}",
        flush=True,
    )


def quantize_weight():
    weight_bf16 = torch.randn(
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    scale = (weight_bf16.abs().amax() + 1e-8) / GAUDI2_FP8_MAX
    weight = torch.ops.hpu.cast_to_fp8_v2(
        weight_bf16,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return weight, scale.float()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--down-layout-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    gate_up = torch.randn(
        (TOKENS, INTERMEDIATE_SIZE * 2),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    weight, weight_scale = quantize_weight()
    activation = swiglu(gate_up)
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    functions = {
        "production_quant": production_quant,
        "jit_quant": jit_quant,
        "swiglu_production_quant": swiglu_production_quant,
        "swiglu_jit_quant": swiglu_jit_quant,
        "down_production": down_production,
        "down_jit": down_jit,
        "down_jit_clone_quantized": down_jit_clone_quantized,
        "down_jit_clone_scale": down_jit_clone_scale,
        "down_jit_clone_both": down_jit_clone_both,
        "down_jit_roundtrip_scale": down_jit_roundtrip_scale,
    }
    compiled = {
        name: torch.compile(function, **compile_args)
        for name, function in functions.items()
    }
    compiled_gemm = torch.compile(
        fp8_gemm_from_quantized,
        **compile_args,
    )
    down_jit_split = make_split_down(
        compiled["swiglu_jit_quant"],
        compiled_gemm,
    )
    print(
        f"tokens={TOKENS} hidden={HIDDEN_SIZE} "
        f"intermediate={INTERMEDIATE_SIZE} fp8_max={GAUDI2_FP8_MAX} "
        f"compile_args={compile_args}",
        flush=True,
    )

    down_inputs = (gate_up, weight, weight_scale)
    if args.down_layout_only:
        reference = benchmark(
            "down_production",
            compiled["down_production"],
            down_inputs,
            args.warmups,
            args.iterations,
        )
        for name in (
            "down_jit",
            "down_jit_clone_quantized",
            "down_jit_clone_scale",
            "down_jit_clone_both",
            "down_jit_roundtrip_scale",
        ):
            output = benchmark(
                name,
                compiled[name],
                down_inputs,
                args.warmups,
                args.iterations,
            )
            compare_output(reference, output)
        split_output = benchmark(
            "down_jit_split_compiled_recipes",
            down_jit_split,
            down_inputs,
            args.warmups,
            args.iterations,
        )
        compare_output(reference, split_output)
        return

    production = benchmark(
        "production_quant",
        compiled["production_quant"],
        (activation,),
        args.warmups,
        args.iterations,
    )
    jit = benchmark(
        "jit_quant",
        compiled["jit_quant"],
        (activation,),
        args.warmups,
        args.iterations,
    )
    compare_quant(production, jit, "quant")

    production_swiglu = benchmark(
        "swiglu_production_quant",
        compiled["swiglu_production_quant"],
        (gate_up,),
        args.warmups,
        args.iterations,
    )
    jit_swiglu = benchmark(
        "swiglu_jit_quant",
        compiled["swiglu_jit_quant"],
        (gate_up,),
        args.warmups,
        args.iterations,
    )
    compare_quant(production_swiglu, jit_swiglu, "swiglu_quant")

    production_down = benchmark(
        "down_production",
        compiled["down_production"],
        down_inputs,
        args.warmups,
        args.iterations,
    )
    jit_down = benchmark(
        "down_jit",
        compiled["down_jit"],
        down_inputs,
        args.warmups,
        args.iterations,
    )
    compare_output(production_down, jit_down)


if __name__ == "__main__":
    main()
