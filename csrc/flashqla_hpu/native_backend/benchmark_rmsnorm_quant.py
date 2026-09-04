import argparse
import statistics
import time
from pathlib import Path

import torch
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
HIDDEN_SIZE = 5_120
EPSILON = 1e-6
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def cguid_quant(x: torch.Tensor):
    scale = torch.ops.hpu.calculate_scale_for_cast(
        x,
        2,
        0,
        -1,
        True,
        FP8_MAX,
        1.0,
    )
    scale = scale + (1e-8 / FP8_MAX)
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        x,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float(), x


def hpex_rmsnorm_quant(x, residual, weight):
    residual_out = x + residual
    normalized = FusedRMSNorm.apply(
        residual_out.reshape(1, TOKENS, HIDDEN_SIZE),
        weight,
        EPSILON,
    ).reshape(TOKENS, HIDDEN_SIZE)
    quantized, scale, _ = cguid_quant(normalized)
    return quantized, scale, normalized, residual_out


def hpu_rmsnorm_quant(x, residual, weight):
    residual_out = x + residual
    normalized = torch.ops.hpu.rms_norm(
        residual_out,
        weight,
        EPSILON,
        None,
        False,
    )[0]
    quantized, scale, _ = cguid_quant(normalized)
    return quantized, scale, normalized, residual_out


def hpu_rmsnorm_fast_quant(x, residual, weight):
    residual_out = x + residual
    normalized = torch.ops.hpu.rms_norm_fast(
        residual_out,
        weight,
        EPSILON,
    )[0]
    quantized, scale, _ = cguid_quant(normalized)
    return quantized, scale, normalized, residual_out


def torch_rmsnorm_quant(x, residual, weight):
    residual_out = x + residual
    variance = residual_out.float().square().mean(dim=-1, keepdim=True)
    normalized = (
        residual_out.float()
        * torch.rsqrt(variance + EPSILON)
        * weight.float()
    ).to(x.dtype)
    quantized, scale, _ = cguid_quant(normalized)
    return quantized, scale, normalized, residual_out


FUNCTIONS = {
    "hpex": hpex_rmsnorm_quant,
    "hpu": hpu_rmsnorm_quant,
    "hpu_fast": hpu_rmsnorm_fast_quant,
    "torch": torch_rmsnorm_quant,
}


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
    saved = tuple(value.detach().clone() for value in output)
    torch.hpu.synchronize()
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return saved


def compare(reference, candidate, name):
    for output_name, expected, actual in zip(
        ("quantized", "scale", "normalized", "residual"),
        reference,
        candidate,
    ):
        expected_f32 = expected.float()
        actual_f32 = actual.float()
        difference = actual_f32 - expected_f32
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(expected_f32).clamp_min(1e-12)
        print(
            f"quality {name}_{output_name} "
            f"max_abs={float(difference.abs().max().cpu()):.9e} "
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
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--only", default="")
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    x = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn(
        (HIDDEN_SIZE,),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02 + 1.0
    inputs = x, residual, weight
    torch.hpu.synchronize()

    selected = args.only.split(",") if args.only else list(FUNCTIONS)
    selected = list(dict.fromkeys(["hpex", *selected]))
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled = {
        name: torch.compile(FUNCTIONS[name], **compile_args)
        for name in selected
    }
    print(
        f"tokens={TOKENS} hidden={HIDDEN_SIZE} compile_args={compile_args}",
        flush=True,
    )

    outputs = {}
    for name in selected:
        outputs[name] = benchmark(
            name,
            compiled[name],
            inputs,
            args.warmups,
            args.iterations,
        )
    for name, output in outputs.items():
        compare(outputs["hpex"], output, name)

    if args.trace_dir is not None:
        for name in selected:
            capture_trace(name, compiled[name], inputs, args.trace_dir)


if __name__ == "__main__":
    main()
