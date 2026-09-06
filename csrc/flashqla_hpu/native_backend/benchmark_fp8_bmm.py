import argparse
import statistics
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


BATCH = 12_288
ROWS = 64
INNER = 128


def bf16_kkt_dot(k):
    return torch.bmm(k, k.transpose(1, 2))


def make_fp8_kkt_dot(scale_inv):
    scale = 1.0 / scale_inv

    def fp8_kkt_dot(k):
        k_fp8 = torch.ops.hpu.cast_to_fp8_v2(
            k,
            scale,
            False,
            False,
            torch.float8_e4m3fn,
        )[0]
        return torch.ops.hpu.fp8_gemm_v2(
            A=k_fp8,
            trans_A=False,
            B=k_fp8,
            trans_B=True,
            D=None,
            out_dtype=torch.bfloat16,
            A_scale_inv=scale_inv,
            B_scale_inv=scale_inv,
            bias=None,
            accumulate=False,
        )

    return fp8_kkt_dot


def make_fp8_cast(scale_inv):
    scale = 1.0 / scale_inv

    def fp8_cast(k):
        return torch.ops.hpu.cast_to_fp8_v2(
            k,
            scale,
            False,
            False,
            torch.float8_e4m3fn,
        )[0]

    return fp8_cast


def make_prequantized_fp8_kkt_dot(scale_inv):
    def prequantized_fp8_kkt_dot(k_fp8):
        return torch.ops.hpu.fp8_gemm_v2(
            A=k_fp8,
            trans_A=False,
            B=k_fp8,
            trans_B=True,
            D=None,
            out_dtype=torch.bfloat16,
            A_scale_inv=scale_inv,
            B_scale_inv=scale_inv,
            bias=None,
            accumulate=False,
        )

    return prequantized_fp8_kkt_dot


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


def report_difference(actual, expected, name):
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    relative_l2 = torch.linalg.vector_norm(actual_float - expected_float)
    relative_l2 /= torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    print(
        f"{name}_max_abs={float(difference.max().cpu()):.9e} "
        f"mean_abs={float(difference.mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument(
        "--scale-inv",
        type=float,
        action="append",
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scales = args.scale_inv or [2**-8, 2**-9, 2**-10]
    torch.manual_seed(739251)
    k = torch.randn(
        BATCH,
        ROWS,
        INNER,
        dtype=torch.bfloat16,
        device="hpu",
    )
    k /= torch.linalg.vector_norm(k.float(), dim=-1, keepdim=True)
    inputs = (k,)

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    bf16 = torch.compile(bf16_kkt_dot, **compile_args)
    fp8_variants = {
        scale_inv: torch.compile(
            make_fp8_kkt_dot(scale_inv),
            **compile_args,
        )
        for scale_inv in scales
    }
    cast = torch.compile(make_fp8_cast(scales[0]), **compile_args)
    prequantized = torch.compile(
        make_prequantized_fp8_kkt_dot(scales[0]),
        **compile_args,
    )
    print(
        f"shape=({BATCH},{ROWS},{INNER}) "
        f"output=({BATCH},{ROWS},{ROWS})",
        flush=True,
    )
    expected = benchmark("bf16", bf16, inputs, args.iterations)
    for scale_inv, function in fp8_variants.items():
        actual = benchmark(
            f"fp8_scale_inv_{scale_inv:.9g}",
            function,
            inputs,
            args.iterations,
        )
        report_difference(
            actual[::1536],
            expected[::1536],
            f"fp8_scale_inv_{scale_inv:.9g}",
        )

    k_fp8 = cast(k)
    torch.hpu.synchronize()
    benchmark("fp8_cast_only", cast, inputs, args.iterations)
    prequantized_output = benchmark(
        "fp8_prequantized_gemm",
        prequantized,
        (k_fp8,),
        args.iterations,
    )
    report_difference(
        prequantized_output[::1536],
        expected[::1536],
        "fp8_prequantized",
    )


if __name__ == "__main__":
    main()
