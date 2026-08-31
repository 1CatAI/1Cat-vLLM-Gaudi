import argparse
import json
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def ordinary_quant(x: torch.Tensor):
    scale = (x.abs().amax(dim=-1, keepdim=True) + 1e-8) / FP8_MAX
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        x,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, scale.float()


def compound_guid_quant(x: torch.Tensor):
    scale = torch.ops.hpu.calculate_scale_for_cast(
        x,
        2,
        0,
        -1,
        True,
        float(FP8_MAX),
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
    return quantized, scale.float()


def synchronize_and_clone(output):
    torch.hpu.synchronize()
    return tuple(value.clone() for value in output)


def timed(function, x, iterations):
    for _ in range(3):
        output = function(x)
    torch.hpu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(x)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    return output, sorted(samples)[len(samples) // 2]


def compare(tokens, ordinary, compound, ordinary_ms, compound_ms):
    ordinary_q, ordinary_scale = ordinary
    compound_q, compound_scale = compound
    ordinary_dq = ordinary_q.float() * ordinary_scale
    compound_dq = compound_q.float() * compound_scale
    delta = compound_dq - ordinary_dq
    relative_l2 = torch.linalg.vector_norm(delta)
    relative_l2 /= torch.linalg.vector_norm(ordinary_dq).clamp_min(1e-12)
    result = {
        "tokens": tokens,
        "ordinary_ms": ordinary_ms,
        "compound_ms": compound_ms,
        "scale_max_abs": float(
            (compound_scale - ordinary_scale).abs().max().cpu()
        ),
        "scale_equal_fraction": float(
            (compound_scale == ordinary_scale).float().mean().cpu()
        ),
        "quant_equal_fraction": float(
            (compound_q == ordinary_q).float().mean().cpu()
        ),
        "dequant_max_abs": float(delta.abs().max().cpu()),
        "dequant_relative_l2": float(relative_l2.cpu()),
        "ordinary_scale_min": float(ordinary_scale.min().cpu()),
        "compound_scale_min": float(compound_scale.min().cpu()),
    }
    print(json.dumps(result), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 16, 128, 2048, 16384],
    )
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    ordinary = torch.compile(ordinary_quant, **compile_args)
    compound = torch.compile(compound_guid_quant, **compile_args)

    for tokens in args.tokens:
        x = torch.randn(
            (tokens, args.hidden_size),
            dtype=torch.bfloat16,
            device="hpu",
        ) * 0.5
        if tokens >= 2:
            x[0].zero_()
            x[1].fill_(torch.finfo(torch.bfloat16).tiny)
        ordinary_output, ordinary_ms = timed(
            ordinary,
            x,
            args.iterations,
        )
        ordinary_output = synchronize_and_clone(ordinary_output)
        compound_output, compound_ms = timed(
            compound,
            x,
            args.iterations,
        )
        compare(
            tokens,
            ordinary_output,
            compound_output,
            ordinary_ms,
            compound_ms,
        )


if __name__ == "__main__":
    main()
