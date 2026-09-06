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
from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn_token_major,
)
from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
HIDDEN_SIZE = 5_120
QKV_SIZE = 10_240
Z_SIZE = 6_144
FP8_DTYPE = torch.float8_e4m3fn
HEAD_DIM = 128
QK_SIZE = 2_048
VALUE_SIZE = 6_144


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


def combined_projection(x, token_mask, weight, weight_scale):
    qkvz = apply_fp8_linear_hpu(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        trans_B=False,
    )
    qkv = qkvz[:, :QKV_SIZE] * token_mask
    z = qkvz[:, QKV_SIZE:].reshape(TOKENS, 48, 128)
    return qkv, z


def split_projection(
    x,
    token_mask,
    qkv_weight,
    qkv_weight_scale,
    z_weight,
    z_weight_scale,
):
    del token_mask
    x_fp8, x_scale = dynamic_quant(x)
    qkv = _prequantized_linear(
        x_fp8,
        x_scale,
        qkv_weight,
        qkv_weight_scale,
    )
    z = _prequantized_linear(
        x_fp8,
        x_scale,
        z_weight,
        z_weight_scale,
    ).reshape(TOKENS, 48, 128)
    return qkv, z


def split_projection_masked(
    x,
    token_mask,
    qkv_weight,
    qkv_weight_scale,
    z_weight,
    z_weight_scale,
):
    qkv, z = split_projection(
        x,
        token_mask,
        qkv_weight,
        qkv_weight_scale,
        z_weight,
        z_weight_scale,
    )
    return qkv * token_mask, z


def _conv_qkv(
    qkv,
    conv_weight,
    conv_bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    post_conv = hpu_causal_conv1d_fn_token_major(
        x=qkv,
        weight=conv_weight.transpose(0, 1).contiguous(),
        bias=conv_bias,
        activation="silu",
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        block_size_to_align=0,
        is_prompt=True,
    )
    query, key = (
        torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
            post_conv
        )
    )
    value = post_conv[:, 2 * QK_SIZE :].reshape(
        TOKENS,
        48,
        HEAD_DIM,
    ).contiguous()
    return query, key, value, conv_state


def combined_producer_chain(
    x,
    token_mask,
    weight,
    weight_scale,
    conv_weight,
    conv_bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    qkv, z = combined_projection(x, token_mask, weight, weight_scale)
    query, key, value, conv_state = _conv_qkv(
        qkv,
        conv_weight,
        conv_bias,
        conv_state,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )
    return query, key, value, z, conv_state


def split_producer_chain(
    x,
    token_mask,
    qkv_weight,
    qkv_weight_scale,
    z_weight,
    z_weight_scale,
    conv_weight,
    conv_bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    qkv, z = split_projection(
        x,
        token_mask,
        qkv_weight,
        qkv_weight_scale,
        z_weight,
        z_weight_scale,
    )
    query, key, value, conv_state = _conv_qkv(
        qkv,
        conv_weight,
        conv_bias,
        conv_state,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )
    return query, key, value, z, conv_state


def make_weight():
    weight_bf16 = torch.randn(
        (HIDDEN_SIZE, QKV_SIZE + Z_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    scale = (
        weight_bf16.abs().amax(dim=0, keepdim=True) + 1e-8
    ) / FP8_MAX
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
    median = statistics.median(samples)
    print(
        f"{name} median_ms={median:.6f} min_ms={min(samples):.6f} "
        f"samples_ms={samples}",
        flush=True,
    )
    return output, median


def compare(reference, candidate, prefix):
    for name, expected, actual in zip(("qkv", "z"), reference, candidate):
        difference = actual.float() - expected.float()
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(
            expected.float()
        ).clamp_min(1e-12)
        print(
            f"{prefix}_{name}_max_abs="
            f"{float(difference.abs().max().cpu()):.9e} "
            f"equal_fraction="
            f"{float((actual == expected).float().mean().cpu()):.9f} "
            f"relative_l2={float(relative_l2.cpu()):.9e}",
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
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=13)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--producer-chain", action="store_true")
    parser.add_argument("--extension", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.extension is not None:
        torch.ops.load_library(str(args.extension.resolve()))
    torch.manual_seed(739251)
    x = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    )
    token_mask = torch.ones(
        (TOKENS, 1),
        dtype=torch.bfloat16,
        device="hpu",
    )
    weight, weight_scale = make_weight()
    qkv_weight = weight[:, :QKV_SIZE].contiguous()
    qkv_weight_scale = weight_scale[:QKV_SIZE].contiguous()
    z_weight = weight[:, QKV_SIZE:].contiguous()
    z_weight_scale = weight_scale[QKV_SIZE:].contiguous()
    conv_weight = torch.randn(
        (4, QKV_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.08
    conv_bias = torch.randn(
        (QKV_SIZE,),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.01
    conv_state = torch.zeros(
        (2, 3, QKV_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    )
    query_start_loc = torch.tensor(
        [0, TOKENS],
        dtype=torch.int32,
        device="hpu",
    )
    cache_indices = torch.tensor([0], dtype=torch.int64, device="hpu")
    has_initial_state = torch.tensor(
        [False],
        dtype=torch.bool,
        device="hpu",
    )
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    if args.producer_chain:
        if args.extension is None:
            raise ValueError("--producer-chain requires --extension")
        combined_chain = torch.compile(
            combined_producer_chain,
            **compile_args,
        )
        split_chain = torch.compile(
            split_producer_chain,
            **compile_args,
        )
        combined_chain_inputs = (
            x,
            token_mask,
            weight,
            weight_scale,
            conv_weight,
            conv_bias,
            conv_state.clone(),
            query_start_loc,
            cache_indices,
            has_initial_state,
        )
        split_chain_inputs = (
            x,
            token_mask,
            qkv_weight,
            qkv_weight_scale,
            z_weight,
            z_weight_scale,
            conv_weight,
            conv_bias,
            conv_state.clone(),
            query_start_loc,
            cache_indices,
            has_initial_state,
        )
        print(
            f"tokens={TOKENS} hidden={HIDDEN_SIZE} qkv={QKV_SIZE} "
            f"z={Z_SIZE} producer_chain=1 compile_args={compile_args}",
            flush=True,
        )
        reference, combined_ms = benchmark(
            "combined_producer_chain",
            combined_chain,
            combined_chain_inputs,
            args.warmups,
            args.iterations,
        )
        candidate, split_ms = benchmark(
            "split_producer_chain",
            split_chain,
            split_chain_inputs,
            args.warmups,
            args.iterations,
        )
        for name, expected, actual in zip(
            ("q", "k", "v", "z"),
            reference[:4],
            candidate[:4],
        ):
            difference = actual.float() - expected.float()
            relative_l2 = torch.linalg.vector_norm(difference)
            relative_l2 /= torch.linalg.vector_norm(
                expected.float()
            ).clamp_min(1e-12)
            print(
                f"producer_chain_{name}_max_abs="
                f"{float(difference.abs().max().cpu()):.9e} "
                f"relative_l2={float(relative_l2.cpu()):.9e}",
                flush=True,
            )
        saved = combined_ms - split_ms
        print(
            f"split_producer_chain_saved_ms={saved:.6f} "
            f"projected_48_layer_saved_ms={saved * 48:.6f}",
            flush=True,
        )
        if args.trace_dir is not None:
            capture_trace(
                "combined_producer_chain",
                combined_chain,
                combined_chain_inputs,
                args.trace_dir,
            )
            capture_trace(
                "split_producer_chain",
                split_chain,
                split_chain_inputs,
                args.trace_dir,
            )
        return
    compiled = {
        "combined_projection": torch.compile(
            combined_projection,
            **compile_args,
        ),
        "split_projection": torch.compile(
            split_projection,
            **compile_args,
        ),
        "split_projection_masked": torch.compile(
            split_projection_masked,
            **compile_args,
        ),
    }
    combined_inputs = x, token_mask, weight, weight_scale
    split_inputs = (
        x,
        token_mask,
        qkv_weight,
        qkv_weight_scale,
        z_weight,
        z_weight_scale,
    )
    print(
        f"tokens={TOKENS} hidden={HIDDEN_SIZE} qkv={QKV_SIZE} z={Z_SIZE} "
        f"compile_args={compile_args}",
        flush=True,
    )
    reference, combined_ms = benchmark(
        "combined_projection",
        compiled["combined_projection"],
        combined_inputs,
        args.warmups,
        args.iterations,
    )
    split, split_ms = benchmark(
        "split_projection",
        compiled["split_projection"],
        split_inputs,
        args.warmups,
        args.iterations,
    )
    split_masked, split_masked_ms = benchmark(
        "split_projection_masked",
        compiled["split_projection_masked"],
        split_inputs,
        args.warmups,
        args.iterations,
    )
    compare(reference, split, "split")
    compare(reference, split_masked, "split_masked")
    for name, latency in (
        ("split_projection", split_ms),
        ("split_projection_masked", split_masked_ms),
    ):
        saved = combined_ms - latency
        print(
            f"{name}_saved_ms={saved:.6f} "
            f"projected_48_layer_saved_ms={saved * 48:.6f}",
            flush=True,
        )

    if args.trace_dir is not None:
        capture_trace(
            "combined_projection",
            compiled["combined_projection"],
            combined_inputs,
            args.trace_dir,
        )
        capture_trace(
            "split_projection",
            compiled["split_projection"],
            split_inputs,
            args.trace_dir,
        )
        capture_trace(
            "split_projection_masked",
            compiled["split_projection_masked"],
            split_inputs,
            args.trace_dir,
        )


if __name__ == "__main__":
    main()
