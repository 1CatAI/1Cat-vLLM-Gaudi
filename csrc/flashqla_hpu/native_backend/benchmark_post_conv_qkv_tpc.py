import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


PACKED_WIDTH = 10_240
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
EPSILON = 1e-6


def reference_qk(packed: torch.Tensor):
    query, key, _ = torch.split(
        packed,
        [QK_WIDTH, QK_WIDTH, VALUE_WIDTH],
        dim=-1,
    )
    query = query.reshape(-1, QK_HEADS, HEAD_DIM).float()
    key = key.reshape(-1, QK_HEADS, HEAD_DIM).float()
    query = query / torch.sqrt(
        torch.sum(query * query, dim=-1, keepdim=True) + EPSILON
    )
    key = key / torch.sqrt(
        torch.sum(key * key, dim=-1, keepdim=True) + EPSILON
    )
    repeat = VALUE_HEADS // QK_HEADS
    return (
        query.repeat_interleave(repeat, dim=1),
        key.repeat_interleave(repeat, dim=1),
    )


def reference(packed: torch.Tensor):
    query, key = reference_qk(packed)
    value = packed[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    return query, key, value


def custom_qk(packed: torch.Tensor):
    return torch.ops.custom_op.qwen38_post_conv_qk_bf16_gaudi2(packed)


def custom_qk_cast_bf16(packed: torch.Tensor):
    query, key = custom_qk(packed)
    return query.to(torch.bfloat16), key.to(torch.bfloat16)


def custom_compact_qk(packed: torch.Tensor):
    return torch.ops.custom_op.qwen38_post_conv_qk_compact_bf16_gaudi2(
        packed,
    )


def custom_bf16_qk(packed: torch.Tensor):
    return torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
        packed,
    )


def custom_compact_qk_expanded(packed: torch.Tensor):
    query, key = custom_compact_qk(packed)
    repeat = VALUE_HEADS // QK_HEADS
    return (
        query.repeat_interleave(repeat, dim=1),
        key.repeat_interleave(repeat, dim=1),
    )


def custom(packed: torch.Tensor):
    query, key = custom_qk(packed)
    value = packed[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    return query, key, value


def make_input(tokens: int):
    torch.manual_seed(739251)
    return torch.randn(
        (tokens, PACKED_WIDTH),
        dtype=torch.bfloat16,
        device="hpu",
    )


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
    return tuple(tensor.clone() for tensor in output), median


def report_accuracy(reference_output, custom_output, prefix):
    for name, expected, actual in zip(
        ("q", "k", "v"),
        reference_output,
        custom_output,
    ):
        difference = actual.float() - expected.float()
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        print(
            f"{prefix}_{name}_max_abs={float(difference.abs().max().cpu()):.9e} "
            f"mean_abs={float(difference.abs().mean().cpu()):.9e} "
            f"relative_l2={float(relative_l2.cpu()):.9e} "
            f"equal_fraction={float((actual == expected).float().mean().cpu()):.9f}",
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
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16_384)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.ops.load_library(str(args.extension.resolve()))
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_reference = torch.compile(reference, **compile_args)
    compiled_custom = torch.compile(custom, **compile_args)
    compiled_reference_qk = torch.compile(reference_qk, **compile_args)
    compiled_custom_qk = torch.compile(custom_qk, **compile_args)
    compiled_custom_qk_cast = torch.compile(
        custom_qk_cast_bf16,
        **compile_args,
    )
    compiled_compact_qk = torch.compile(custom_compact_qk, **compile_args)
    compiled_bf16_qk = torch.compile(custom_bf16_qk, **compile_args)
    compiled_compact_qk_expanded = torch.compile(
        custom_compact_qk_expanded,
        **compile_args,
    )

    small = make_input(8)
    expected_small = compiled_reference(small)
    actual_small = compiled_custom(small)
    torch.hpu.synchronize()
    report_accuracy(expected_small, actual_small, "small")

    packed = make_input(args.tokens)
    inputs = (packed,)
    print(
        f"shape={tuple(packed.shape)} compile_args={compile_args}",
        flush=True,
    )
    expected_qk, reference_qk_ms = benchmark(
        "compiled_reference_qk_only",
        compiled_reference_qk,
        inputs,
        args.warmups,
        args.iterations,
    )
    actual_qk, custom_qk_ms = benchmark(
        "custom_tpc_qk_only",
        compiled_custom_qk,
        inputs,
        args.warmups,
        args.iterations,
    )
    report_accuracy(expected_qk, actual_qk, "qk_only")
    print(
        f"qk_only_speedup={reference_qk_ms / custom_qk_ms:.6f} "
        f"qk_only_saved_ms={reference_qk_ms - custom_qk_ms:.6f}",
        flush=True,
    )
    bf16_qk, bf16_qk_ms = benchmark(
        "custom_tpc_expanded_bf16_qk",
        compiled_bf16_qk,
        inputs,
        args.warmups,
        args.iterations,
    )
    expected_bf16 = tuple(tensor.to(torch.bfloat16) for tensor in actual_qk)
    report_accuracy(expected_bf16, bf16_qk, "bf16_vs_current_bf16")
    print(
        f"bf16_kernel_speedup={custom_qk_ms / bf16_qk_ms:.6f} "
        f"bf16_kernel_saved_ms={custom_qk_ms - bf16_qk_ms:.6f}",
        flush=True,
    )
    cast_qk, cast_qk_ms = benchmark(
        "custom_tpc_f32_qk_plus_bf16_cast",
        compiled_custom_qk_cast,
        inputs,
        args.warmups,
        args.iterations,
    )
    report_accuracy(cast_qk, bf16_qk, "bf16_direct_vs_cast")
    print(
        f"bf16_boundary_speedup={cast_qk_ms / bf16_qk_ms:.6f} "
        f"bf16_boundary_saved_ms={cast_qk_ms - bf16_qk_ms:.6f} "
        f"projected_48_layer_saved_ms="
        f"{(cast_qk_ms - bf16_qk_ms) * 48:.6f}",
        flush=True,
    )
    compact_qk, compact_qk_ms = benchmark(
        "custom_tpc_compact_bf16_qk",
        compiled_compact_qk,
        inputs,
        args.warmups,
        args.iterations,
    )
    compact_expanded, compact_expanded_ms = benchmark(
        "custom_tpc_compact_bf16_qk_plus_expand",
        compiled_compact_qk_expanded,
        inputs,
        args.warmups,
        args.iterations,
    )
    report_accuracy(
        expected_bf16,
        compact_expanded,
        "compact_vs_current_bf16",
    )
    print(
        f"compact_kernel_speedup={custom_qk_ms / compact_qk_ms:.6f} "
        f"compact_expanded_speedup="
        f"{custom_qk_ms / compact_expanded_ms:.6f} "
        f"compact_expanded_saved_ms="
        f"{custom_qk_ms - compact_expanded_ms:.6f}",
        flush=True,
    )

    expected, reference_ms = benchmark(
        "compiled_reference",
        compiled_reference,
        inputs,
        args.warmups,
        args.iterations,
    )
    actual, custom_ms = benchmark(
        "custom_tpc_qk_plus_dma_v",
        compiled_custom,
        inputs,
        args.warmups,
        args.iterations,
    )
    report_accuracy(expected, actual, "full")
    print(
        f"speedup={reference_ms / custom_ms:.6f} "
        f"saved_ms={reference_ms - custom_ms:.6f} "
        f"projected_48_layer_saved_ms={(reference_ms - custom_ms) * 48:.6f}",
        flush=True,
    )

    if args.trace_dir is not None:
        capture_trace("compiled_reference", compiled_reference, inputs, args.trace_dir)
        capture_trace("custom_tpc", compiled_custom, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
