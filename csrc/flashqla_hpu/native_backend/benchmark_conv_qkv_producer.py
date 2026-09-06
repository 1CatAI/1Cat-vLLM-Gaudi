import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_fn_token_major,
)
from vllm_gaudi.utils import HPUCompileConfig


PACKED_WIDTH = 10_240
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM


def causal_conv_silu(packed, initial_state, weight, bias):
    tokens = packed.shape[0]
    sequence = torch.cat((initial_state, packed), dim=0)
    output = torch.zeros_like(packed)
    for tap in range(4):
        output = output + sequence[tap : tap + tokens] * weight[tap]
    output = output + bias
    return torch.nn.functional.silu(output)


def fused_cpu_reference(inputs):
    packed, initial_state, weight, bias = (
        tensor.cpu() for tensor in inputs[:4]
    )
    post_conv = causal_conv_silu(packed, initial_state, weight, bias)
    query = post_conv[:, :QK_WIDTH].reshape(-1, QK_HEADS, HEAD_DIM)
    key = post_conv[:, QK_WIDTH : 2 * QK_WIDTH].reshape(
        -1, QK_HEADS, HEAD_DIM
    )
    query = (
        query.float()
        * torch.rsqrt(query.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    key = (
        key.float()
        * torch.rsqrt(key.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1, VALUE_HEADS, HEAD_DIM
    )
    return (
        query.repeat_interleave(3, dim=1),
        key.repeat_interleave(3, dim=1),
        value,
    )


def current(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    del initial_state
    post_conv = hpu_causal_conv1d_fn(
        x=packed.transpose(0, 1),
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
        activation="silu",
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        block_size_to_align=0,
        is_prompt=True,
    ).transpose(0, 1)
    query, key = (
        torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
            post_conv
        )
    )
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1, VALUE_HEADS, HEAD_DIM
    ).contiguous()
    return query, key, value, conv_state


def token_major_torch(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    del initial_state
    post_conv = hpu_causal_conv1d_fn_token_major(
        x=packed,
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
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
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    return query, key, value, conv_state


def token_major_explicit_sigmoid(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    del initial_state
    post_conv = hpu_causal_conv1d_fn_token_major(
        x=packed,
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
        activation=None,
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        block_size_to_align=0,
        is_prompt=True,
    )
    post_conv = post_conv * torch.sigmoid(post_conv)
    query, key = (
        torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
            post_conv
        )
    )
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    return query, key, value, conv_state


def token_major_with_padding_mask(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
    token_mask,
    gate,
    beta,
):
    del initial_state
    packed = packed * token_mask
    post_conv = hpu_causal_conv1d_fn_token_major(
        x=packed,
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
        activation="silu",
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        block_size_to_align=0,
        is_prompt=True,
    )
    post_conv = post_conv * token_mask
    query, key = (
        torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
            post_conv
        )
    )
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    token_mask_h = token_mask.view(1, -1, 1).to(gate.dtype)
    return (
        query,
        key,
        value,
        gate * token_mask_h,
        beta * token_mask_h,
        conv_state,
    )


def token_major_without_padding_mask(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
    token_mask,
    gate,
    beta,
):
    del initial_state, token_mask
    post_conv = hpu_causal_conv1d_fn_token_major(
        x=packed,
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
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
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    return query, key, value, gate, beta, conv_state


def token_major_conv_only(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    del initial_state
    post_conv = hpu_causal_conv1d_fn_token_major(
        x=packed,
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
        activation="silu",
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        block_size_to_align=0,
        is_prompt=True,
    )
    return post_conv, conv_state


def native_qkv_from_post_conv(post_conv, conv_state):
    query, key = (
        torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
            post_conv
        )
    )
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1,
        VALUE_HEADS,
        HEAD_DIM,
    ).contiguous()
    return query, key, value, conv_state


def fused(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    del conv_state, query_start_loc, cache_indices, has_initial_state
    return torch.ops.custom_op.qwen38_conv_qkv_prep_bf16_gaudi2(
        packed,
        initial_state,
        weight,
        bias,
    )


def official_hpu(
    packed,
    initial_state,
    weight,
    bias,
    conv_state,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    del initial_state
    post_conv, conv_state_out = torch.ops.hpu.causal_conv1d_fwd(
        packed,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
        activation=True,
        pad_slot_id=-1,
    )
    query, key = (
        torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2(
            post_conv
        )
    )
    value = post_conv[:, 2 * QK_WIDTH :].reshape(
        -1, VALUE_HEADS, HEAD_DIM
    ).contiguous()
    return query, key, value, conv_state_out


def make_inputs(tokens, random_state):
    torch.manual_seed(739251)
    packed = torch.randn(
        (tokens, PACKED_WIDTH), dtype=torch.bfloat16, device="hpu"
    ) * 0.08
    weight = torch.randn(
        (4, PACKED_WIDTH), dtype=torch.bfloat16, device="hpu"
    ) * 0.08
    bias = torch.randn(
        (PACKED_WIDTH,), dtype=torch.bfloat16, device="hpu"
    ) * 0.01
    if random_state:
        state = torch.randn(
            (3, PACKED_WIDTH), dtype=torch.bfloat16, device="hpu"
        ) * 0.08
    else:
        state = torch.zeros(
            (3, PACKED_WIDTH), dtype=torch.bfloat16, device="hpu"
        )
    spare_state = torch.zeros_like(state)
    conv_state = torch.stack((state, spare_state), dim=0)
    query_start_loc = torch.tensor(
        [0, tokens], dtype=torch.int32, device="hpu"
    )
    cache_indices = torch.tensor([0], dtype=torch.int64, device="hpu")
    has_initial_state = torch.tensor(
        [random_state], dtype=torch.bool, device="hpu"
    )
    return (
        packed,
        state,
        weight,
        bias,
        conv_state,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


def make_mask_ab_inputs(tokens):
    base_inputs = make_inputs(tokens, random_state=False)
    token_mask = torch.ones(
        (tokens, 1),
        dtype=torch.bfloat16,
        device="hpu",
    )
    gate = torch.randn(
        (1, tokens, VALUE_HEADS),
        dtype=torch.float32,
        device="hpu",
    )
    beta = torch.rand(
        (1, tokens, VALUE_HEADS),
        dtype=torch.bfloat16,
        device="hpu",
    )
    return (*base_inputs, token_mask, gate, beta)


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


def report_accuracy(expected, actual, prefix):
    for name, reference, candidate in zip(("q", "k", "v"), expected, actual):
        difference = candidate.float() - reference.float()
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
        print(
            f"{prefix}_{name}_max_abs={float(difference.abs().max().cpu()):.9e} "
            f"mean_abs={float(difference.abs().mean().cpu()):.9e} "
            f"relative_l2={float(relative_l2.cpu()):.9e} "
            f"equal_fraction="
            f"{float((candidate == reference).float().mean().cpu()):.9f}",
            flush=True,
        )


def report_cpu_reference(inputs, current_output, fused_output):
    packed, initial_state, weight, bias = (
        tensor.cpu() for tensor in inputs[:4]
    )
    tokens = packed.shape[0]
    sequence = torch.cat((initial_state, packed), dim=0)
    sequential = torch.zeros_like(packed)
    for tap in range(4):
        sequential = sequential + (
            sequence[tap : tap + tokens] * weight[tap]
        )
    sequential = torch.nn.functional.silu(sequential + bias)
    value = sequential[:, 2 * QK_WIDTH :]
    current_v = current_output[2].cpu().reshape(tokens, VALUE_WIDTH)
    fused_v = fused_output[2].cpu().reshape(tokens, VALUE_WIDTH)
    for name, candidate in (("current", current_v), ("fused", fused_v)):
        difference = candidate.float() - value.float()
        per_token = difference.abs().amax(dim=1)
        mismatched = torch.nonzero(per_token != 0).flatten()
        first = int(mismatched[0]) if mismatched.numel() else -1
        last = int(mismatched[-1]) if mismatched.numel() else -1
        print(
            f"cpu_reference_vs_{name}_v_equal="
            f"{float((candidate == value).float().mean()):.9f} "
            f"max_abs={float(difference.abs().max()):.9e} "
            f"mismatched_tokens={mismatched.numel()} first={first} last={last}",
            flush=True,
        )


def report_state(reference, candidate, candidate_input, prefix):
    reference_state = reference[3].float()
    candidate_state = candidate[3].float()
    difference = candidate_state - reference_state
    input_difference = candidate_input.float() - reference_state
    print(
        f"{prefix}_state_max_abs={float(difference.abs().max().cpu()):.9e} "
        f"state_equal_fraction="
        f"{float((candidate_state == reference_state).float().mean().cpu()):.9f} "
        f"input_state_max_abs="
        f"{float(input_difference.abs().max().cpu()):.9e}",
        flush=True,
    )


def capture_trace(name, function, inputs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    handler = torch.profiler.tensorboard_trace_handler(
        str(output_dir), worker_name=name, use_gzip=True
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
    parser.add_argument("--token-major-only", action="store_true")
    parser.add_argument("--fused-only", action="store_true")
    parser.add_argument("--validate-fused", action="store_true")
    parser.add_argument("--split-recipe", action="store_true")
    parser.add_argument("--mask-ab", action="store_true")
    parser.add_argument("--activation-ab", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.ops.load_library(str(args.extension.resolve()))
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    if args.activation_ab:
        current_activation = torch.compile(
            token_major_torch,
            **compile_args,
        )
        explicit_activation = torch.compile(
            token_major_explicit_sigmoid,
            **compile_args,
        )
        current_inputs = make_inputs(args.tokens, random_state=False)
        explicit_inputs = make_inputs(args.tokens, random_state=False)
        print(
            f"tokens={args.tokens} packed_width={PACKED_WIDTH} "
            f"compile_args={compile_args}",
            flush=True,
        )
        expected, current_ms = benchmark(
            "token_major_native_silu",
            current_activation,
            current_inputs,
            args.warmups,
            args.iterations,
        )
        actual, explicit_ms = benchmark(
            "token_major_explicit_sigmoid",
            explicit_activation,
            explicit_inputs,
            args.warmups,
            args.iterations,
        )
        report_accuracy(expected, actual, "explicit_sigmoid")
        report_state(
            expected,
            actual,
            explicit_inputs[4],
            "explicit_sigmoid",
        )
        print(
            f"explicit_sigmoid_saved_ms={current_ms - explicit_ms:.6f} "
            f"projected_48_layer_saved_ms="
            f"{(current_ms - explicit_ms) * 48:.6f}",
            flush=True,
        )
        if args.trace_dir is not None:
            capture_trace(
                "token_major_native_silu",
                current_activation,
                current_inputs,
                args.trace_dir,
            )
            capture_trace(
                "token_major_explicit_sigmoid",
                explicit_activation,
                explicit_inputs,
                args.trace_dir,
            )
        return
    if args.mask_ab:
        masked = torch.compile(
            token_major_with_padding_mask,
            **compile_args,
        )
        unmasked = torch.compile(
            token_major_without_padding_mask,
            **compile_args,
        )
        masked_inputs = make_mask_ab_inputs(args.tokens)
        unmasked_inputs = make_mask_ab_inputs(args.tokens)
        print(
            f"tokens={args.tokens} packed_width={PACKED_WIDTH} "
            f"compile_args={compile_args}",
            flush=True,
        )
        expected, masked_ms = benchmark(
            "token_major_with_padding_mask",
            masked,
            masked_inputs,
            args.warmups,
            args.iterations,
        )
        actual, unmasked_ms = benchmark(
            "token_major_without_padding_mask",
            unmasked,
            unmasked_inputs,
            args.warmups,
            args.iterations,
        )
        for name, reference, candidate in zip(
            ("q", "k", "v", "g", "beta"),
            expected[:5],
            actual[:5],
        ):
            difference = candidate.float() - reference.float()
            relative_l2 = torch.linalg.vector_norm(difference)
            relative_l2 /= torch.linalg.vector_norm(
                reference.float()
            ).clamp_min(1e-12)
            print(
                f"mask_ab_{name}_max_abs="
                f"{float(difference.abs().max().cpu()):.9e} "
                f"relative_l2={float(relative_l2.cpu()):.9e}",
                flush=True,
            )
        print(
            f"mask_removal_saved_ms={masked_ms - unmasked_ms:.6f} "
            f"mask_removal_projected_48_layer_saved_ms="
            f"{(masked_ms - unmasked_ms) * 48:.6f}",
            flush=True,
        )
        if args.trace_dir is not None:
            capture_trace(
                "token_major_with_padding_mask",
                masked,
                masked_inputs,
                args.trace_dir,
            )
            capture_trace(
                "token_major_without_padding_mask",
                unmasked,
                unmasked_inputs,
                args.trace_dir,
            )
        return
    if args.fused_only:
        compiled_fused = torch.compile(fused, **compile_args)
        if args.validate_fused:
            validation_inputs = make_inputs(8, random_state=True)
            actual = compiled_fused(*validation_inputs)
            torch.hpu.synchronize()
            expected = fused_cpu_reference(validation_inputs)
            report_accuracy(
                expected,
                tuple(tensor.cpu() for tensor in actual),
                "fused_cpu_reference",
            )
        inputs = make_inputs(args.tokens, random_state=False)
        print(
            f"tokens={args.tokens} packed_width={PACKED_WIDTH} "
            f"compile_args={compile_args}",
            flush=True,
        )
        benchmark(
            "fused_conv_qkv_producer",
            compiled_fused,
            inputs,
            args.warmups,
            args.iterations,
        )
        if args.trace_dir is not None:
            capture_trace(
                "fused_conv_qkv_producer",
                compiled_fused,
                inputs,
                args.trace_dir,
            )
        return
    compiled_current = torch.compile(current, **compile_args)
    compiled_token_major = torch.compile(token_major_torch, **compile_args)
    if args.token_major_only:
        candidate_name = "token_major_direct_plus_native_qk"
        candidate = compiled_token_major
        if args.split_recipe:
            compiled_token_major_conv = torch.compile(
                token_major_conv_only,
                **compile_args,
            )
            compiled_native_qkv = torch.compile(
                native_qkv_from_post_conv,
                **compile_args,
            )

            def token_major_split_recipe(*inputs):
                return compiled_native_qkv(
                    *compiled_token_major_conv(*inputs)
                )

            candidate_name = "token_major_split_recipe_plus_native_qk"
            candidate = token_major_split_recipe

        small_inputs = make_inputs(8, random_state=True)
        small_candidate_inputs = make_inputs(8, random_state=True)
        expected_small = compiled_current(*small_inputs)
        actual_small = candidate(*small_candidate_inputs)
        torch.hpu.synchronize()
        report_accuracy(expected_small, actual_small, "small_token_major")
        report_state(
            expected_small,
            actual_small,
            small_candidate_inputs[4],
            "small_token_major",
        )

        inputs = make_inputs(args.tokens, random_state=False)
        candidate_inputs = make_inputs(args.tokens, random_state=False)
        print(
            f"tokens={args.tokens} packed_width={PACKED_WIDTH} "
            f"compile_args={compile_args}",
            flush=True,
        )
        expected, current_ms = benchmark(
            "current_conv_plus_native_qk",
            compiled_current,
            inputs,
            args.warmups,
            args.iterations,
        )
        actual, token_major_ms = benchmark(
            candidate_name,
            candidate,
            candidate_inputs,
            args.warmups,
            args.iterations,
        )
        report_accuracy(expected, actual, "token_major_zero_state")
        report_state(
            expected,
            actual,
            candidate_inputs[4],
            "token_major_zero_state",
        )
        print(
            f"token_major_speedup={current_ms / token_major_ms:.6f} "
            f"token_major_saved_ms={current_ms - token_major_ms:.6f} "
            f"token_major_projected_48_layer_saved_ms="
            f"{(current_ms - token_major_ms) * 48:.6f}",
            flush=True,
        )
        if args.trace_dir is not None:
            capture_trace(
                "current_conv_plus_native_qk",
                compiled_current,
                inputs,
                args.trace_dir,
            )
            capture_trace(
                candidate_name,
                candidate,
                candidate_inputs,
                args.trace_dir,
            )
        return

    compiled_official = torch.compile(official_hpu, **compile_args)
    compiled_fused = torch.compile(fused, **compile_args)

    small_inputs = make_inputs(8, random_state=True)
    small_official_inputs = make_inputs(8, random_state=True)
    small_fused_inputs = make_inputs(8, random_state=True)
    expected_small = compiled_current(*small_inputs)
    official_small = compiled_official(*small_official_inputs)
    actual_small = compiled_fused(*small_fused_inputs)
    torch.hpu.synchronize()
    report_accuracy(expected_small, official_small, "small_official")
    report_state(
        expected_small,
        official_small,
        small_official_inputs[4],
        "small_official",
    )
    report_accuracy(expected_small, actual_small, "small_random_state")
    report_cpu_reference(small_fused_inputs, expected_small, actual_small)

    inputs = make_inputs(args.tokens, random_state=False)
    token_major_inputs = make_inputs(args.tokens, random_state=False)
    official_inputs = make_inputs(args.tokens, random_state=False)
    fused_inputs = make_inputs(args.tokens, random_state=False)
    print(
        f"tokens={args.tokens} packed_width={PACKED_WIDTH} "
        f"compile_args={compile_args}",
        flush=True,
    )
    expected, current_ms = benchmark(
        "current_conv_plus_native_qk",
        compiled_current,
        inputs,
        args.warmups,
        args.iterations,
    )
    token_major, token_major_ms = benchmark(
        "token_major_torch_conv_plus_native_qk",
        compiled_token_major,
        token_major_inputs,
        args.warmups,
        args.iterations,
    )
    official, official_ms = benchmark(
        "official_hpu_conv_plus_native_qk",
        compiled_official,
        official_inputs,
        args.warmups,
        args.iterations,
    )
    actual, fused_ms = benchmark(
        "fused_conv_qkv_producer",
        compiled_fused,
        fused_inputs,
        args.warmups,
        args.iterations,
    )
    report_accuracy(expected, official, "official_zero_state")
    report_accuracy(expected, token_major, "token_major_zero_state")
    report_state(
        expected,
        token_major,
        token_major_inputs[4],
        "token_major_zero_state",
    )
    report_state(
        expected,
        official,
        official_inputs[4],
        "official_zero_state",
    )
    report_accuracy(expected, actual, "full_zero_state")
    report_cpu_reference(fused_inputs, expected, actual)
    print(
        f"token_major_speedup={current_ms / token_major_ms:.6f} "
        f"token_major_saved_ms={current_ms - token_major_ms:.6f} "
        f"token_major_projected_48_layer_saved_ms="
        f"{(current_ms - token_major_ms) * 48:.6f} "
        f"official_speedup={current_ms / official_ms:.6f} "
        f"official_saved_ms={current_ms - official_ms:.6f} "
        f"speedup={current_ms / fused_ms:.6f} "
        f"saved_ms={current_ms - fused_ms:.6f} "
        f"projected_48_layer_saved_ms={(current_ms - fused_ms) * 48:.6f}",
        flush=True,
    )

    if args.trace_dir is not None:
        capture_trace("current_conv_plus_native_qk", compiled_current, inputs, args.trace_dir)
        capture_trace(
            "token_major_torch_conv_plus_native_qk",
            compiled_token_major,
            token_major_inputs,
            args.trace_dir,
        )
        capture_trace(
            "official_hpu_conv_plus_native_qk",
            compiled_official,
            official_inputs,
            args.trace_dir,
        )
        capture_trace(
            "fused_conv_qkv_producer",
            compiled_fused,
            fused_inputs,
            args.trace_dir,
        )


if __name__ == "__main__":
    main()
