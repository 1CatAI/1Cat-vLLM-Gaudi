import argparse
import math
import os
import statistics
import time
from pathlib import Path

import torch
import habana_frameworks.torch.utils.experimental as htexp
from habana_frameworks.torch.hpex.kernels import FusedSDPA, fp8_fused_sdpa

from vllm_gaudi.utils import HPUCompileConfig


NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
ATTENTION_SCALE = HEAD_DIM**-0.5
FP8_DTYPE = torch.float8_e4m3fn
FP8_SOFTMAX_MODE = "fast"
if htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2:
    FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max
else:
    FP8_MAX = torch.finfo(FP8_DTYPE).max


def bf16_masked(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    del valid_seq_len, softmax_scale, descale_softmax
    return FusedSDPA.apply(
        q,
        k,
        v,
        bias,
        0.0,
        False,
        ATTENTION_SCALE,
        "fast",
        True,
        None,
        "right",
        False,
        False,
        (-1, -1),
        None,
    )


def bf16_causal(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    del bias, softmax_scale, descale_softmax
    return FusedSDPA.apply(
        q,
        k,
        v,
        None,
        0.0,
        True,
        ATTENTION_SCALE,
        "fast",
        True,
        valid_seq_len,
        "right",
        False,
        False,
        (-1, -1),
        None,
    )


def _bf16_causal_head_split(
    q,
    k,
    v,
    valid_seq_len,
    parts,
):
    q_heads = q.shape[1] // parts
    kv_heads = k.shape[1] // parts
    outputs = []
    for index in range(parts):
        outputs.append(
            FusedSDPA.apply(
                q[:, index * q_heads:(index + 1) * q_heads],
                k[:, index * kv_heads:(index + 1) * kv_heads],
                v[:, index * kv_heads:(index + 1) * kv_heads],
                None,
                0.0,
                True,
                ATTENTION_SCALE,
                "fast",
                True,
                valid_seq_len,
                "right",
                False,
                False,
                (-1, -1),
                None,
            )
        )
    return torch.cat(outputs, dim=1)


def bf16_causal_head_split_2(
    q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax
):
    del bias, softmax_scale, descale_softmax
    return _bf16_causal_head_split(q, k, v, valid_seq_len, 2)


def bf16_causal_head_split_4(
    q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax
):
    del bias, softmax_scale, descale_softmax
    return _bf16_causal_head_split(q, k, v, valid_seq_len, 4)


def bf16_causal_expanded_kv(
    q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax
):
    del bias, softmax_scale, descale_softmax
    repeats = q.shape[1] // k.shape[1]
    return FusedSDPA.apply(
        q,
        k.repeat_interleave(repeats, dim=1),
        v.repeat_interleave(repeats, dim=1),
        None,
        0.0,
        True,
        ATTENTION_SCALE,
        "fast",
        True,
        valid_seq_len,
        "right",
        False,
        False,
        (-1, -1),
        None,
    )


def per_tensor_quant(x):
    descale = (x.abs().amax() + 1e-8) / FP8_MAX
    quantized = torch.ops.hpu.cast_to_fp8_v2(
        x,
        descale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return quantized, descale.float()


def _roundtrip_fp8(x):
    quantized, descale = per_tensor_quant(x)
    return torch.ops.hpu.cast_from_fp8(quantized, descale, torch.bfloat16)


def bf16_qk_roundtrip(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    return bf16_masked(
        _roundtrip_fp8(q),
        _roundtrip_fp8(k),
        v,
        bias,
        valid_seq_len,
        softmax_scale,
        descale_softmax,
    )


def bf16_v_roundtrip(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    return bf16_masked(
        q,
        k,
        _roundtrip_fp8(v),
        bias,
        valid_seq_len,
        softmax_scale,
        descale_softmax,
    )


def bf16_qkv_roundtrip(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    return bf16_masked(
        _roundtrip_fp8(q),
        _roundtrip_fp8(k),
        _roundtrip_fp8(v),
        bias,
        valid_seq_len,
        softmax_scale,
        descale_softmax,
    )


def _fp8_sdpa(
    q,
    k,
    v,
    bias,
    valid_seq_len,
    descale_q,
    descale_k,
    descale_v,
    softmax_scale,
    descale_softmax,
    is_causal,
):
    results = fp8_fused_sdpa(
        q,
        k,
        v,
        attn_mask=None if is_causal else bias,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=ATTENTION_SCALE,
        softmax_mode=FP8_SOFTMAX_MODE,
        d_scale_q=descale_q,
        d_scale_k=descale_k,
        d_scale_v=descale_v,
        q_scale_s=softmax_scale,
        q_scale_o=None,
        d_scale_s=descale_softmax,
        is_amax_s=False,
        is_amax_o=False,
        valid_seq_len=valid_seq_len if is_causal else None,
        seq_padding_type="right",
        recompute=True,
        requires_grad=False,
    )
    return results[0]


def fp8_dynamic_masked(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    q_fp8, descale_q = per_tensor_quant(q)
    k_fp8, descale_k = per_tensor_quant(k)
    v_fp8, descale_v = per_tensor_quant(v)
    return _fp8_sdpa(
        q_fp8,
        k_fp8,
        v_fp8,
        bias,
        valid_seq_len,
        descale_q,
        descale_k,
        descale_v,
        softmax_scale,
        descale_softmax,
        False,
    )


def fp8_dynamic_causal(q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax):
    q_fp8, descale_q = per_tensor_quant(q)
    k_fp8, descale_k = per_tensor_quant(k)
    v_fp8, descale_v = per_tensor_quant(v)
    return _fp8_sdpa(
        q_fp8,
        k_fp8,
        v_fp8,
        bias,
        valid_seq_len,
        descale_q,
        descale_k,
        descale_v,
        softmax_scale,
        descale_softmax,
        True,
    )


def fp8_prequant_masked(
    q,
    k,
    v,
    bias,
    valid_seq_len,
    descale_q,
    descale_k,
    descale_v,
    softmax_scale,
    descale_softmax,
):
    return _fp8_sdpa(
        q,
        k,
        v,
        bias,
        valid_seq_len,
        descale_q,
        descale_k,
        descale_v,
        softmax_scale,
        descale_softmax,
        False,
    )


def fp8_prequant_causal(
    q,
    k,
    v,
    bias,
    valid_seq_len,
    descale_q,
    descale_k,
    descale_v,
    softmax_scale,
    descale_softmax,
):
    return _fp8_sdpa(
        q,
        k,
        v,
        bias,
        valid_seq_len,
        descale_q,
        descale_k,
        descale_v,
        softmax_scale,
        descale_softmax,
        True,
    )


def benchmark(name, function, args, warmups, samples):
    output = function(*args)
    torch.hpu.synchronize()
    for _ in range(warmups):
        output = function(*args)
    torch.hpu.synchronize()

    times_ms = []
    for _ in range(samples):
        start = time.perf_counter()
        output = function(*args)
        torch.hpu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    saved = output.detach().clone()
    torch.hpu.synchronize()
    print(
        f"{name} dtype={output.dtype} median_ms={statistics.median(times_ms):.6f} "
        f"min_ms={min(times_ms):.6f} samples_ms={times_ms}",
        flush=True,
    )
    return saved


def report_quality(name, reference, candidate):
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    delta = candidate_f32 - reference_f32
    relative_l2 = delta.norm() / reference_f32.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        reference_f32.flatten(),
        candidate_f32.flatten(),
        dim=0,
    )
    finite = torch.isfinite(candidate_f32).float().mean()
    print(
        f"quality {name} rel_l2={relative_l2.cpu().item():.9f} "
        f"cosine={cosine.cpu().item():.9f} "
        f"mean_abs={delta.abs().mean().cpu().item():.9f} "
        f"max_abs={delta.abs().max().cpu().item():.9f} "
        f"finite_fraction={finite.cpu().item():.9f}",
        flush=True,
    )


def capture_trace(trace_dir, functions, args_by_name):
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    handler = torch.profiler.tensorboard_trace_handler(str(trace_path))
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.HPU,
        ],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        for name, function in functions.items():
            with torch.profiler.record_function(name):
                function(*args_by_name[name])
                torch.hpu.synchronize()
            profiler.step()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=16_384)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--softmax-scale", type=float, default=120.0)
    parser.add_argument("--fp8-softmax-mode", choices=("fast", "none", "fp32"), default="fast")
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument("--quality-softmax-scales", default="")
    parser.add_argument("--trace-dir")
    parser.add_argument("--only", default="")
    parser.add_argument("--output-path")
    return parser.parse_args()


def main():
    global FP8_SOFTMAX_MODE
    args = parse_args()
    FP8_SOFTMAX_MODE = args.fp8_softmax_mode
    torch.manual_seed(123)
    seq_len = args.seq_len

    q = torch.randn(
        (1, NUM_Q_HEADS, seq_len, HEAD_DIM),
        dtype=torch.bfloat16,
        device="hpu",
    )
    k = torch.randn(
        (1, NUM_KV_HEADS, seq_len, HEAD_DIM),
        dtype=torch.bfloat16,
        device="hpu",
    )
    v = torch.randn(
        (1, NUM_KV_HEADS, seq_len, HEAD_DIM),
        dtype=torch.bfloat16,
        device="hpu",
    )
    bias = torch.full(
        (1, 1, seq_len, seq_len),
        float("-inf"),
        dtype=torch.bfloat16,
        device="hpu",
    ).triu(diagonal=1)
    valid_seq_len = torch.full((1,), seq_len, dtype=torch.int32, device="hpu")
    softmax_scale = torch.tensor(args.softmax_scale, dtype=torch.float32, device="hpu")
    descale_softmax = softmax_scale.reciprocal()
    q_fp8, descale_q = per_tensor_quant(q)
    k_fp8, descale_k = per_tensor_quant(k)
    v_fp8, descale_v = per_tensor_quant(v)
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    print(
        f"seq_len={seq_len} q={tuple(q.shape)} kv={tuple(k.shape)} "
        f"fp8_max={FP8_MAX} softmax_scale={args.softmax_scale} "
        f"fp8_softmax_mode={FP8_SOFTMAX_MODE} "
        f"compile_args={compile_args}",
        flush=True,
    )
    sdpa_env = {
        name: os.environ.get(name)
        for name in (
            "PT_HPU_SDPA_QKV_SLICE_MODE_FWD",
            "PT_HPU_SDPA_BR_FACTOR",
            "PT_HPU_SDPA_BC_FACTOR",
            "PT_HPU_SDPA_SOFTMAX_SLICE_SIZE",
        )
    }
    print(f"sdpa_env={sdpa_env}", flush=True)
    functions = {
        "bf16_masked": bf16_masked,
        "bf16_causal": bf16_causal,
        "bf16_causal_head_split_2": bf16_causal_head_split_2,
        "bf16_causal_head_split_4": bf16_causal_head_split_4,
        "bf16_causal_expanded_kv": bf16_causal_expanded_kv,
        "fp8_dynamic_masked": fp8_dynamic_masked,
        "fp8_dynamic_causal": fp8_dynamic_causal,
        "fp8_prequant_masked": fp8_prequant_masked,
        "fp8_prequant_causal": fp8_prequant_causal,
    }
    if args.attribution:
        functions.update({
            "bf16_qk_roundtrip": bf16_qk_roundtrip,
            "bf16_v_roundtrip": bf16_v_roundtrip,
            "bf16_qkv_roundtrip": bf16_qkv_roundtrip,
        })
    if args.only:
        requested = args.only.split(",")
        functions = {
            name: functions[name]
            for name in dict.fromkeys(["bf16_causal", *requested])
        }
    compiled = {
        name: torch.compile(function, **compile_args)
        for name, function in functions.items()
    }
    bf16_args = (q, k, v, bias, valid_seq_len, softmax_scale, descale_softmax)
    prequant_args = (
        q_fp8,
        k_fp8,
        v_fp8,
        bias,
        valid_seq_len,
        descale_q,
        descale_k,
        descale_v,
        softmax_scale,
        descale_softmax,
    )
    args_by_name = {
        "bf16_masked": bf16_args,
        "bf16_causal": bf16_args,
        "bf16_causal_head_split_2": bf16_args,
        "bf16_causal_head_split_4": bf16_args,
        "bf16_causal_expanded_kv": bf16_args,
        "fp8_dynamic_masked": bf16_args,
        "fp8_dynamic_causal": bf16_args,
        "fp8_prequant_masked": prequant_args,
        "fp8_prequant_causal": prequant_args,
    }
    if args.attribution:
        args_by_name.update({
            "bf16_qk_roundtrip": bf16_args,
            "bf16_v_roundtrip": bf16_args,
            "bf16_qkv_roundtrip": bf16_args,
        })

    outputs = {}
    for name, function in compiled.items():
        outputs[name] = benchmark(
            name,
            function,
            args_by_name[name],
            args.warmups,
            args.samples,
        )

    reference = outputs.get("bf16_masked", outputs["bf16_causal"])
    for name, output in outputs.items():
        report_quality(name, reference, output)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs["bf16_causal"].cpu(), output_path)
        print(f"saved bf16_causal output to {output_path}", flush=True)

    if args.quality_softmax_scales:
        for value in args.quality_softmax_scales.split(","):
            candidate_scale = torch.tensor(float(value), dtype=torch.float32, device="hpu")
            candidate_args = prequant_args[:-2] + (candidate_scale, candidate_scale.reciprocal())
            candidate = compiled["fp8_prequant_masked"](*candidate_args).detach().clone()
            torch.hpu.synchronize()
            report_quality(f"fp8_prequant_masked_s{value}", reference, candidate)

    if args.trace_dir:
        capture_trace(args.trace_dir, compiled, args_by_name)


if __name__ == "__main__":
    main()
