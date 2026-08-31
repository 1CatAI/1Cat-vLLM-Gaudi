import argparse
from pathlib import Path
import statistics
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
Q_DIM = QK_HEADS * HEAD_DIM
V_DIM = VALUE_HEADS * HEAD_DIM
EPSILON = 1e-6


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + EPSILON)


def current_gating(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
):
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    softplus_x = torch.where(
        x <= 20.0,
        torch.log1p(torch.exp(x)),
        x,
    )
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    beta = torch.sigmoid(b.to(torch.float32)).to(b.dtype)
    return g.unsqueeze(0), beta.unsqueeze(0)


def current_rearrange(mixed_qkv: torch.Tensor):
    query, key, value = torch.split(
        mixed_qkv,
        [Q_DIM, Q_DIM, V_DIM],
        dim=-1,
    )
    fused = torch.cat(
        [query.reshape(-1), key.reshape(-1), value.reshape(-1)],
        dim=0,
    )
    q_size = TOKENS * Q_DIM
    k_size = TOKENS * Q_DIM
    query = fused[:q_size].view(1, TOKENS, QK_HEADS, HEAD_DIM)
    key = fused[q_size:q_size + k_size].view(
        1,
        TOKENS,
        QK_HEADS,
        HEAD_DIM,
    )
    value = fused[q_size + k_size:].view(
        1,
        TOKENS,
        VALUE_HEADS,
        HEAD_DIM,
    )
    return query, key, value


def current_normalize_expand(q: torch.Tensor, k: torch.Tensor):
    repeat = VALUE_HEADS // QK_HEADS
    q = _l2norm(q).to(torch.bfloat16)
    k = _l2norm(k).to(torch.bfloat16)
    return (
        q.repeat_interleave(repeat, dim=2),
        k.repeat_interleave(repeat, dim=2),
    )


def merged_post_conv_prep(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
):
    query, key, value = torch.split(
        mixed_qkv,
        [Q_DIM, Q_DIM, V_DIM],
        dim=-1,
    )
    query = query.view(TOKENS, QK_HEADS, HEAD_DIM)
    key = key.view(TOKENS, QK_HEADS, HEAD_DIM)
    value = value.view(TOKENS, VALUE_HEADS, HEAD_DIM)

    repeat = VALUE_HEADS // QK_HEADS
    query = _l2norm(query).to(torch.bfloat16).repeat_interleave(
        repeat,
        dim=1,
    )
    key = _l2norm(key).to(torch.bfloat16).repeat_interleave(
        repeat,
        dim=1,
    )

    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    softplus_x = torch.where(
        x <= 20.0,
        torch.log1p(torch.exp(x)),
        x,
    )
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    beta = torch.sigmoid(b.to(torch.float32)).to(b.dtype)
    return (
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.contiguous().unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
    )


def make_current_runner(compiled_gating, compiled_rearrange, compiled_norm):
    def run(mixed_qkv, a, b, A_log, dt_bias):
        g, beta = compiled_gating(a, b, A_log, dt_bias)
        q, k, v = compiled_rearrange(mixed_qkv)
        q, k = compiled_norm(q, k)
        return q, k, v, g, beta

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

    median = statistics.median(samples)
    print(
        f"{name} median_ms={median:.6f} min_ms={min(samples):.6f} "
        f"samples_ms={samples}",
        flush=True,
    )
    return tuple(tensor.clone() for tensor in output), median


def compare(reference, candidate):
    for name, expected, actual in zip(
        ("q", "k", "v", "g", "beta"),
        reference,
        candidate,
    ):
        difference = actual.float() - expected.float()
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        print(
            f"quality_{name} max_abs={float(difference.abs().max().cpu()):.9e} "
            f"equal_fraction={float((actual == expected).float().mean().cpu()):.9f} "
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
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--gating-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    mixed_qkv = torch.randn(
        (TOKENS, Q_DIM * 2 + V_DIM),
        dtype=torch.bfloat16,
        device="hpu",
    )
    a = torch.randn(
        (TOKENS, VALUE_HEADS),
        dtype=torch.bfloat16,
        device="hpu",
    )
    b = torch.randn_like(a)
    A_log = torch.randn(
        VALUE_HEADS,
        dtype=torch.float32,
        device="hpu",
    )
    dt_bias = torch.randn_like(A_log)
    inputs = mixed_qkv, a, b, A_log, dt_bias
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_gating = torch.compile(current_gating, **compile_args)
    if args.gating_only:
        gating_inputs = a, b, A_log, dt_bias
        benchmark(
            "current_gating",
            compiled_gating,
            gating_inputs,
            args.warmups,
            args.iterations,
        )
        if args.trace_dir is not None:
            capture_trace(
                "current_gating",
                compiled_gating,
                gating_inputs,
                args.trace_dir,
            )
        return
    compiled_rearrange = torch.compile(current_rearrange, **compile_args)
    compiled_norm = torch.compile(current_normalize_expand, **compile_args)
    current = make_current_runner(
        compiled_gating,
        compiled_rearrange,
        compiled_norm,
    )
    merged = torch.compile(merged_post_conv_prep, **compile_args)
    print(
        f"mixed_qkv_shape={tuple(mixed_qkv.shape)} "
        f"qk_heads={QK_HEADS} value_heads={VALUE_HEADS} "
        f"compile_args={compile_args}",
        flush=True,
    )

    reference, current_ms = benchmark(
        "current_multi_graph",
        current,
        inputs,
        args.warmups,
        args.iterations,
    )
    candidate, merged_ms = benchmark(
        "merged_compile_graph",
        merged,
        inputs,
        args.warmups,
        args.iterations,
    )
    compare(reference, candidate)
    print(
        f"speedup={current_ms / merged_ms:.6f} "
        f"saved_ms={current_ms - merged_ms:.6f} "
        f"projected_48_layer_saved_ms={(current_ms - merged_ms) * 48:.6f}",
        flush=True,
    )
    if args.trace_dir is not None:
        capture_trace("current_multi_graph", current, inputs, args.trace_dir)
        capture_trace("merged_compile_graph", merged, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
