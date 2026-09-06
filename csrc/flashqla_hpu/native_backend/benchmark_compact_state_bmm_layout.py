import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


NUM_CHUNKS = 256
QK_HEADS = 16
HEAD_REPEAT = 3
CHUNK_SIZE = 64
HEAD_DIM = 128


def current_transposed_output(grouped_u, grouped_w, compact_k):
    grouped_k = compact_k.unsqueeze(2)
    n = torch.matmul(grouped_u.transpose(-1, -2), grouped_k)
    r = torch.matmul(grouped_w.transpose(-1, -2), grouped_k)
    return n.transpose(-1, -2), r.transpose(-1, -2)


def direct_output(grouped_u, grouped_w, compact_k):
    grouped_k_t = compact_k.transpose(-1, -2).unsqueeze(2)
    return (
        torch.matmul(grouped_k_t, grouped_u),
        torch.matmul(grouped_k_t, grouped_w),
    )


def direct_fused_rhs(grouped_u, grouped_w, compact_k):
    grouped_k_t = compact_k.transpose(-1, -2).unsqueeze(2)
    combined = torch.matmul(
        grouped_k_t,
        torch.cat((grouped_u, grouped_w), dim=-1),
    )
    return combined.split(HEAD_DIM, dim=-1)


def direct_unrolled_repeat(grouped_u, grouped_w, compact_k):
    compact_k_t = compact_k.transpose(-1, -2)
    n = torch.stack(
        [
            torch.matmul(compact_k_t, grouped_u[:, :, repeat])
            for repeat in range(HEAD_REPEAT)
        ],
        dim=2,
    )
    r = torch.stack(
        [
            torch.matmul(compact_k_t, grouped_w[:, :, repeat])
            for repeat in range(HEAD_REPEAT)
        ],
        dim=2,
    )
    return n, r


def direct_repeat_outer(grouped_u, grouped_w, compact_k):
    u_repeat_outer = grouped_u.permute(2, 0, 1, 3, 4)
    w_repeat_outer = grouped_w.permute(2, 0, 1, 3, 4)
    k_repeat_outer = compact_k.transpose(-1, -2).unsqueeze(0)
    n = torch.matmul(k_repeat_outer, u_repeat_outer)
    r = torch.matmul(k_repeat_outer, w_repeat_outer)
    return n.permute(1, 2, 0, 3, 4), r.permute(1, 2, 0, 3, 4)


def direct_flattened(grouped_u, grouped_w, compact_k):
    batch = NUM_CHUNKS * QK_HEADS * HEAD_REPEAT
    expanded_k = compact_k.unsqueeze(2).expand(
        NUM_CHUNKS,
        QK_HEADS,
        HEAD_REPEAT,
        CHUNK_SIZE,
        HEAD_DIM,
    )
    k_flat = expanded_k.reshape(batch, CHUNK_SIZE, HEAD_DIM)
    u_flat = grouped_u.reshape(batch, CHUNK_SIZE, HEAD_DIM)
    w_flat = grouped_w.reshape(batch, CHUNK_SIZE, HEAD_DIM)
    n = torch.bmm(k_flat.transpose(1, 2), u_flat)
    r = torch.bmm(k_flat.transpose(1, 2), w_flat)
    shape = NUM_CHUNKS, QK_HEADS, HEAD_REPEAT, HEAD_DIM, HEAD_DIM
    return n.reshape(shape), r.reshape(shape)


def benchmark(name, function, inputs, warmups, samples):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(warmups):
        output = function(*inputs)
    torch.hpu.synchronize()

    times_ms = []
    for _ in range(samples):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1_000)
    print(
        f"{name} median_ms={statistics.median(times_ms):.6f} "
        f"min_ms={min(times_ms):.6f} samples_ms={times_ms}",
        flush=True,
    )
    return tuple(tensor.clone() for tensor in output)


def report_quality(name, reference, candidate):
    for index, (expected, actual) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        difference = actual.float() - expected.float()
        relative_l2 = difference.norm() / expected.float().norm().clamp_min(
            1e-12
        )
        print(
            f"quality {name}_{index} equal={torch.equal(actual, expected)} "
            f"rel_l2={float(relative_l2.cpu()):.9e} "
            f"mean_abs={float(difference.abs().mean().cpu()):.9e} "
            f"max_abs={float(difference.abs().max().cpu()):.9e}",
            flush=True,
        )


def capture_trace(name, function, inputs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
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
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    shape = (
        NUM_CHUNKS,
        QK_HEADS,
        HEAD_REPEAT,
        CHUNK_SIZE,
        HEAD_DIM,
    )
    grouped_u = torch.randn(shape, dtype=torch.bfloat16, device="hpu")
    grouped_w = torch.randn_like(grouped_u)
    compact_k = torch.randn(
        NUM_CHUNKS,
        QK_HEADS,
        CHUNK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    )
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    functions = {
        "current_transposed_output": current_transposed_output,
        "direct_output": direct_output,
        "direct_fused_rhs": direct_fused_rhs,
        "direct_unrolled_repeat": direct_unrolled_repeat,
        "direct_repeat_outer": direct_repeat_outer,
        "direct_flattened": direct_flattened,
    }
    compiled = {
        name: torch.compile(function, **compile_args)
        for name, function in functions.items()
    }
    inputs = grouped_u, grouped_w, compact_k
    print(
        f"chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
        f"head_repeat={HEAD_REPEAT} chunk={CHUNK_SIZE} dim={HEAD_DIM}",
        flush=True,
    )
    outputs = {
        name: benchmark(
            name,
            function,
            inputs,
            args.warmups,
            args.samples,
        )
        for name, function in compiled.items()
    }
    reference = outputs["current_transposed_output"]
    for name, output in outputs.items():
        if name != "current_transposed_output":
            report_quality(name, reference, output)

    if args.trace_dir is not None:
        for name, function in compiled.items():
            capture_trace(name, function, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
