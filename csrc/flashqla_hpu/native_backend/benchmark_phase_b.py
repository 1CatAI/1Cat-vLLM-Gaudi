import argparse
import math
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.ops.hpu_gdn_pytorch import (
    _hpu_chunk_gdr_phase_b_optimized,
)
from vllm_gaudi.utils import HPUCompileConfig


NUM_CHUNKS = 256
NUM_HEADS = 48
QK_HEADS = 16
QK_HEAD_REPEAT = NUM_HEADS // QK_HEADS
CHUNK_SIZE = 64
KEY_DIM = 128
VALUE_DIM = 128


def phase_b_baseline(u, w, q, k, g, initial_state):
    return _hpu_chunk_gdr_phase_b_optimized(
        u,
        w,
        q,
        k,
        g,
        initial_state,
        KEY_DIM**-0.5,
        1,
        NUM_CHUNKS,
        NUM_CHUNKS * CHUNK_SIZE,
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        True,
        torch.float32,
        True,
        False,
        False,
        None,
    )


def phase_b_compact_qk(u, w, q, k, g, initial_state):
    return _hpu_chunk_gdr_phase_b_optimized(
        u,
        w,
        q,
        k,
        g,
        initial_state,
        KEY_DIM**-0.5,
        1,
        NUM_CHUNKS,
        NUM_CHUNKS * CHUNK_SIZE,
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        True,
        torch.float32,
        True,
        False,
        False,
        None,
    )


def phase_b_compact_qk_deferred(u, w, q, k, g, initial_state):
    return _hpu_chunk_gdr_phase_b_optimized(
        u,
        w,
        q,
        k,
        g,
        initial_state,
        KEY_DIM**-0.5,
        1,
        NUM_CHUNKS,
        NUM_CHUNKS * CHUNK_SIZE,
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        True,
        torch.float32,
        True,
        True,
        False,
        None,
    )


def phase_b_deferred(u, w, q, k, g, initial_state):
    return _hpu_chunk_gdr_phase_b_optimized(
        u,
        w,
        q,
        k,
        g,
        initial_state,
        KEY_DIM**-0.5,
        1,
        NUM_CHUNKS,
        NUM_CHUNKS * CHUNK_SIZE,
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        True,
        torch.float32,
        True,
        True,
        False,
        None,
    )


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

    traces = sorted(output_dir.glob(f"{name}*.pt.trace.json.gz"))
    if not traces:
        raise RuntimeError(f"Profiler did not create a trace for {name}")
    trace_path = output_dir / f"{name}.json.gz"
    traces[-1].replace(trace_path)
    print(f"trace={trace_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--trace-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(739251)
    shape = (1, NUM_CHUNKS, CHUNK_SIZE, NUM_HEADS, KEY_DIM)
    u = torch.randn(shape, dtype=torch.float32, device="hpu") * 0.01
    w = torch.randn(shape, dtype=torch.float32, device="hpu") * 0.01
    q_unique = torch.randn(
        1,
        NUM_CHUNKS,
        CHUNK_SIZE,
        QK_HEADS,
        KEY_DIM,
        dtype=torch.float32,
        device="hpu",
    ) * 0.01
    k_unique = torch.randn_like(q_unique) * 0.01
    q = q_unique.repeat_interleave(QK_HEAD_REPEAT, dim=3)
    k = k_unique.repeat_interleave(QK_HEAD_REPEAT, dim=3)
    raw_g = -torch.rand(
        1,
        NUM_CHUNKS,
        CHUNK_SIZE,
        NUM_HEADS,
        dtype=torch.float32,
        device="hpu",
    ) * 0.02
    g = torch.cumsum(raw_g, dim=2)
    initial_state = torch.randn(
        1,
        NUM_HEADS,
        VALUE_DIM,
        KEY_DIM,
        dtype=torch.float32,
        device="hpu",
    ) * 0.01
    inputs = u, w, q, k, g, initial_state

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    baseline = torch.compile(phase_b_baseline, **compile_args)
    compact_qk = torch.compile(phase_b_compact_qk, **compile_args)
    compact_qk_deferred = torch.compile(
        phase_b_compact_qk_deferred,
        **compile_args,
    )
    deferred = torch.compile(phase_b_deferred, **compile_args)
    print(
        f"tokens={NUM_CHUNKS * CHUNK_SIZE} chunks={NUM_CHUNKS} "
        f"qk_heads={QK_HEADS} value_heads={NUM_HEADS} "
        f"qk_repeat={QK_HEAD_REPEAT} dim={KEY_DIM} "
        f"dtype=float32 scale={1 / math.sqrt(KEY_DIM):.9f}",
        flush=True,
    )

    expected = benchmark("baseline", baseline, inputs, args.iterations)
    compact = benchmark("compact_phase_b_qk", compact_qk, inputs, args.iterations)
    actual = benchmark(
        "compact_phase_b_qk_deferred",
        compact_qk_deferred,
        inputs,
        args.iterations,
    )
    deferred_output = benchmark(
        "phase_b_deferred",
        deferred,
        inputs,
        args.iterations,
    )
    expected_output = expected[0].float()
    actual_output = actual[0].float()
    difference = (actual_output - expected_output).abs()
    relative_l2 = torch.linalg.vector_norm(actual_output - expected_output)
    relative_l2 /= torch.linalg.vector_norm(expected_output).clamp_min(1e-12)
    print(
        f"output_max_abs={float(difference.max().cpu()):.9e} "
        f"output_mean_abs={float(difference.mean().cpu()):.9e} "
        f"output_relative_l2={float(relative_l2.cpu()):.9e} "
        f"output_equal_fraction="
        f"{float((actual[0] == expected[0]).float().mean().cpu()):.9f}",
        flush=True,
    )
    torch.testing.assert_close(actual[0].cpu(), expected[0].cpu(), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual[1].cpu(), expected[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual[0].cpu(), compact[0].cpu(), rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(actual[1].cpu(), compact[1].cpu(), rtol=0, atol=0)
    print(
        "output_correctness=pass final_state=bitwise_pass "
        "deferred_vs_compact=tolerance_pass",
        flush=True,
    )
    torch.testing.assert_close(deferred_output[0].cpu(), expected[0].cpu(), rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(deferred_output[1].cpu(), expected[1].cpu(), rtol=0, atol=0)
    print(
        "deferred_vs_baseline=tolerance_pass "
        "deferred_final_state=bitwise_pass",
        flush=True,
    )

    if args.trace_dir is not None:
        capture_trace("baseline", baseline, inputs, args.trace_dir)
        capture_trace("compact_phase_b_qk", compact_qk, inputs, args.trace_dir)
        capture_trace(
            "compact_phase_b_qk_deferred",
            compact_qk_deferred,
            inputs,
            args.trace_dir,
        )
        capture_trace("phase_b_deferred", deferred, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
