import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gdr_phase_a,
    hpu_flashqla_chunk_gdr_phase_a,
)
from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
NUM_CHUNKS = 256
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_REPEAT = VALUE_HEADS // QK_HEADS
CHUNK_SIZE = 64
KEY_DIM = 128
VALUE_DIM = 128


def phase_a_baseline(q, k, v, beta, gate):
    return hpu_chunk_gdr_phase_a(
        q,
        k,
        v,
        beta,
        gate,
        TOKENS,
        CHUNK_SIZE,
        1,
        NUM_CHUNKS,
        VALUE_HEADS,
        KEY_DIM,
        VALUE_DIM,
        14,
        16,
        True,
        HEAD_REPEAT,
    )


def phase_a_gate_free(q, k, v, beta, gate):
    return hpu_flashqla_chunk_gdr_phase_a(
        q,
        k,
        v,
        beta,
        gate,
        TOKENS,
        CHUNK_SIZE,
        1,
        NUM_CHUNKS,
        VALUE_HEADS,
        KEY_DIM,
        VALUE_DIM,
        14,
        16,
        True,
        HEAD_REPEAT,
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


def report_difference(name, actual, expected):
    actual_sample = actual[::64, :2, :8].float()
    expected_sample = expected[::64, :2, :8].float()
    difference = (actual_sample - expected_sample).abs()
    relative_l2 = torch.linalg.vector_norm(actual_sample - expected_sample)
    relative_l2 /= torch.linalg.vector_norm(expected_sample).clamp_min(1e-12)
    print(
        f"{name}_max_abs={float(difference.max().cpu()):.9e} "
        f"mean_abs={float(difference.mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e}",
        flush=True,
    )
    if float(relative_l2.cpu()) >= 0.02:
        raise AssertionError(f"{name} exceeded the 2% relative-L2 gate")


def main() -> None:
    args = parse_args()
    torch.manual_seed(739251)
    q_unique = torch.randn(
        TOKENS,
        QK_HEADS,
        KEY_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    )
    k_unique = torch.randn_like(q_unique)
    q_unique /= torch.linalg.vector_norm(q_unique.float(), dim=-1, keepdim=True)
    k_unique /= torch.linalg.vector_norm(k_unique.float(), dim=-1, keepdim=True)
    q = q_unique.repeat_interleave(HEAD_REPEAT, dim=1)
    k = k_unique.repeat_interleave(HEAD_REPEAT, dim=1)
    v = torch.randn(
        TOKENS,
        VALUE_HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.01
    beta = torch.rand(
        TOKENS,
        VALUE_HEADS,
        dtype=torch.bfloat16,
        device="hpu",
    )
    raw_gate = -torch.rand(
        1,
        NUM_CHUNKS,
        CHUNK_SIZE,
        VALUE_HEADS,
        dtype=torch.float32,
        device="hpu",
    ) * 0.02
    gate = torch.cumsum(raw_gate, dim=2).reshape(TOKENS, VALUE_HEADS)
    inputs = q, k, v, beta, gate

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    baseline = torch.compile(phase_a_baseline, **compile_args)
    gate_free = torch.compile(phase_a_gate_free, **compile_args)
    print(
        f"tokens={TOKENS} chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
        f"value_heads={VALUE_HEADS} dim={KEY_DIM}",
        flush=True,
    )

    expected = benchmark("baseline", baseline, inputs, args.iterations)
    actual = benchmark("gate_free", gate_free, inputs, args.iterations)
    report_difference("u", actual[0], expected[0])
    report_difference("w", actual[1], expected[1])

    if args.trace_dir is not None:
        capture_trace("baseline", baseline, inputs, args.trace_dir)
        capture_trace("gate_free", gate_free, inputs, args.trace_dir)


if __name__ == "__main__":
    main()
