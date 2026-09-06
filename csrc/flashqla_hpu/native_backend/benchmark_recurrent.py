import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


NUM_CHUNKS = 256
NUM_HEADS = 48
CHUNK_SIZE = 64
KEY_DIM = 128
VALUE_DIM = 128


def recurrent_scan(
    state_projections: torch.Tensor,
    core: torch.Tensor,
    state_bias: torch.Tensor,
    initial_state: torch.Tensor,
):
    state = initial_state
    for chunk in range(NUM_CHUNKS):
        projected = torch.matmul(state_projections[chunk], state)
        core[chunk].add_(projected[:, :CHUNK_SIZE])
        state = projected[:, CHUNK_SIZE:] + state_bias[chunk]
    return core, state


def recurrent_state_first(
    state_projections: torch.Tensor,
    core: torch.Tensor,
    state_bias: torch.Tensor,
    initial_state: torch.Tensor,
):
    state = initial_state
    for chunk in range(NUM_CHUNKS):
        projected = torch.matmul(state_projections[chunk], state)
        next_state = projected[:, CHUNK_SIZE:] + state_bias[chunk]
        core[chunk].add_(projected[:, :CHUNK_SIZE])
        state = next_state
    return core, state


def recurrent_deferred_output(
    state_projections: torch.Tensor,
    core: torch.Tensor,
    state_bias: torch.Tensor,
    initial_state: torch.Tensor,
):
    state = initial_state
    output_terms = []
    for chunk in range(NUM_CHUNKS):
        projected = torch.matmul(state_projections[chunk], state)
        state = projected[:, CHUNK_SIZE:] + state_bias[chunk]
        output_terms.append(projected[:, :CHUNK_SIZE])
    return core + torch.stack(output_terms), state


def recurrent_blocked_output(
    state_projections,
    core,
    state_bias,
    initial_state,
    block_size,
):
    state = initial_state
    output_blocks = []
    for block_start in range(0, NUM_CHUNKS, block_size):
        output_terms = []
        block_end = min(block_start + block_size, NUM_CHUNKS)
        for chunk in range(block_start, block_end):
            projected = torch.matmul(state_projections[chunk], state)
            state = projected[:, CHUNK_SIZE:] + state_bias[chunk]
            output_terms.append(projected[:, :CHUNK_SIZE])
        output_blocks.append(
            core[block_start:block_end] + torch.stack(output_terms)
        )
    return torch.cat(output_blocks), state


def recurrent_blocked_output_16(*args):
    return recurrent_blocked_output(*args, 16)


def recurrent_blocked_output_32(*args):
    return recurrent_blocked_output(*args, 32)


def recurrent_blocked_output_64(*args):
    return recurrent_blocked_output(*args, 64)


def recurrent_split_state_first(
    state_projections: torch.Tensor,
    core: torch.Tensor,
    state_bias: torch.Tensor,
    initial_state: torch.Tensor,
):
    state = initial_state
    for chunk in range(NUM_CHUNKS):
        next_state = torch.matmul(
            state_projections[chunk, :, CHUNK_SIZE:],
            state,
        ) + state_bias[chunk]
        output_term = torch.matmul(
            state_projections[chunk, :, :CHUNK_SIZE],
            state,
        )
        core[chunk].add_(output_term)
        state = next_state
    return core, state


def recurrent_native_scan(
    state_projections: torch.Tensor,
    core: torch.Tensor,
    state_bias: torch.Tensor,
    initial_state: torch.Tensor,
):
    updates, final_state = torch.ops.custom_op.flashqla_recurrent_scan_probe(
        state_projections.reshape(-1, CHUNK_SIZE + KEY_DIM, KEY_DIM),
        core.reshape(-1, CHUNK_SIZE, VALUE_DIM),
        state_bias.reshape(-1, KEY_DIM, VALUE_DIM),
        initial_state,
    )
    return core + updates.view_as(core), final_state


def recurrent_native_scan_bundled16(
    state_projections: torch.Tensor,
    core: torch.Tensor,
    state_bias: torch.Tensor,
    initial_state: torch.Tensor,
):
    updates, final_state = (
        torch.ops.custom_op.flashqla_recurrent_scan_probe_bundled16(
            state_projections.reshape(
                -1,
                CHUNK_SIZE + KEY_DIM,
                KEY_DIM,
            ),
            core.reshape(-1, CHUNK_SIZE, VALUE_DIM),
            state_bias.reshape(-1, KEY_DIM, VALUE_DIM),
            initial_state,
        )
    )
    return core + updates.view_as(core), final_state


FUNCTIONS = {
    "current": recurrent_scan,
    "state_first": recurrent_state_first,
    "deferred_output": recurrent_deferred_output,
    "blocked_output_16": recurrent_blocked_output_16,
    "blocked_output_32": recurrent_blocked_output_32,
    "blocked_output_64": recurrent_blocked_output_64,
    "split_state_first": recurrent_split_state_first,
    "native_scan": recurrent_native_scan,
    "native_scan_bundled16": recurrent_native_scan_bundled16,
}


def fresh_inputs(inputs):
    return inputs[0], inputs[1].clone(), inputs[2], inputs[3]


def benchmark(name, function, inputs, iterations):
    inputs = fresh_inputs(inputs)
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


def run_once(function, inputs):
    output = function(*fresh_inputs(inputs))
    torch.hpu.synchronize()
    return output


def capture_trace(name, function, inputs, output_dir):
    inputs = fresh_inputs(inputs)
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
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument(
        "--trace-variant",
        action="append",
        choices=tuple(FUNCTIONS),
    )
    parser.add_argument(
        "--benchmark-variant",
        action="append",
        choices=tuple(FUNCTIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_variants = {
        *(args.benchmark_variant or []),
        *(args.trace_variant or []),
    }
    if any(name.startswith("native_scan") for name in requested_variants):
        if args.extension is None:
            raise ValueError("Native scan variants require --extension")
        torch.ops.load_library(str(args.extension.resolve()))
    torch.manual_seed(739251)
    dtype = torch.bfloat16
    state_projections = torch.randn(
        NUM_CHUNKS,
        NUM_HEADS,
        CHUNK_SIZE + KEY_DIM,
        KEY_DIM,
        dtype=dtype,
        device="hpu",
    ) * 0.01
    core = torch.randn(
        NUM_CHUNKS,
        NUM_HEADS,
        CHUNK_SIZE,
        VALUE_DIM,
        dtype=dtype,
        device="hpu",
    ) * 0.01
    state_bias = torch.randn(
        NUM_CHUNKS,
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        dtype=dtype,
        device="hpu",
    ) * 0.01
    initial_state = torch.randn(
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        dtype=dtype,
        device="hpu",
    ) * 0.01
    inputs = state_projections, core, state_bias, initial_state

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    selected = args.benchmark_variant or [
        name for name in FUNCTIONS if not name.startswith("native_scan")
    ]
    compile_names = dict.fromkeys([
        "current",
        *selected,
        *(args.trace_variant or []),
    ])
    compiled = {
        name: torch.compile(FUNCTIONS[name], **compile_args)
        for name in compile_names
    }
    print(
        f"chunks={NUM_CHUNKS} heads={NUM_HEADS} chunk={CHUNK_SIZE} "
        f"dim={KEY_DIM} dtype={dtype} compile_args={compile_args}",
        flush=True,
    )
    expected = run_once(compiled["current"], inputs)
    expected_core = expected[0][::64, :2, :4, :4].cpu()
    expected_state = expected[1][:2, :4, :4].cpu()
    benchmark(
        "current",
        compiled["current"],
        inputs,
        args.iterations,
    )
    checksum = float(expected_state.float().sum())
    print(f"current_state_checksum={checksum:.9f}", flush=True)
    for name in selected:
        if name == "current":
            continue
        benchmark(name, compiled[name], inputs, args.iterations)
        output = run_once(compiled[name], inputs)
        torch.testing.assert_close(
            output[0][::64, :2, :4, :4].cpu(),
            expected_core,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output[1][:2, :4, :4].cpu(),
            expected_state,
            rtol=0,
            atol=0,
        )
        print(f"{name}_correctness=bitwise_pass", flush=True)
    if args.trace_dir is not None:
        variants = args.trace_variant or ["current"]
        for name in variants:
            capture_trace(name, compiled[name], inputs, args.trace_dir)


if __name__ == "__main__":
    main()
