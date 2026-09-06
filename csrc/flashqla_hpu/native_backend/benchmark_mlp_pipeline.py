import argparse
import statistics
import time
from pathlib import Path

import torch

from benchmark_mlp_scale_cguid import (
    FP8_DTYPE,
    FP8_MAX,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    TOKENS,
    current_per_token_quant,
    fp8_linear_current,
    mlp_current,
    swiglu,
)
from vllm_gaudi.utils import HPUCompileConfig


def _chunks(x, count):
    chunk_size = x.shape[0] // count
    return tuple(
        x[index * chunk_size:(index + 1) * chunk_size]
        for index in range(count)
    )


def _mlp_interleaved(
    x,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    count,
):
    outputs = []
    for chunk in _chunks(x, count):
        outputs.append(
            mlp_current(
                chunk,
                gate_up_weight,
                gate_up_scale,
                down_weight,
                down_scale,
            )
        )
    return torch.cat(outputs, dim=0)


def _mlp_phased(
    x,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    count,
):
    gate_up_chunks = [
        fp8_linear_current(chunk, gate_up_weight, gate_up_scale)
        for chunk in _chunks(x, count)
    ]
    output_chunks = [
        fp8_linear_current(swiglu(gate_up), down_weight, down_scale)
        for gate_up in gate_up_chunks
    ]
    return torch.cat(output_chunks, dim=0)


def _mlp_down_chunked(
    x,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    count,
):
    gate_up = fp8_linear_current(x, gate_up_weight, gate_up_scale)
    outputs = [
        fp8_linear_current(swiglu(chunk), down_weight, down_scale)
        for chunk in _chunks(gate_up, count)
    ]
    return torch.cat(outputs, dim=0)


def mlp_interleaved_2(*args):
    return _mlp_interleaved(*args, 2)


def mlp_interleaved_4(*args):
    return _mlp_interleaved(*args, 4)


def mlp_interleaved_8(*args):
    return _mlp_interleaved(*args, 8)


def mlp_phased_2(*args):
    return _mlp_phased(*args, 2)


def mlp_phased_4(*args):
    return _mlp_phased(*args, 4)


def mlp_phased_8(*args):
    return _mlp_phased(*args, 8)


def mlp_down_chunked_2(*args):
    return _mlp_down_chunked(*args, 2)


def mlp_down_chunked_4(*args):
    return _mlp_down_chunked(*args, 4)


def mlp_down_chunked_8(*args):
    return _mlp_down_chunked(*args, 8)


def _mlp_moe(
    x,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    expert_ids,
    router_weights,
    chunk_size,
):
    quantized, input_scale = current_per_token_quant(x)
    return torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic(
        hidden_states=quantized,
        expert_routing_table=expert_ids,
        router_weights=router_weights,
        w12=[gate_up_weight],
        w3=[down_weight],
        d_scale_hidden_states=input_scale,
        d_scale_w12=[gate_up_scale],
        d_scale_w3=[down_scale],
        permuted_weights=False,
        activation="silu",
        experts_min=0,
        experts_max=0,
        chunk_size=chunk_size,
        total_experts=1,
    )


def mlp_moe_0(*args):
    return _mlp_moe(*args, 0)


def mlp_moe_1024(*args):
    return _mlp_moe(*args, 1024)


def mlp_moe_4096(*args):
    return _mlp_moe(*args, 4096)


def mlp_moe_8192(*args):
    return _mlp_moe(*args, 8192)


def make_stream_runner(compiled_chunk, count):
    streams = [torch.hpu.Stream() for _ in range(count)]
    output = torch.empty(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    )
    chunk_size = TOKENS // count

    def run(
        x,
        gate_up_weight,
        gate_up_scale,
        down_weight,
        down_scale,
    ):
        current = torch.hpu.current_stream()
        for index, stream in enumerate(streams):
            start = index * chunk_size
            end = start + chunk_size
            with torch.hpu.stream(stream):
                chunk_output = compiled_chunk(
                    x[start:end],
                    gate_up_weight,
                    gate_up_scale,
                    down_weight,
                    down_scale,
                )
                output[start:end].copy_(chunk_output)
        for stream in streams:
            current.wait_stream(stream)
        return output

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

    saved = output.detach().clone()
    torch.hpu.synchronize()
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return saved


def compare_output(name, reference, candidate):
    difference = candidate.float() - reference.float()
    relative_l2 = torch.linalg.vector_norm(difference)
    relative_l2 /= torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    print(
        f"quality {name} max_abs={float(difference.abs().max().cpu()):.9e} "
        f"equal_fraction={float((candidate == reference).float().mean().cpu()):.9f} "
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

    traces = sorted(output_dir.glob(f"{name}*.pt.trace.json.gz"))
    if not traces:
        raise RuntimeError(f"Profiler did not create a trace for {name}")
    trace_path = output_dir / f"{name}.json.gz"
    traces[-1].replace(trace_path)
    print(f"trace={trace_path}", flush=True)


def quantize_weight(rows, columns):
    weight_bf16 = torch.randn(
        (rows, columns),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.02
    scale = (weight_bf16.abs().amax(dim=0, keepdim=True) + 1e-8) / FP8_MAX
    weight = torch.ops.hpu.cast_to_fp8_v2(
        weight_bf16,
        scale.reciprocal(),
        False,
        False,
        FP8_DTYPE,
    )[0]
    return weight, scale.squeeze(0).float()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace", default="")
    parser.add_argument("--only", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    hidden = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.5
    gate_up_weight, gate_up_scale = quantize_weight(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE * 2,
    )
    down_weight, down_scale = quantize_weight(
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    )
    torch.hpu.synchronize()

    functions = {
        "mlp_current": mlp_current,
        "mlp_interleaved_2": mlp_interleaved_2,
        "mlp_interleaved_4": mlp_interleaved_4,
        "mlp_interleaved_8": mlp_interleaved_8,
        "mlp_phased_2": mlp_phased_2,
        "mlp_phased_4": mlp_phased_4,
        "mlp_phased_8": mlp_phased_8,
        "mlp_down_chunked_2": mlp_down_chunked_2,
        "mlp_down_chunked_4": mlp_down_chunked_4,
        "mlp_down_chunked_8": mlp_down_chunked_8,
        "mlp_moe_0": mlp_moe_0,
        "mlp_moe_1024": mlp_moe_1024,
        "mlp_moe_4096": mlp_moe_4096,
        "mlp_moe_8192": mlp_moe_8192,
    }
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled = {
        name: torch.compile(function, **compile_args)
        for name, function in functions.items()
    }
    stream_functions = {
        "mlp_stream_2": make_stream_runner(compiled["mlp_current"], 2),
        "mlp_stream_4": make_stream_runner(compiled["mlp_current"], 4),
    }
    benchmark_functions = compiled | stream_functions
    inputs = (
        hidden,
        gate_up_weight,
        gate_up_scale,
        down_weight,
        down_scale,
    )
    expert_ids = torch.zeros(
        (TOKENS, 1),
        dtype=torch.int64,
        device="hpu",
    )
    router_weights = torch.ones(
        (TOKENS, 1),
        dtype=torch.bfloat16,
        device="hpu",
    )
    moe_inputs = inputs + (expert_ids, router_weights)
    inputs_by_name = {
        name: (moe_inputs if name.startswith("mlp_moe_") else inputs)
        for name in benchmark_functions
    }
    print(
        f"tokens={TOKENS} hidden={HIDDEN_SIZE} intermediate={INTERMEDIATE_SIZE} "
        f"compile_args={compile_args}",
        flush=True,
    )

    outputs = {}
    selected_names = list(benchmark_functions)
    if args.only:
        requested = args.only.split(",")
        selected_names = list(dict.fromkeys(["mlp_current", *requested]))
    for name in selected_names:
        function = benchmark_functions[name]
        outputs[name] = benchmark(
            name,
            function,
            inputs_by_name[name],
            args.warmups,
            args.iterations,
        )
    reference = outputs["mlp_current"]
    for name, output in outputs.items():
        compare_output(name, reference, output)

    if args.trace_dir:
        trace_names = args.trace.split(",") if args.trace else list(benchmark_functions)
        for name in trace_names:
            capture_trace(name, benchmark_functions[name], inputs_by_name[name], args.trace_dir)


if __name__ == "__main__":
    main()
