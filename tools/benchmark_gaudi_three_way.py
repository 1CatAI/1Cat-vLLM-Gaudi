# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen-sized HPU, FlashInfer-Gaudi, and Triton operator graphs.

All candidates use the same inputs, precision, Bridge, stream, and fullgraph
compiler. FlashInfer's shared vendor ops are labelled explicitly. Failed
correctness checks never produce a speedup. This is not an end-to-end benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--ops", default="rmsnorm,silu,quant,gdn,gdn_conv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--correctness-steps", type=int, default=3)
    parser.add_argument("--isolate-cases",
                        action="store_true",
                        help="Use a fresh process for each shape; omit to diagnose cross-shape recipe reuse")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    operations = args.ops.split(",")
    if any(value <= 0 for value in (*batches, args.warmup, args.iterations, args.rounds, args.correctness_steps)):
        parser.error("batches and iteration counts must be positive")
    if not set(operations) <= {"rmsnorm", "silu", "quant", "gdn", "gdn_conv"}:
        parser.error("unknown operator")

    if args.isolate_cases:
        case_dir = args.output.parent / (args.output.stem + "-cases")
        case_dir.mkdir(parents=True, exist_ok=True)
        case_dir = Path(tempfile.mkdtemp(prefix="run-", dir=case_dir))
        combined = {"schema_version": 1, "isolation": "fresh_process_per_case", "cases": []}
        failed = False
        for operation in operations:
            for batch in batches:
                output = case_dir / f"{operation}-b{batch}.json"
                command = [
                    sys.executable, __file__, "--ops", operation, "--batches",
                    str(batch), "--warmup",
                    str(args.warmup), "--iterations",
                    str(args.iterations), "--rounds",
                    str(args.rounds), "--correctness-steps",
                    str(args.correctness_steps), "--output",
                    str(output)
                ]
                with output.with_suffix(".log").open("w") as log:
                    child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                failed |= child.returncode != 0
                if output.exists():
                    result = json.loads(output.read_text())
                    combined["cases"].extend(result.pop("cases"))
                    combined["metadata"] = result
                else:
                    combined["cases"].append({
                        "op": operation,
                        "batch": batch,
                        "error": f"child exited {child.returncode} before writing results"
                    })
                args.output.write_text(json.dumps(combined, indent=2) + "\n")
                print(f"{operation} batch={batch}: exit={child.returncode}", flush=True)
        raise SystemExit(int(failed))

    os.environ["VLLM_HPU_TRITON_MODE"] = "strict"
    os.environ["VLLM_GDN_COMPUTE_FP32"] = "1"
    from triton.backends.gaudi.driver import prepare_environment
    prepare_environment()

    import torch
    import torch.nn.functional as F
    from habana_frameworks.torch.hpex.normalization.FusedRMSNorm import FusedRMSNorm
    from flashinfer_gaudi._reference import packed_recurrent_decode, qwen38_fused_decode_step_direct
    from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
    from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating, hpu_fused_recurrent_gated_delta_rule
    from vllm_gaudi.ops.triton_gaudi import runtime

    runtime.prepare_if_enabled()
    names = ("pytorch_hpu", "flashinfer_gaudi", "triton_gaudi")
    results = {
        "schema_version": 1,
        "scope": "operator_fullgraph_not_model_throughput",
        "torch_version": torch.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "module_id": os.environ.get("HLS_MODULE_ID"),
        "timing": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds
        },
        "cases": [],
    }

    def persist():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(args.output)

    def make_inputs(operation, batch):
        generator = torch.Generator().manual_seed(20260905 + batch)

        def rand(*shape, scale=0.1, dtype=torch.bfloat16):
            return torch.randn(shape, dtype=dtype, generator=generator) * scale

        if operation == "rmsnorm":
            return (rand(batch, 5120), rand(batch, 5120), rand(5120, scale=0.1) + 1), ()
        if operation in ("silu", "quant"):
            return (rand(batch, 34816 if operation == "silu" else 5120), ), ()
        # The owned contiguous cache is also the indexed pool for the baseline
        # and Triton. No candidate gets a smaller allocation or lower precision.
        tensors = (
            rand(batch, 48, 128, 128, scale=0.01, dtype=torch.float32),
            rand(batch, 10240),
            rand(batch, 48),
            rand(batch, 48),
            rand(48, dtype=torch.float32) - 2,
            rand(48).float(),
            torch.arange(batch, dtype=torch.int32),
            torch.arange(batch + 1, dtype=torch.int32),
        )
        if operation == "gdn_conv":
            tensors += (rand(batch, 3, 10240), rand(10240, 4, scale=0.05))
        return tensors, (0, 8) if operation == "gdn_conv" else (0, )

    def make_function(operation, backend):

        def run(*inputs):
            if operation == "rmsnorm":
                x, residual, weight = inputs
                if backend == "triton_gaudi":
                    result = runtime.fused_add_rms_norm(x, residual, weight, 1e-6)
                    if result is None:
                        raise RuntimeError("Triton RMSNorm fallback")
                    return result
                summed = x + residual
                return FusedRMSNorm.apply(summed, weight, 1e-6), summed
            if operation == "silu":
                x, = inputs
                if backend == "triton_gaudi":
                    result = runtime.silu_and_mul(x)
                    if result is None:
                        raise RuntimeError("Triton SiLU fallback")
                    return result
                gate, value = x.chunk(2, dim=-1)
                return F.silu(gate) * value
            if operation == "quant":
                x, = inputs
                if backend == "triton_gaudi":
                    result = runtime.dynamic_quant(x)
                    if result is None:
                        raise RuntimeError("Triton quantization fallback")
                    return result
                if backend == "flashinfer_gaudi":
                    scale = torch.ops.hpu.calculate_scale_for_cast(x, 2, 0, -1, True, 240.0, 1.0)
                    scale = scale + (1e-8 / 240.0)
                else:
                    scale = (x.abs().amax(dim=-1, keepdim=True) + 1e-8) / 240.0
                quantized = torch.ops.hpu.cast_to_fp8_v2(x, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
                return quantized, scale.float()
            state, packed, a, b, a_log, dt_bias, indices, cu_seqlens = inputs[:8]
            if operation == "gdn_conv":
                conv_state, conv_weight = inputs[8:]
                if backend == "flashinfer_gaudi":
                    output, _, _ = qwen38_fused_decode_step_direct(packed, a, b, a_log, dt_bias, conv_state,
                                                                   conv_weight, None, state, 128**-0.5)
                    return output
                packed = hpu_causal_conv1d_update(x=packed,
                                                  conv_state=conv_state,
                                                  weight=conv_weight,
                                                  bias=None,
                                                  activation="silu",
                                                  query_start_loc=cu_seqlens,
                                                  direct_state_layout=True)
            if backend == "triton_gaudi":
                result = runtime.gdn_decode_packed(state, packed, a, b, a_log, dt_bias, indices)
                if result is None:
                    raise RuntimeError("Triton GDN fallback")
                return result
            log_decay, beta = hpu_fused_gdn_gating(a_log, a, b, dt_bias)
            if backend == "flashinfer_gaudi":
                output, _ = packed_recurrent_decode(packed, log_decay, beta, state, indices, indices, 128**-0.5, True,
                                                    True)
                return output
            batch = packed.shape[0]
            q, k, v = packed.split((2048, 2048, 6144), dim=-1)
            output, _ = hpu_fused_recurrent_gated_delta_rule(q=q.reshape(1, batch, 16, 128).contiguous(),
                                                             k=k.reshape(1, batch, 16, 128).contiguous(),
                                                             v=v.reshape(1, batch, 48, 128).contiguous(),
                                                             g=log_decay,
                                                             beta=beta,
                                                             initial_state=state,
                                                             inplace_final_state=True,
                                                             cu_seqlens=cu_seqlens,
                                                             ssm_state_indices=indices,
                                                             use_qk_l2norm_in_kernel=True)
            return output.squeeze(0)

        return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)

    def flatten(output):
        return output if isinstance(output, tuple) else (output, )

    def timed(function, inputs):
        start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        start.record()
        begin = time.perf_counter_ns()
        for _ in range(args.iterations):
            function(*inputs)
        end.record()
        end.synchronize()
        return {
            "device_ms": start.elapsed_time(end) / args.iterations,
            "wall_ms": (time.perf_counter_ns() - begin) / 1e6 / args.iterations
        }

    for operation in operations:
        for batch in batches:
            case = {"op": operation, "batch": batch, "backends": {}}
            results["cases"].append(case)
            cpu_inputs, mutated = make_inputs(operation, batch)
            functions, inputs = {}, {}
            reference_outputs, reference_states = [], []
            for name in names:
                record = {"status": "pending"}
                case["backends"][name] = record
                if name == "flashinfer_gaudi":
                    if operation in ("rmsnorm", "silu"):
                        record["shared_implementation_with"] = "pytorch_hpu"
                    record["implementation"] = {
                        "rmsnorm": "shared_hpu_vendor_rmsnorm",
                        "silu": "shared_hpu_silu_mul",
                        "quant": "hpu_cguid_scale_and_cast",
                        "gdn": "compiled_direct_state_graph",
                        "gdn_conv": "compiled_fused_direct_state_graph",
                    }[operation]
                elif name == "triton_gaudi":
                    implementation = "generated_tpc" if operation != "gdn_conv" else "hpu_conv_plus_generated_tpc"
                    record["implementation"] = implementation
                    if operation in ("gdn", "gdn_conv"):
                        record["placement"] = "native_graph" if batch == 8 else "custom_op_partition"
                else:
                    record["implementation"] = "mainline_hpu_compiled_graph"
                try:
                    function = make_function(operation, name)
                    device_inputs = tuple(tensor.to("hpu") for tensor in cpu_inputs)
                    output_max_abs, state_max_abs, state_rmse = 0.0, 0.0, 0.0
                    for step in range(args.correctness_steps):
                        output = function(*device_inputs)
                        torch.hpu.synchronize()
                        output_cpu = tuple(tensor.cpu().float() for tensor in flatten(output))
                        state_cpu = tuple(device_inputs[index].cpu().float() for index in mutated)
                        if name == names[0]:
                            reference_outputs.append(output_cpu)
                            reference_states.append(state_cpu)
                        else:
                            if operation == "quant":
                                actual_q, actual_scale = output_cpu
                                expected_q, expected_scale = reference_outputs[step]
                                # Use the existing dynamic-quant numerical gate.
                                # F32 scale arithmetic can move FP8 rounding bins;
                                # report code equality without claiming bit parity.
                                torch.testing.assert_close(actual_scale, expected_scale, rtol=0.005, atol=1e-8)
                                torch.testing.assert_close(actual_q * actual_scale,
                                                           expected_q * expected_scale,
                                                           rtol=0.08,
                                                           atol=0.02)
                                record["quantized_equal_fraction"] = (actual_q == expected_q).float().mean().item()
                                record["scale_max_abs"] = (actual_scale - expected_scale).abs().max().item()
                                record["dequantized_max_abs"] = (actual_q * actual_scale -
                                                                 expected_q * expected_scale).abs().max().item()
                                record["correctness_gate"] = "existing_fp8_numeric_tolerance_not_bitwise"
                            for actual, expected in zip(output_cpu, reference_outputs[step]):
                                output_max_abs = max(output_max_abs, (actual - expected).abs().max().item())
                                if operation != "quant":
                                    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.002)
                            for index, actual, expected in zip(mutated, state_cpu, reference_states[step]):
                                difference = actual - expected
                                state_max_abs = max(state_max_abs, difference.abs().max().item())
                                state_rmse = max(state_rmse, difference.square().mean().sqrt().item())
                                if index == 8:
                                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                else:
                                    torch.testing.assert_close(actual, expected, rtol=0.025, atol=2e-5)
                                    if difference.square().mean().sqrt().item() > 5e-5:
                                        raise AssertionError("GDN state RMSE exceeded 5e-5")
                    for index in mutated:
                        if torch.equal(device_inputs[index].cpu(), cpu_inputs[index]):
                            raise AssertionError("state mutation did not reach the input cache")
                    record.update(status="correct",
                                  output_max_abs=output_max_abs,
                                  state_max_abs=state_max_abs,
                                  state_rmse=state_rmse)
                    functions[name], inputs[name] = function, device_inputs
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(json.dumps({"op": operation, "batch": batch, "backend": name, **record}), flush=True)
                persist()
                if name == names[0] and record["status"] == "failed":
                    break
            active = list(functions)
            for name in active:
                for tensor, original in zip(inputs[name], cpu_inputs):
                    tensor.copy_(original)
                for _ in range(args.warmup):
                    functions[name](*inputs[name])
            torch.hpu.synchronize()
            samples = {name: [] for name in active}
            for round_index in range(args.rounds):
                order = active[round_index % len(active):] + active[:round_index % len(active)] if active else []
                for name in order:
                    samples[name].append(timed(functions[name], inputs[name]))
            for name in active:
                record = case["backends"][name]
                record.update(status="passed", samples=samples[name])
                for metric in ("device_ms", "wall_ms"):
                    record[metric] = statistics.median(sample[metric] for sample in samples[name])
            if names[0] in active:
                reference = case["backends"][names[0]]
                for name in active:
                    record = case["backends"][name]
                    record["device_speedup"] = reference["device_ms"] / record["device_ms"]
                    record["wall_speedup"] = reference["wall_ms"] / record["wall_ms"]
            print(json.dumps(case), flush=True)
            persist()
            torch._dynamo.reset()
    if any(record["status"] == "failed" for case in results["cases"] for record in case["backends"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
