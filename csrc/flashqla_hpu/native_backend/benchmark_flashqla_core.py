import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gdr_phase_a,
    hpu_chunk_gdr_phase_b,
    hpu_chunk_gdr_preprocess,
    hpu_flashqla_chunk_gdr_phase_a,
)
from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
CHUNK_SIZE = 64
NUM_CHUNKS = TOKENS // CHUNK_SIZE
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_REPEAT = VALUE_HEADS // QK_HEADS
HEAD_DIM = 128


def gdn_core(
    q,
    k,
    v,
    gate,
    beta,
    initial_state,
    phase_a,
    deferred_output_add,
    factorized_local_decay,
    gate_cumsum_bf16=False,
    compact_repeated_local_attn=False,
    preserve_compact_qk=False,
    compact_qk_factor_gate=False,
    native_recurrent_scan=None,
    centered_gate_free_phase_a=False,
    phase_a_head_major=False,
    solver_base=16,
    native_compact_kkt=False,
):
    (
        qf,
        kf,
        vf,
        bf,
        gate_cumsum,
        init_state,
        heads,
        num_chunks,
        scale,
        key_dim,
        value_dim,
        num_seqs,
    ) = hpu_chunk_gdr_preprocess(
        q,
        k,
        v,
        gate,
        beta,
        None,
        initial_state,
        False,
        CHUNK_SIZE,
        1,
        TOKENS,
        False,
        preserve_compact_qk,
    )
    if gate_cumsum_bf16:
        gate_cumsum = gate_cumsum.to(torch.bfloat16)
    phase_a_kwargs = {}
    if phase_a is hpu_flashqla_chunk_gdr_phase_a:
        phase_a_kwargs["return_local_decay"] = True
        phase_a_kwargs["compact_qk_factor_gate"] = compact_qk_factor_gate
        phase_a_kwargs["native_compact_kkt"] = native_compact_kkt
    phase_a_result = phase_a(
        qf,
        kf,
        vf,
        bf,
        gate_cumsum,
        TOKENS,
        CHUNK_SIZE,
        num_seqs,
        num_chunks,
        heads,
        key_dim,
        value_dim,
        14,
        solver_base,
        True,
        HEAD_REPEAT,
        compact_qk_inputs=preserve_compact_qk,
        **phase_a_kwargs,
    )
    if phase_a is hpu_flashqla_chunk_gdr_phase_a:
        (
            u,
            w,
            q_chunks,
            k_chunks,
            gate_chunks,
            local_decay,
        ) = phase_a_result
    else:
        u, w, q_chunks, k_chunks, gate_chunks = phase_a_result
        local_decay = None
    return hpu_chunk_gdr_phase_b(
        u,
        w,
        q_chunks,
        k_chunks,
        gate_chunks,
        init_state,
        scale,
        num_seqs,
        num_chunks,
        TOKENS,
        heads,
        key_dim,
        value_dim,
        True,
        torch.float32,
        True,
        deferred_output_add,
        factorized_local_decay,
        local_decay,
        compact_repeated_local_attn,
        HEAD_REPEAT,
        preserve_compact_qk,
        native_recurrent_scan,
        centered_gate_free_phase_a,
        phase_a_head_major,
    )


def control(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_chunk_gdr_phase_a,
        False,
        False,
    )


def baseline_deferred(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_chunk_gdr_phase_a,
        True,
        False,
    )


def flashqla(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        True,
    )


def flashqla_gate_bf16(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        True,
        True,
    )


def flash_phase_a_only(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
    )


def flash_phase_a_solver_base_4(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        solver_base=4,
    )


def flash_phase_a_solver_base_8(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        solver_base=8,
    )


def flash_phase_a_solver_base_32(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        solver_base=32,
    )


def flash_phase_a_native_control(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    )


def flash_phase_a_native_recurrent(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    )


def flash_phase_a_compact_local_attn(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        False,
        True,
    )


def flash_phase_a_compact_qk(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        False,
        True,
        True,
    )


def flash_phase_a_compact_qk_native(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        deferred_output_add=True,
        factorized_local_decay=False,
        compact_repeated_local_attn=True,
        preserve_compact_qk=True,
        native_compact_kkt=True,
    )


def flash_phase_a_compact_qk_factored_gate(
    q,
    k,
    v,
    gate,
    beta,
    initial_state,
):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_flashqla_chunk_gdr_phase_a,
        True,
        False,
        False,
        True,
        True,
        True,
    )


def factorized_phase_b_only(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        hpu_chunk_gdr_phase_a,
        True,
        True,
    )


def _flash_phase_a_factorized_gate(*args, **kwargs):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        **kwargs,
        factorized_gate_transform=True,
    )


def flashqla_factorized_gate(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_factorized_gate,
        True,
        True,
        centered_gate_free_phase_a=True,
        phase_a_head_major=True,
    )


def _flash_phase_a_fp32_rhs(*args):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        fp32_rhs_scaling=True,
    )


def _flash_phase_a_fp32_output(*args):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        fp32_output_scaling=True,
    )


def _flash_phase_a_fp32_both(*args):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        fp32_rhs_scaling=True,
        fp32_output_scaling=True,
    )


def _flash_phase_a_center_zero(*args):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        fp32_rhs_scaling=True,
        fp32_output_scaling=True,
        gate_center_mode=1,
    )


def _flash_phase_a_center_first(*args):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        fp32_rhs_scaling=True,
        fp32_output_scaling=True,
        gate_center_mode=2,
    )


def _flash_phase_a_center_last(*args):
    return hpu_flashqla_chunk_gdr_phase_a(
        *args,
        fp32_rhs_scaling=True,
        fp32_output_scaling=True,
        gate_center_mode=3,
    )


def flash_phase_a_fp32_rhs(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_fp32_rhs,
        True,
        False,
    )


def flash_phase_a_fp32_output(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_fp32_output,
        True,
        False,
    )


def flash_phase_a_fp32_both(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_fp32_both,
        True,
        False,
    )


def flash_phase_a_center_zero(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_center_zero,
        True,
        False,
    )


def flash_phase_a_center_first(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_center_first,
        True,
        False,
    )


def flash_phase_a_center_last(q, k, v, gate, beta, initial_state):
    return gdn_core(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        _flash_phase_a_center_last,
        True,
        False,
    )


def make_inputs(expanded_qk_input=False):
    torch.manual_seed(739251)
    q = torch.randn(
        1,
        TOKENS,
        QK_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    )
    k = torch.randn_like(q)
    q /= torch.linalg.vector_norm(q.float(), dim=-1, keepdim=True)
    k /= torch.linalg.vector_norm(k.float(), dim=-1, keepdim=True)
    if expanded_qk_input:
        q = q.repeat_interleave(HEAD_REPEAT, dim=2)
        k = k.repeat_interleave(HEAD_REPEAT, dim=2)
    v = torch.randn(
        1,
        TOKENS,
        VALUE_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.01
    gate = -torch.rand(
        1,
        TOKENS,
        VALUE_HEADS,
        dtype=torch.float32,
        device="hpu",
    ) * 0.02
    beta = torch.rand(
        1,
        TOKENS,
        VALUE_HEADS,
        dtype=torch.bfloat16,
        device="hpu",
    )
    initial_state = torch.randn(
        1,
        VALUE_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        dtype=torch.float32,
        device="hpu",
    ) * 0.01
    return q, k, v, gate, beta, initial_state


def benchmark(name, function, inputs, iterations):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(2):
        output = function(*inputs)
    torch.hpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    median_ms = statistics.median(samples)
    print(
        f"{name} median_ms={median_ms:.6f} "
        f"core_tokens_per_second={TOKENS / (median_ms / 1_000):.3f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    saved = tuple(value.detach().clone() for value in output)
    torch.hpu.synchronize()
    return saved


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


def report_difference(actual, expected, name):
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    relative_l2 = torch.linalg.vector_norm(actual_float - expected_float)
    relative_l2 /= torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    print(
        f"{name}_max_abs={float(difference.max().cpu()):.9e} "
        f"mean_abs={float(difference.mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e} "
        f"equal_fraction="
        f"{float((actual == expected).float().mean().cpu()):.9f}",
        flush=True,
    )
    if float(relative_l2.cpu()) >= 0.02:
        raise AssertionError(f"{name} exceeded the 2% relative-L2 gate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--optimized-only", action="store_true")
    parser.add_argument("--safe-only", action="store_true")
    parser.add_argument("--precision-variants-only", action="store_true")
    parser.add_argument("--gate-precision-only", action="store_true")
    parser.add_argument("--expanded-qk-input", action="store_true")
    parser.add_argument("--compact-local-attn-only", action="store_true")
    parser.add_argument("--compact-qk-end-to-end-only", action="store_true")
    parser.add_argument("--native-recurrent-only", action="store_true")
    parser.add_argument("--factorized-gate-only", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--solver-base-sweep-only", action="store_true")
    parser.add_argument("--solver-base-8-only", action="store_true")
    parser.add_argument("--native-compact-kkt-only", action="store_true")
    parser.add_argument("--extension", type=Path)
    return parser.parse_args()


def main():
    global TOKENS, NUM_CHUNKS
    args = parse_args()
    if args.tokens <= 0 or args.tokens % CHUNK_SIZE:
        raise ValueError(
            f"tokens must be a positive multiple of {CHUNK_SIZE}"
        )
    TOKENS = args.tokens
    NUM_CHUNKS = TOKENS // CHUNK_SIZE
    if args.native_compact_kkt_only:
        if args.extension is None:
            raise ValueError("--native-compact-kkt-only requires --extension")
        torch.ops.load_library(str(args.extension.resolve()))
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    inputs = make_inputs(args.expanded_qk_input)
    if args.native_compact_kkt_only:
        if args.expanded_qk_input:
            raise ValueError(
                "--native-compact-kkt-only requires compact Q/K inputs"
            )
        functions = {
            "flashqla_compact_kkt_control": flash_phase_a_compact_qk,
            "flashqla_compact_kkt_native": (
                flash_phase_a_compact_qk_native
            ),
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_compact_kkt_control"]
        candidate = outputs["flashqla_compact_kkt_native"]
        report_difference(
            candidate[0], reference[0], "native_compact_kkt_output"
        )
        report_difference(
            candidate[1],
            reference[1],
            "native_compact_kkt_final_state",
        )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.solver_base_8_only:
        functions = {
            "flashqla_solver_base_8": flash_phase_a_solver_base_8,
            "flashqla_solver_base_16": flash_phase_a_only,
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_solver_base_16"]
        candidate = outputs["flashqla_solver_base_8"]
        report_difference(
            candidate[0],
            reference[0],
            "solver_base_8_output",
        )
        report_difference(
            candidate[1],
            reference[1],
            "solver_base_8_final_state",
        )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.solver_base_sweep_only:
        functions = {
            "flashqla_solver_base_16": flash_phase_a_only,
            "flashqla_solver_base_4": flash_phase_a_solver_base_4,
            "flashqla_solver_base_8": flash_phase_a_solver_base_8,
            "flashqla_solver_base_32": flash_phase_a_solver_base_32,
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_solver_base_16"]
        for base in (4, 8, 32):
            candidate = outputs[f"flashqla_solver_base_{base}"]
            report_difference(
                candidate[0],
                reference[0],
                f"solver_base_{base}_output",
            )
            report_difference(
                candidate[1],
                reference[1],
                f"solver_base_{base}_final_state",
            )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.factorized_gate_only:
        functions = {"flashqla_factorized_gate": flashqla_factorized_gate}
        if not args.candidate_only:
            functions = {"flashqla_safe_control": flash_phase_a_only, **functions}
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        if not args.candidate_only:
            reference = outputs["flashqla_safe_control"]
            candidate = outputs["flashqla_factorized_gate"]
            report_difference(
                candidate[0], reference[0], "factorized_gate_output"
            )
            report_difference(
                candidate[1],
                reference[1],
                "factorized_gate_final_state",
            )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.native_recurrent_only:
        functions = {
            "flashqla_native_control": flash_phase_a_native_control,
            "flashqla_native_recurrent": flash_phase_a_native_recurrent,
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_native_control"]
        candidate = outputs["flashqla_native_recurrent"]
        report_difference(candidate[0], reference[0], "native_recurrent_output")
        report_difference(
            candidate[1],
            reference[1],
            "native_recurrent_final_state",
        )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.compact_qk_end_to_end_only:
        if args.expanded_qk_input:
            raise ValueError(
                "--compact-qk-end-to-end-only requires compact Q/K inputs"
            )
        functions = {
            "flashqla_safe": flash_phase_a_only,
            "flashqla_compact_qk": flash_phase_a_compact_qk,
            "flashqla_compact_qk_factored_gate": (
                flash_phase_a_compact_qk_factored_gate
            ),
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_safe"]
        candidate = outputs["flashqla_compact_qk"]
        report_difference(candidate[0], reference[0], "compact_qk_output")
        report_difference(
            candidate[1],
            reference[1],
            "compact_qk_final_state",
        )
        factored = outputs["flashqla_compact_qk_factored_gate"]
        report_difference(
            factored[0],
            reference[0],
            "compact_qk_factored_gate_output",
        )
        report_difference(
            factored[1],
            reference[1],
            "compact_qk_factored_gate_final_state",
        )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.compact_local_attn_only:
        functions = {
            "flashqla_safe": flash_phase_a_only,
            "flashqla_compact_local_attn": (
                flash_phase_a_compact_local_attn
            ),
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_safe"]
        candidate = outputs["flashqla_compact_local_attn"]
        report_difference(candidate[0], reference[0], "compact_local_output")
        report_difference(
            candidate[1],
            reference[1],
            "compact_local_final_state",
        )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return
    if args.safe_only:
        compiled_safe = torch.compile(flash_phase_a_only, **compile_args)
        print(
            f"tokens={TOKENS} chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
            f"value_heads={VALUE_HEADS} dim={HEAD_DIM}",
            flush=True,
        )
        benchmark(
            "flashqla_safe",
            compiled_safe,
            inputs,
            args.iterations,
        )
        if args.trace_dir is not None:
            capture_trace(
                "flashqla_safe",
                compiled_safe,
                inputs,
                args.trace_dir,
            )
        return
    if args.optimized_only:
        compiled_flashqla = torch.compile(flashqla, **compile_args)
        print(
            f"tokens={TOKENS} chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
            f"value_heads={VALUE_HEADS} dim={HEAD_DIM}",
            flush=True,
        )
        benchmark(
            "flashqla_deferred",
            compiled_flashqla,
            inputs,
            args.iterations,
        )
        if args.trace_dir is not None:
            capture_trace(
                "flashqla_deferred",
                compiled_flashqla,
                inputs,
                args.trace_dir,
            )
        return

    if args.precision_variants_only:
        functions = {
            "baseline_deferred": baseline_deferred,
            "flash_phase_a_only": flash_phase_a_only,
            "flash_phase_a_fp32_rhs": flash_phase_a_fp32_rhs,
            "flash_phase_a_fp32_output": flash_phase_a_fp32_output,
            "flash_phase_a_fp32_both": flash_phase_a_fp32_both,
            "flash_phase_a_center_zero": flash_phase_a_center_zero,
            "flash_phase_a_center_first": flash_phase_a_center_first,
            "flash_phase_a_center_last": flash_phase_a_center_last,
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        print(
            f"tokens={TOKENS} chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
            f"value_heads={VALUE_HEADS} dim={HEAD_DIM}",
            flush=True,
        )
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["baseline_deferred"]
        for name, output in outputs.items():
            report_difference(
                output[0],
                reference[0],
                f"{name}_output",
            )
            report_difference(
                output[1],
                reference[1],
                f"{name}_final_state",
            )
        return

    if args.gate_precision_only:
        functions = {
            "flashqla_fp32_gate": flashqla,
            "flashqla_bf16_gate": flashqla_gate_bf16,
        }
        compiled = {
            name: torch.compile(function, **compile_args)
            for name, function in functions.items()
        }
        print(
            f"tokens={TOKENS} chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
            f"value_heads={VALUE_HEADS} dim={HEAD_DIM}",
            flush=True,
        )
        outputs = {
            name: benchmark(name, function, inputs, args.iterations)
            for name, function in compiled.items()
        }
        reference = outputs["flashqla_fp32_gate"]
        candidate = outputs["flashqla_bf16_gate"]
        report_difference(
            candidate[0],
            reference[0],
            "bf16_gate_output",
        )
        report_difference(
            candidate[1],
            reference[1],
            "bf16_gate_final_state",
        )
        if args.trace_dir is not None:
            for name, function in compiled.items():
                capture_trace(name, function, inputs, args.trace_dir)
        return

    compiled_control = torch.compile(control, **compile_args)
    compiled_baseline = torch.compile(baseline_deferred, **compile_args)
    compiled_flash_phase_a = torch.compile(flash_phase_a_only, **compile_args)
    compiled_factorized_phase_b = torch.compile(
        factorized_phase_b_only,
        **compile_args,
    )
    compiled_flashqla = torch.compile(flashqla, **compile_args)
    print(
        f"tokens={TOKENS} chunks={NUM_CHUNKS} qk_heads={QK_HEADS} "
        f"value_heads={VALUE_HEADS} dim={HEAD_DIM}",
        flush=True,
    )
    control_output = benchmark(
        "control",
        compiled_control,
        inputs,
        args.iterations,
    )
    deferred_output = benchmark(
        "baseline_deferred",
        compiled_baseline,
        inputs,
        args.iterations,
    )
    flash_phase_a_output = benchmark(
        "flash_phase_a_only",
        compiled_flash_phase_a,
        inputs,
        args.iterations,
    )
    factorized_phase_b_output = benchmark(
        "factorized_phase_b_only",
        compiled_factorized_phase_b,
        inputs,
        args.iterations,
    )
    flashqla_output = benchmark(
        "flashqla_deferred",
        compiled_flashqla,
        inputs,
        args.iterations,
    )
    report_difference(
        deferred_output[0],
        control_output[0],
        "deferred_output",
    )
    report_difference(
        deferred_output[1],
        control_output[1],
        "deferred_final_state",
    )
    report_difference(
        flash_phase_a_output[0],
        deferred_output[0],
        "flash_phase_a_only_output",
    )
    report_difference(
        flash_phase_a_output[1],
        deferred_output[1],
        "flash_phase_a_only_final_state",
    )
    report_difference(
        factorized_phase_b_output[0],
        deferred_output[0],
        "factorized_phase_b_only_output",
    )
    report_difference(
        factorized_phase_b_output[1],
        deferred_output[1],
        "factorized_phase_b_only_final_state",
    )
    report_difference(
        flashqla_output[0],
        control_output[0],
        "flashqla_output",
    )
    report_difference(
        flashqla_output[1],
        control_output[1],
        "flashqla_final_state",
    )
    if args.trace_dir is not None:
        capture_trace(
            "control",
            compiled_control,
            inputs,
            args.trace_dir,
        )
        capture_trace(
            "flashqla_deferred",
            compiled_flashqla,
            inputs,
            args.trace_dir,
        )


if __name__ == "__main__":
    main()
