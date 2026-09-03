# SPDX-License-Identifier: Apache-2.0
"""Compare the general HPU and FlashQLA-graph GDN prefill tactics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from flashinfer_gaudi.gdn_prefill import _chunk_gated_delta_rule_log_gate


def _synchronize() -> None:
    torch.hpu.synchronize()


def _make_inputs(tokens: int, *, pre_normalize_qk: bool) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(97 + tokens)
    q = torch.randn(1, tokens, 16, 128, dtype=torch.float32, generator=generator)
    k = torch.randn(1, tokens, 16, 128, dtype=torch.float32, generator=generator)
    if pre_normalize_qk:
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
    q = q.to(torch.bfloat16).to("hpu")
    k = k.to(torch.bfloat16).to("hpu")
    v = torch.randn(1, tokens, 48, 128, dtype=torch.bfloat16, generator=generator).to("hpu")
    log_decay = (-torch.rand(1, tokens, 48, dtype=torch.float32, generator=generator) * 0.02).to("hpu")
    beta = torch.sigmoid(torch.randn(1, tokens, 48, dtype=torch.float32,
                                     generator=generator)).to(torch.bfloat16).to("hpu")
    state = (torch.randn(1, 48, 128, 128, dtype=torch.float32, generator=generator) * 0.01).to("hpu")
    return q, k, v, log_decay, beta, state


def _compile(
    *,
    flashqla: bool,
    compact_qk: bool = False,
    qk_l2norm: str = "none",
    masked_triangular_decay: bool = False,
):

    use_qk_l2norm = qk_l2norm != "none"
    compile_qk_l2norm = qk_l2norm == "compiled"

    def run(q, k, v, log_decay, beta, initial_state):
        return _chunk_gated_delta_rule_log_gate(
            q,
            k,
            v,
            log_decay,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            chunk_size=128,
            prefill_num_seqs=1,
            prefill_seq_len=q.shape[1],
            neumann_iters=14,
            fused_state_matmul=flashqla,
            recursive_solver_base=16 if flashqla else 0,
            compact_repeated_kkt=flashqla,
            compile_qk_l2norm=compile_qk_l2norm,
            flashqla_reformulation=flashqla,
            deferred_output_add=flashqla,
            compute_dtype=torch.float32,
            solve_in_fp32=flashqla,
            state_in_fp32=flashqla,
            preserve_compact_qk=compact_qk,
            masked_triangular_decay=masked_triangular_decay,
        )

    # The production compatibility helper deliberately creates a Dynamo
    # graph boundary. ``fullgraph=False`` is required to reproduce it.
    return torch.compile(run, backend="hpu_backend", fullgraph=qk_l2norm != "eager", dynamic=False)


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(candidate.float() - reference.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return float((numerator / denominator).cpu().item())


def _measure_wave(function, inputs: tuple[torch.Tensor, ...], iterations: int) -> float:
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function(*inputs)
    end.record()
    _synchronize()
    return start.elapsed_time(end) / iterations


def _measure_pair(reference, candidate, inputs, iterations: int, waves: int) -> tuple[dict[str, float], ...]:
    samples: tuple[list[float], list[float]] = ([], [])
    measurements = ((reference, samples[0]), (candidate, samples[1]))
    for wave in range(waves):
        ordered = measurements if wave % 2 == 0 else tuple(reversed(measurements))
        for function, target in ordered:
            target.append(_measure_wave(function, inputs, iterations))
    return tuple({
        "device_median_ms": statistics.median(values),
        "device_min_ms": min(values),
        "device_max_ms": max(values),
    } for values in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="2048,4096,16384")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--waves", type=int, default=7)
    parser.add_argument("--max-relative-l2", type=float, default=1e-3)
    parser.add_argument("--reference", choices=("general", "flashqla-expanded", "flashqla-compact"), default="general")
    parser.add_argument("--reference-qk-l2norm", choices=("none", "eager", "compiled"), default="none")
    parser.add_argument("--candidate-qk-l2norm", choices=("none", "eager", "compiled"))
    parser.add_argument("--reference-masked-triangular-decay", action="store_true")
    parser.add_argument("--candidate-masked-triangular-decay", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate_qk_l2norm = args.candidate_qk_l2norm or args.reference_qk_l2norm
    reference_uses_norm = args.reference_qk_l2norm != "none"
    candidate_uses_norm = candidate_qk_l2norm != "none"
    if reference_uses_norm != candidate_uses_norm:
        parser.error("reference and candidate must either both include Q/K L2Norm or both exclude it")

    reference_flashqla = args.reference != "general"
    reference_compact_qk = args.reference == "flashqla-compact"
    reference_name = {
        "general": "general_hpu",
        "flashqla-expanded": "flashqla_expanded",
        "flashqla-compact": "flashqla_compact_qk",
    }[args.reference]
    if args.reference_qk_l2norm != "none":
        reference_name += f"_{args.reference_qk_l2norm}_qk_l2norm"
    candidate_name = "flashqla_compact_qk"
    if candidate_qk_l2norm != "none":
        candidate_name += f"_{candidate_qk_l2norm}_qk_l2norm"
    if reference_name == candidate_name:
        reference_name += "_reference"
    results: dict[str, object] = {
        "operation": "qwen38_gdn_prefill",
        "reference": args.reference,
        "reference_qk_l2norm": args.reference_qk_l2norm,
        "candidate_qk_l2norm": candidate_qk_l2norm,
        "reference_masked_triangular_decay": args.reference_masked_triangular_decay,
        "candidate_masked_triangular_decay": args.candidate_masked_triangular_decay,
        "tokens": {},
    }
    for tokens in (int(value) for value in args.tokens.split(",") if value):
        torch._dynamo.reset()
        inputs = _make_inputs(tokens, pre_normalize_qk=not reference_uses_norm)
        reference = _compile(
            flashqla=reference_flashqla,
            compact_qk=reference_compact_qk,
            qk_l2norm=args.reference_qk_l2norm,
            masked_triangular_decay=args.reference_masked_triangular_decay,
        )
        candidate = _compile(
            flashqla=True,
            compact_qk=True,
            qk_l2norm=candidate_qk_l2norm,
            masked_triangular_decay=args.candidate_masked_triangular_decay,
        )

        reference_output, reference_state = reference(*inputs)
        candidate_output, candidate_state = candidate(*inputs)
        _synchronize()
        assert reference_state is not None and candidate_state is not None
        output_relative_l2 = _relative_l2(candidate_output, reference_output)
        state_relative_l2 = _relative_l2(candidate_state, reference_state)
        quality_values = (output_relative_l2, state_relative_l2)
        if not all(math.isfinite(value) and value <= args.max_relative_l2 for value in quality_values):
            raise AssertionError("FlashQLA prefill failed the relative-L2 gate: "
                                 f"output={output_relative_l2:.6f}, state={state_relative_l2:.6f}")

        for _ in range(args.warmups):
            reference(*inputs)
            candidate(*inputs)
        _synchronize()
        reference_stats, candidate_stats = _measure_pair(reference, candidate, inputs, args.iterations, args.waves)
        results["tokens"][str(tokens)] = {
            reference_name: reference_stats,
            candidate_name: candidate_stats,
            "device_speedup": reference_stats["device_median_ms"] / candidate_stats["device_median_ms"],
            "output_relative_l2": output_relative_l2,
            "state_relative_l2": state_relative_l2,
        }

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
