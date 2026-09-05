# SPDX-License-Identifier: Apache-2.0
"""Compare FlashInfer-Gaudi GDN tactics on production Qwen decode shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flashinfer_gaudi import get_capabilities
from flashinfer_gaudi._native import bridge_packed_gdn_op, public_packed_gdn_op
from flashinfer_gaudi._reference import packed_recurrent_decode
from flashinfer_gaudi.gdn_decode import BackendUnavailableError, _call_native_packed


def _synchronize() -> None:
    torch.hpu.synchronize()


def _make_inputs(batch: int):
    generator = torch.Generator().manual_seed(31 + batch)
    key_heads, value_heads, dim = 16, 48, 128
    width = 2 * key_heads * dim + value_heads * dim
    packed = torch.randn(batch, width, dtype=torch.bfloat16, generator=generator).to("hpu")
    log_decay = (-torch.rand(batch, value_heads, dtype=torch.float32, generator=generator)).to("hpu")
    beta = torch.sigmoid(torch.randn(batch, value_heads, dtype=torch.float32,
                                     generator=generator)).to(torch.bfloat16).to("hpu")
    state = torch.randn(batch + 1, value_heads, dim, dim, dtype=torch.float32, generator=generator).to("hpu")
    indices = torch.arange(1, batch + 1, dtype=torch.int32).to("hpu")
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32).to("hpu")
    return packed, log_decay, beta, state, indices, cu_seqlens


def _compile(policy: str, reference_layout: str = "direct"):
    if policy == "pytorch":

        def run(packed, log_decay, beta, state, indices, cu_seqlens):
            del cu_seqlens
            if reference_layout == "direct":
                direct_state = state.narrow(0, 1, packed.shape[0])
                return packed_recurrent_decode(
                    packed,
                    log_decay,
                    beta,
                    direct_state,
                    None,
                    None,
                    None,
                    True,
                    True,
                )[0]
            return packed_recurrent_decode(
                packed,
                log_decay,
                beta,
                state,
                indices,
                indices,
                None,
                True,
            )[0]

    elif policy == "legacy":
        from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_recurrent_gated_delta_rule

        def run(packed, log_decay, beta, state, indices, cu_seqlens):
            key_heads, value_heads, dim = 16, 48, 128
            q_raw, k_raw, v_raw = packed.split((key_heads * dim, key_heads * dim, value_heads * dim), dim=-1)
            batch = packed.shape[0]
            output, _ = hpu_fused_recurrent_gated_delta_rule(
                q=q_raw.reshape(1, batch, key_heads, dim).contiguous(),
                k=k_raw.reshape(1, batch, key_heads, dim).contiguous(),
                v=v_raw.reshape(1, batch, value_heads, dim).contiguous(),
                g=log_decay.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=state,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )
            return output.squeeze(0)

    else:
        op = public_packed_gdn_op() if policy == "public" else bridge_packed_gdn_op()
        if op is None:
            raise BackendUnavailableError(f"The {policy} backend is not loaded.")

        def run(packed, log_decay, beta, state, indices, cu_seqlens):
            del cu_seqlens
            return _call_native_packed(op, packed, log_decay, beta, state, indices)

    # vLLM executes fixed HPU decode buckets. Dynamic recipes distort this
    # comparison and can select materially different kernels from production.
    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


def _warmup_pair(reference, reference_inputs, candidate, candidate_inputs, warmups: int) -> None:
    for _ in range(warmups):
        reference(*reference_inputs)
        candidate(*candidate_inputs)
    _synchronize()


def _measure_synchronized_pair(
    reference,
    reference_inputs,
    candidate,
    candidate_inputs,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float]]:
    samples: tuple[list[float], list[float]] = ([], [])
    measurements = ((reference, reference_inputs, samples[0]), (candidate, candidate_inputs, samples[1]))
    for _ in range(iterations):
        ordered = measurements if len(samples[0]) % 2 == 0 else tuple(reversed(measurements))
        for function, inputs, target in ordered:
            started = time.perf_counter_ns()
            function(*inputs)
            _synchronize()
            target.append((time.perf_counter_ns() - started) / 1e6)
    return tuple({
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
    } for values in samples)


def _measure_wave(function, inputs, iterations: int) -> tuple[float, float]:
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function(*inputs)
    end_event.record()
    _synchronize()
    return ((time.perf_counter_ns() - started) / 1e6 / iterations, start_event.elapsed_time(end_event) / iterations)


def _measure_wave_pair(
    reference,
    reference_inputs,
    candidate,
    candidate_inputs,
    iterations: int,
    waves: int,
) -> tuple[dict[str, float], dict[str, float]]:
    samples: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    measurements = ((reference, reference_inputs, samples[0]), (candidate, candidate_inputs, samples[1]))
    for wave in range(waves):
        ordered = measurements if wave % 2 == 0 else tuple(reversed(measurements))
        for function, inputs, target in ordered:
            target.append(_measure_wave(function, inputs, iterations))
    return tuple({
        "queued_median_ms": statistics.median(sample[0] for sample in values),
        "device_median_ms": statistics.median(sample[1] for sample in values),
        "device_min_ms": min(sample[1] for sample in values),
    } for values in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,16,32")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing-mode", choices=("sync", "wave", "both"), default="both")
    parser.add_argument("--wave-iterations", type=int, default=100)
    parser.add_argument("--waves", type=int, default=7)
    parser.add_argument("--backend", choices=("public", "bridge", "pytorch"), default="public")
    parser.add_argument("--reference-layout", choices=("direct", "indexed", "legacy"), default="direct")
    args = parser.parse_args()

    results: dict[str, object] = {"capabilities": get_capabilities(), "batches": {}}
    for batch in (int(item) for item in args.batches.split(",") if item):
        reference_inputs = _make_inputs(batch)
        candidate_inputs = tuple(tensor.clone() for tensor in reference_inputs)
        reference_policy = "legacy" if args.reference_layout == "legacy" else "pytorch"
        reference = _compile(reference_policy, args.reference_layout)
        candidate = _compile(args.backend, "direct")

        reference_output = reference(*reference_inputs)
        candidate_output = candidate(*candidate_inputs)
        _synchronize()
        candidate_output_cpu = candidate_output.cpu()
        reference_output_cpu = reference_output.cpu()
        candidate_state_cpu = candidate_inputs[3].cpu()
        reference_state_cpu = reference_inputs[3].cpu()
        output_max_abs = (candidate_output_cpu.float() - reference_output_cpu.float()).abs().max().item()
        state_max_abs = (candidate_state_cpu - reference_state_cpu).abs().max().item()
        # The TPC reduction tree is not bit-identical to the graph-compiler
        # reduction. Keep a tight state gate and a BF16-appropriate output
        # gate; model-level token validation remains required for promotion.
        torch.testing.assert_close(candidate_output_cpu, reference_output_cpu, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(candidate_state_cpu, reference_state_cpu, atol=2e-5, rtol=2e-4)

        _warmup_pair(reference, reference_inputs, candidate, candidate_inputs, args.warmups)
        reference_stats: dict[str, float] = {}
        candidate_stats: dict[str, float] = {}
        synchronized_speedup = None
        device_speedup = None
        if args.timing_mode in ("sync", "both"):
            reference_sync, candidate_sync = _measure_synchronized_pair(
                reference,
                reference_inputs,
                candidate,
                candidate_inputs,
                args.iterations,
            )
            reference_stats.update(reference_sync)
            candidate_stats.update(candidate_sync)
            synchronized_speedup = reference_sync["median_ms"] / candidate_sync["median_ms"]
        if args.timing_mode in ("wave", "both"):
            reference_wave, candidate_wave = _measure_wave_pair(
                reference,
                reference_inputs,
                candidate,
                candidate_inputs,
                args.wave_iterations,
                args.waves,
            )
            reference_stats.update(reference_wave)
            candidate_stats.update(candidate_wave)
            device_speedup = reference_wave["device_median_ms"] / candidate_wave["device_median_ms"]
        reference_name = "vllm_legacy" if args.reference_layout == "legacy" else f"pytorch_{args.reference_layout}"
        results["batches"][str(batch)] = {
            reference_name: reference_stats,
            args.backend: candidate_stats,
            "output_max_abs": output_max_abs,
            "state_max_abs": state_max_abs,
        }
        if synchronized_speedup is not None:
            results["batches"][str(batch)]["median_speedup"] = synchronized_speedup
        if device_speedup is not None:
            results["batches"][str(batch)]["device_speedup"] = device_speedup

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
