# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Gaudi2 DFlash2 GDN verification kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flashinfer_gaudi._native import public_mtp_gdn_op
from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_packed_reference

_TOKENS = 8
_QK_HEADS = 16
_VALUE_HEADS = 48
_DIM = 128
_PACKED_WIDTH = (2 * _QK_HEADS + _VALUE_HEADS) * _DIM


def _make_inputs(batch: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1201 + batch)
    packed = torch.randn(
        batch,
        _TOKENS,
        _PACKED_WIDTH,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    log_decay = -torch.rand(batch, _TOKENS, _VALUE_HEADS, generator=generator) * 0.1
    beta = torch.sigmoid(torch.randn(batch, _TOKENS, _VALUE_HEADS, dtype=torch.bfloat16, generator=generator))
    state = torch.randn(
        batch * _TOKENS + 2,
        _VALUE_HEADS,
        _DIM,
        _DIM,
        generator=generator,
    ) * 0.01
    state_indices = torch.arange(1, batch * _TOKENS + 1, dtype=torch.int32).reshape(batch, _TOKENS)
    accepted = torch.arange(batch, dtype=torch.int32).remainder(_TOKENS).add_(1)
    query_lengths = torch.full((batch, ), _TOKENS, dtype=torch.int32)
    if batch > 1:
        query_lengths[-1] = _TOKENS - 3
    return tuple(
        tensor.to("hpu") for tensor in (
            packed,
            log_decay,
            beta,
            state,
            state_indices,
            accepted,
            query_lengths,
        ))


def _compile_reference():

    def run(*inputs: torch.Tensor) -> torch.Tensor:
        output, _ = _gated_delta_rule_mtp_packed_reference(*inputs)
        return output

    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


def _compile_native(op):

    def run(
        packed_qkv: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return op(
            state_pool,
            packed_qkv,
            torch.exp(log_decay).contiguous(),
            beta,
            state_indices,
            num_accepted_tokens,
            query_lengths,
        )

    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


def _measure(function, inputs: tuple[torch.Tensor, ...], iterations: int) -> tuple[float, float]:
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function(*inputs)
    end_event.record()
    torch.hpu.synchronize()
    host_ms = (time.perf_counter_ns() - started) / 1e6 / iterations
    return host_ms, start_event.elapsed_time(end_event) / iterations


def _stats(samples: list[tuple[float, float]]) -> dict[str, float]:
    return {
        "host_median_ms": statistics.median(item[0] for item in samples),
        "device_median_ms": statistics.median(item[1] for item in samples),
        "device_min_ms": min(item[1] for item in samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--wave-iterations", type=int, default=100)
    parser.add_argument("--waves", type=int, default=7)
    args = parser.parse_args()

    native_op = public_mtp_gdn_op()
    if native_op is None:
        raise RuntimeError("The native DFlash2 MTP op is not loaded. Rebuild with "
                           "`python3 tools/build_flashinfer_gaudi.py`.")

    report: dict[str, object] = {"batches": {}}
    for batch in (int(value) for value in args.batches.split(",") if value):
        torch._dynamo.reset()
        base_inputs = _make_inputs(batch)
        reference_inputs = (*base_inputs[:3], base_inputs[3].clone(), *base_inputs[4:])
        native_inputs = (*base_inputs[:3], base_inputs[3].clone(), *base_inputs[4:])

        reference = _compile_reference()
        reference_output = reference(*reference_inputs)
        native = _compile_native(native_op)
        native_output = native(*native_inputs)
        torch.hpu.synchronize()

        output_error = float((native_output.float() - reference_output.float()).abs().max().cpu())
        state_error = float((native_inputs[3] - reference_inputs[3]).abs().max().cpu())
        torch.testing.assert_close(native_output.cpu(), reference_output.cpu(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(native_inputs[3].cpu(), reference_inputs[3].cpu(), rtol=2e-2, atol=2e-2)

        for _ in range(args.warmups):
            reference(*reference_inputs)
            native(*native_inputs)
        torch.hpu.synchronize()

        samples: dict[str, list[tuple[float, float]]] = {
            "pytorch": [],
            "native": [],
        }
        order = (
            ("pytorch", reference, reference_inputs),
            ("native", native, native_inputs),
        )
        for wave in range(args.waves):
            wave_order = order if wave % 2 == 0 else tuple(reversed(order))
            for name, function, inputs in wave_order:
                samples[name].append(_measure(function, inputs, args.wave_iterations))

        reference_stats = _stats(samples["pytorch"])
        native_stats = _stats(samples["native"])
        report["batches"][str(batch)] = {
            "pytorch": reference_stats,
            "native": native_stats,
            "device_speedup": reference_stats["device_median_ms"] / native_stats["device_median_ms"],
            "host_speedup": reference_stats["host_median_ms"] / native_stats["host_median_ms"],
            "output_max_abs": output_error,
            "state_max_abs": state_error,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
