# SPDX-License-Identifier: Apache-2.0
"""Benchmark indexed versus direct-state Qwen GDN decode convolution."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    _depthwise_conv1d_tpc,
    hpu_causal_conv1d_update,
)


def _synchronize() -> None:
    torch.hpu.synchronize()


def _make_inputs(batch: int):
    generator = torch.Generator().manual_seed(73 + batch)
    dim, width = 10240, 4
    x = torch.randn(batch, dim, dtype=torch.bfloat16, generator=generator).to("hpu")
    weight = torch.randn(dim, width, dtype=torch.bfloat16, generator=generator).to("hpu")
    bias = torch.randn(dim, dtype=torch.bfloat16, generator=generator).to("hpu")
    indexed_state = torch.randn(batch + 2, width - 1, dim, dtype=torch.bfloat16, generator=generator).to("hpu")
    direct_state = indexed_state.narrow(0, 1, batch).clone()
    indices = torch.arange(1, batch + 1, dtype=torch.int32).to("hpu")
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32).to("hpu")
    return (x, weight, bias, indexed_state, indices, query_start_loc), direct_state


def _compile(mode: str):

    def run(x, weight, bias, state, indices, query_start_loc):
        if mode == "direct_window":
            del indices, query_start_loc
            state_rows = state[:, -3:, :]
            window = torch.cat([state_rows.transpose(-1, -2), x.unsqueeze(-1)], dim=2)
            output = torch.nn.functional.silu(_depthwise_conv1d_tpc(window, weight, bias))
            state.copy_(window[:, :, 1:].transpose(-1, -2))
            return output.squeeze(-1)
        direct = mode == "direct"
        return hpu_causal_conv1d_update(
            x=x,
            conv_state=state,
            weight=weight,
            bias=bias,
            activation="silu",
            conv_state_indices=None if direct else indices,
            query_start_loc=query_start_loc,
            direct_state_layout=direct,
        )

    return torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)


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


def _measure_pair(indexed, indexed_inputs, direct, direct_inputs, iterations: int,
                  waves: int) -> tuple[dict[str, float], dict[str, float]]:
    samples: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    measurements = ((indexed, indexed_inputs, samples[0]), (direct, direct_inputs, samples[1]))
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
    parser.add_argument("--wave-iterations", type=int, default=100)
    parser.add_argument("--waves", type=int, default=7)
    args = parser.parse_args()

    results: dict[str, object] = {"batches": {}}
    for batch in (int(item) for item in args.batches.split(",") if item):
        torch._dynamo.reset()
        indexed_inputs, direct_state = _make_inputs(batch)
        direct_inputs = (*indexed_inputs[:3], direct_state, *indexed_inputs[4:])
        window_inputs = (*indexed_inputs[:3], direct_state.clone(), *indexed_inputs[4:])
        indexed = _compile("indexed")
        direct = _compile("direct")
        direct_window = _compile("direct_window")

        indexed_output = indexed(*indexed_inputs)
        direct_output = direct(*direct_inputs)
        window_output = direct_window(*window_inputs)
        _synchronize()
        torch.testing.assert_close(direct_output.cpu(), indexed_output.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(window_output.cpu(), indexed_output.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            direct_inputs[3].cpu(),
            indexed_inputs[3].narrow(0, 1, batch).cpu(),
            rtol=0,
            atol=0,
        )

        for _ in range(args.warmups):
            indexed(*indexed_inputs)
            direct(*direct_inputs)
            direct_window(*window_inputs)
        _synchronize()
        indexed_stats, direct_stats = _measure_pair(
            indexed,
            indexed_inputs,
            direct,
            direct_inputs,
            args.wave_iterations,
            args.waves,
        )
        window_stats, direct_confirmation = _measure_pair(
            direct_window,
            window_inputs,
            direct,
            direct_inputs,
            args.wave_iterations,
            args.waves,
        )
        results["batches"][str(batch)] = {
            "indexed": indexed_stats,
            "direct": direct_stats,
            "direct_confirmation": direct_confirmation,
            "direct_window": window_stats,
            "device_speedup": indexed_stats["device_median_ms"] / direct_stats["device_median_ms"],
            "direct_over_window_speedup": window_stats["device_median_ms"] / direct_confirmation["device_median_ms"],
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
