# SPDX-License-Identifier: Apache-2.0
"""Qualify the full-query graph-native MTP core, including checkpoint writes.

This benchmark does not enable the experimental route in serving. Numerical
tolerances and microbenchmarks do not replace model-level quality validation.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from unittest import mock

import torch

from flashinfer_gaudi._native import public_mtp_prepared_op
from flashinfer_gaudi.gdn_decode import (
    _gated_delta_rule_mtp_packed_reference,
    _gated_delta_rule_mtp_prepared,
)


def make_inputs(batch: int, slots: int, seed: int) -> tuple[torch.Tensor, ...]:
    rng = torch.Generator().manual_seed(seed)
    packed = torch.randn(batch, 8, 10240, dtype=torch.bfloat16, generator=rng) * 0.1
    decay = -torch.rand(batch, 8, 48, generator=rng) * 0.1
    beta = torch.sigmoid(torch.randn(batch, 8, 48, dtype=torch.bfloat16, generator=rng))
    pool = torch.randn(slots, 48, 128, 128, generator=rng) * 0.01
    indices = torch.arange(1, batch * 8 + 1, dtype=torch.int32).reshape(batch, 8)
    accepted = torch.ones(batch, dtype=torch.int32)
    lengths = torch.full((batch, ), 8, dtype=torch.int32)
    return tuple(x.to("hpu") for x in (packed, decay, beta, pool, indices, accepted, lengths))


def reference(packed, decay, beta, pool, indices, accepted, lengths):
    return _gated_delta_rule_mtp_packed_reference(packed,
                                                  decay,
                                                  beta,
                                                  pool,
                                                  indices,
                                                  accepted,
                                                  lengths,
                                                  assume_full_query=True)[0]


def prepared(packed, decay, beta, pool, indices, accepted, lengths):
    return _gated_delta_rule_mtp_prepared(packed,
                                          decay,
                                          beta,
                                          pool,
                                          indices,
                                          accepted,
                                          assume_distinct_checkpoints=True)[0]


def measure(function, inputs, iterations: int) -> float:
    begin, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function(*inputs)
    end.record()
    torch.hpu.synchronize()
    return begin.elapsed_time(end) / iterations


def run_batch(batch: int, slots: int, args) -> dict:
    from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config

    torch._dynamo.reset()
    original = make_inputs(batch, slots, args.seed + batch)
    functions = {
        name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
        for name, fn in (("reference", reference), ("prepared", prepared))
    }
    states = {name: (*original[:3], original[3].clone(), *original[4:]) for name in functions}
    checks = []
    with mock.patch.object(hpu_config, "use_eager_fallback", False):
        for checkpoint in range(1, 9):
            original[-2].fill_(checkpoint)
            output = {name: fn(*states[name]) for name, fn in functions.items()}
            torch.hpu.synchronize()
            expected, actual = output["reference"].cpu(), output["prepared"].cpu()
            expected_state, actual_state = states["reference"][3].cpu(), states["prepared"][3].cpu()
            checks.append({
                "accepted_checkpoint": checkpoint,
                "output_exact": torch.equal(actual, expected),
                "state_exact": torch.equal(actual_state, expected_state),
                "output_max_abs": (actual.float() - expected.float()).abs().max().item(),
                "state_max_abs": (actual_state - expected_state).abs().max().item(),
            })
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-5)
            torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-6)
        for name, fn in functions.items():
            for _ in range(args.warmups):
                fn(*states[name])
        torch.hpu.synchronize()
        samples = {name: [] for name in functions}
        names = list(functions)
        for wave in range(args.waves):
            for name in names if wave % 2 == 0 else names[::-1]:
                samples[name].append(measure(functions[name], states[name], args.iterations))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "state_pool_slots": slots,
        "checkpoint_checks": checks,
        "samples_device_ms": samples,
        "median_device_ms": medians,
        "device_speedup": medians["reference"] / medians["prepared"],
        "backend_eager_fallback_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,2")
    parser.add_argument("--pool-slots", type=int)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--waves", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if not batches or any(batch < 1 or batch > 16 for batch in batches):
        parser.error("batch sizes must be between 1 and 16")
    if min(args.warmups, args.waves, args.iterations) < 1:
        parser.error("warmups, waves and iterations must be positive")
    if args.pool_slots is not None and args.pool_slots < max(batches) * 8 + 2:
        parser.error("pool-slots must hold all checkpoints and two guard rows")
    if public_mtp_prepared_op() is None:
        raise RuntimeError("Rebuild native libraries with tools/build_flashinfer_gaudi.py first")
    report = {"torch_version": torch.__version__, "batches": {}}
    for batch in batches:
        report["batches"][str(batch)] = run_batch(batch, args.pool_slots or batch * 8 + 2, args)
        if args.output is not None:
            args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
