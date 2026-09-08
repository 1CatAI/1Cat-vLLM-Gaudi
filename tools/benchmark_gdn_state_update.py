# SPDX-License-Identifier: Apache-2.0
"""Qualify the changed state epilogue, including normal compiled recurrence.

Only candidate timings are collected. The ordinary HPU arithmetic is used
as a correctness oracle for the new, continuously changing state cases.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import habana_frameworks.torch  # noqa: F401
import torch

from flashinfer_gaudi._reference import _direct_qwen38_tp2_single_token_packed_decode
from vllm_gaudi.ops.gdn_state_update import register_gdn_state_update_pass, state_update_lowering_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("epilogue", "recurrence"), default="epilogue")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("tp2_fused_ar_norm_bridge", args.bridge)
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    register_gdn_state_update_pass(args.output.parent / "graphs")
    rng = torch.Generator().manual_seed(817)

    def rand(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=rng) * .1).to(device="hpu", dtype=dtype)

    bank = rand((5, 24, 128, 128))
    expected_bank = bank.clone()
    initial = bank.cpu()
    state, expected_state = bank[1:2], expected_bank[1:2]
    if args.stage == "epilogue":
        delta, key = rand((1, 8, 3, 128, 1)), rand((1, 8, 1, 1, 128))

        def candidate(state, delta, key):
            decayed = state.reshape(1, 8, 3, 128, 128) * .95
            value = torch.ops.custom_op.gdn_state_update(decayed, delta, key, state)
            output = value.sum(-1)
            state.copy_(value)
            return output

        def reference(state, delta, key):
            value = torch.addcmul(state.reshape(1, 8, 3, 128, 128) * .95, delta, key).reshape_as(state)
            output = value.sum(-1)
            state.copy_(value)
            return output

        inputs = [delta, key]
    else:
        packed = rand((1, 5120), torch.bfloat16)
        log_decay, beta = -rand((1, 24)).abs(), rand((1, 24), torch.bfloat16)

        def candidate(state, packed, log_decay, beta):
            return _direct_qwen38_tp2_single_token_packed_decode(packed, log_decay, beta, state, 128**-.5, True, True,
                                                                 True)[0]

        def reference(state, packed, log_decay, beta):
            return _direct_qwen38_tp2_single_token_packed_decode(packed, log_decay, beta, state, 128**-.5, True, True,
                                                                 False)[0]

        inputs = [packed, log_decay, beta]
    compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    oracle = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    with torch.no_grad():
        for step in range(12):
            inputs[0].copy_(rand(inputs[0].shape, inputs[0].dtype) * (0, .01, .1, 1)[step % 4])
            expected = oracle(expected_state, *inputs)
            actual = compiled(state, *inputs)
            torch.hpu.synchronize()
            a, b = actual.cpu(), expected.cpu()
            x, y = state.cpu(), expected_state.cpu()
            records.append({
                "step": step,
                "output_exact": torch.equal(a, b),
                "state_exact": torch.equal(x, y),
                "output_max_abs": (a.float() - b.float()).abs().max().item(),
                "state_max_abs": (x - y).abs().max().item()
            })
            args.output.write_text(
                json.dumps({
                    "stage": args.stage,
                    "checks": records,
                    "lowering": state_update_lowering_stats()
                },
                           indent=2))
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            torch.testing.assert_close(bank.cpu()[[0, 2, 3, 4]], initial[[0, 2, 3, 4]], rtol=0, atol=0)
        assert state_update_lowering_stats()["updates"] == 1
        samples = []
        for _ in range(15):
            torch.hpu.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                compiled(state, *inputs)
            torch.hpu.synchronize()
            samples.append((time.perf_counter() - started) * 1000 / 20)
    result = {
        "stage": args.stage,
        "checks": records,
        "lowering": state_update_lowering_stats(),
        "candidate_device_drained_ms": samples,
        "candidate_median_ms": statistics.median(samples)
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
