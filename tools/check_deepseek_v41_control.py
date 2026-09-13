# SPDX-License-Identifier: Apache-2.0
"""Check a trace-motivated FP32 control GEMV; no model performance claim."""

import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    op = torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2
    run = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(torch.nn.functional.linear, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(41)
    cases = []
    weights = []
    for pp, layer in ((0, 14), (1, 20)):
        shard = PreparedV41Shard(args.prepared, pp, 0)
        for field in ("hc_attn_fn", "hc_ffn_fn"):
            name = f"layers.{layer}.{field}"
            weights.append((name, shard.tensor(name, "cpu")))
    weights += [("random", torch.randn(24, 20480) * 0.001)]
    weight = weights[-1][1]
    fixtures = [(name, torch.randn(1, 20480).bfloat16().float(), value) for name, value in weights]
    fixtures.append(("zero", torch.zeros(1, 20480), weight))
    last = torch.zeros(1, 20480)
    last[0, -1] = 1
    fixtures.append(("last-K-element", last, weight))
    for name, activation, weight in fixtures:
        a, w = activation.to("hpu"), weight.to("hpu")
        expected = reference(a, w).cpu()
        actual, repeated = run(a, w).cpu(), run(a, w).cpu()
        ordinary = op(a, w).cpu()
        exact64 = activation.double() @ weight.double().T
        bound = 3e-6 * (activation.double().abs() @ weight.double().abs().T).clamp_min(1e-12)
        error = (actual.double() - exact64).abs()
        torch.save({"activation": activation, "weight": weight, "mme": expected,
                    "actual": actual, "fp64": exact64}, evidence / f"{name}.pt")
        assert torch.isfinite(actual).all() and torch.all(error <= bound), (name, error.max(), bound.max())
        assert torch.equal(actual.view(torch.int32), repeated.view(torch.int32))
        assert torch.equal(actual.view(torch.int32), ordinary.view(torch.int32))
        cases.append({"name": name, "max_abs_vs_fp64": error.max().item(),
                      "max_abs_vs_mme": (actual - expected).abs().max().item(),
                      "mme_bitwise_equal": torch.equal(actual.view(torch.int32), expected.view(torch.int32)),
                      "ordinary_compiled_replay_bitwise_equal": True})
    # Only the changed candidate is profiled. The saved full-model trace is
    # the MME timing reference; these calls do not establish end-to-end gain.
    a = fixtures[0][1].to("hpu")
    w = fixtures[0][2].to("hpu")
    for _ in range(4):
        run(a, w)
    torch.hpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.HPU],
                                record_shapes=True, with_stack=False) as profiler:
        for _ in range(32):
            run(a, w)
        torch.hpu.synchronize()
    profiler.export_chrome_trace(str(evidence / "control.trace.json.gz"))
    (evidence / "result.json").write_text(json.dumps({
        "status": "component_pass", "cases": cases, "candidate_profile_calls": 32,
        "precision": "FP32 TPC FMA and lane reduction; differs from MME reduction order",
        "numerical_bound": "absolute error <= 3e-6 times sum(abs(x*w)); component check only",
        "quality_qualified": False, "model_performance_claim": False,
        "peak_device_bytes": torch.hpu.max_memory_allocated(),
    }, indent=2) + "\n")
    print(json.dumps(cases), flush=True)


if __name__ == "__main__":
    main()
