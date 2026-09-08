# SPDX-License-Identifier: Apache-2.0
"""Candidate-only timing after continuous TP2 shared-state correctness checks."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]

import habana_frameworks.torch  # noqa: E402,F401
import torch  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from flashinfer_gaudi._reference import qwen38_fused_decode_step_direct  # noqa: E402
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime  # noqa: E402
from vllm_gaudi.ops.gdn_state_update import state_update_lowering_stats, validate_direct_state_views  # noqa: E402


class Group(torch.nn.Module):

    def __init__(self, states, convs, tensors, native):
        super().__init__()
        self.states, self.convs, self.tensors, self.native = states, convs, tensors, native

    def forward(self, hidden, residual):
        a, b, a_log, bias, conv_weight, projection, norm_weight = self.tensors
        for state, conv in zip(self.states, self.convs, strict=True):
            output, _, _ = qwen38_fused_decode_step_direct(hidden,
                                                           a,
                                                           b,
                                                           a_log,
                                                           bias,
                                                           conv,
                                                           conv_weight,
                                                           None,
                                                           state,
                                                           128**-.5,
                                                           direct_state_update=self.native)
            partial = output.reshape(1, 3072) @ projection
            hidden, residual = torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm(partial, residual, norm_weight,
                                                                                    1e-6)
        return hidden, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reuse-correctness", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cfg = VllmConfig()
    init_distributed_environment(2, rank, "env://", rank, backend="hccl")
    with set_current_vllm_config(cfg):
        initialize_model_parallel(2, 1)
    initialize_tp2_fused_ar_norm_runtime()
    bridge = _resolve_runtime()[0]
    rng = torch.Generator().manual_seed(351)

    def rand(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=rng) * .03).to(device="hpu", dtype=dtype)

    tensors = (rand((1, 24)), rand((1, 24)), rand((24, ), torch.float32) - 2, rand((24, )), rand(
        (5120, 4)), rand((3072, 5120)), torch.ones(5120, dtype=torch.bfloat16, device="hpu"))
    pools = [rand((18, 24, 128, 128), torch.float32) for _ in range(2)]
    native_pools = [x.clone() for x in pools]
    initial = [x.cpu() for x in pools]
    convs = [rand((1, 3, 5120)) for _ in range(18)]
    native_convs = [x.clone() for x in convs]
    groups = [[], []]
    for arm, (banks, cs) in enumerate(((pools, convs), (native_pools, native_convs))):
        views = [banks[i % 2][1 + i // 2:2 + i // 2] for i in range(18)]
        assert validate_direct_state_views(views) == 18
        for group in range(3):
            begin = 6 * group
            module = Group(views[begin:begin + 6], cs[begin:begin + 6], tensors, bool(arm))
            groups[arm].append(torch.compile(module, backend="hpu_backend", fullgraph=True, dynamic=False))
    hidden = rand((1, 5120))
    residual = torch.zeros_like(hidden)

    def token(arm):
        h, r = hidden, residual
        for group in groups[arm]:
            h, r = group(h, r)
        return h

    checks = []
    result_path = args.output_directory / f"rank{rank}.json"
    if args.reuse_correctness:
        saved = json.loads((args.reuse_correctness / f"rank{rank}.json").read_text())
        assert len(saved["checks"]) == 12 and all(x["state_output_conv_exact"] for x in saved["checks"])
        checks = saved["checks"]
    with torch.no_grad():
        if args.compile_only:
            token(1)
            torch.hpu.synchronize()
            result_path.write_text(
                json.dumps({
                    "compile_only": True,
                    "lowering": state_update_lowering_stats()
                }, indent=2))
            return
        for step in range(0 if args.reuse_correctness else 12):
            hidden.copy_(rand((1, 5120)) * (0, .01, .1, 1)[step % 4])
            before = bridge.collective_launch_count()
            expected, actual = token(0), token(1)
            torch.hpu.synchronize()
            reductions = bridge.collective_launch_count() - before
            assert reductions == 36, reductions
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
            for bank, other, original in zip(pools, native_pools, initial, strict=True):
                torch.testing.assert_close(bank.cpu(), other.cpu(), atol=0, rtol=0)
                torch.testing.assert_close(other.cpu()[[0, *range(10, 18)]],
                                           original[[0, *range(10, 18)]],
                                           atol=0,
                                           rtol=0)
            for conv, other in zip(convs, native_convs, strict=True):
                torch.testing.assert_close(conv.cpu(), other.cpu(), atol=0, rtol=0)
            checks.append({"step": step, "state_output_conv_exact": True, "reductions": reductions})
            result_path.write_text(json.dumps({"checks": checks, "lowering": state_update_lowering_stats()}, indent=2))
        if args.reuse_correctness:
            token(1)
            torch.hpu.synchronize()
        # One compiled six-layer graph is reused across three groups.
        assert state_update_lowering_stats()["updates"] == 6
        assert tuple(bridge.gdn_state_dma_counts()) == (0, 0, 0)
        samples = []
        for _ in range(15):
            torch.hpu.synchronize()
            start = time.perf_counter()
            token(1)
            torch.hpu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    result = {
        "checks": checks,
        "lowering": state_update_lowering_stats(),
        "reused_correctness": str(args.reuse_correctness) if args.reuse_correctness else None,
        "candidate_device_drained_ms": samples,
        "candidate_median_ms": statistics.median(samples),
        "state_dma": tuple(bridge.gdn_state_dma_counts())
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
