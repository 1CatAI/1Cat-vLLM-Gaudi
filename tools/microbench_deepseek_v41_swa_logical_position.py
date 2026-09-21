# SPDX-License-Identifier: Apache-2.0
"""Measure moving the decoded SWA ring mapping into its existing writer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from benchmark_deepseek_v41_projection_chains import summarize
from deepseek_v41_micro_replay import RecipeRecorder
import torch

from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers


def reference(value, packed, decoded, position):
    ring = position.remainder(256).to(torch.int32).contiguous()
    return torch.ops.custom_op.custom_deepseek_v41_swa_paged_decoded_write_bf16_gaudi2(
        packed, value, ring, ring, decoded, 0)


def candidate(value, packed, decoded, position):
    logical = position.to(torch.int32).contiguous()
    return torch.ops.custom_op.custom_deepseek_v41_swa_paged_decoded_write_bf16_gaudi2(
        packed, value, logical, logical, decoded, 0)


def main() -> None:
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(20260921)
    layers = 20
    values = [torch.randn(1, 512, dtype=torch.bfloat16, device="hpu")
              for _ in range(layers)]
    # Production runner owns an int32 persistent positions buffer.  Keep the
    # microbenchmark on that exact contract so the measured delta is only the
    # redundant ring modulo, not an artificial int64-to-int32 cast.
    position = [torch.tensor([511], dtype=torch.int32, device="hpu")
                for _ in range(layers)]
    # Register recipe observation before either arm is compiled.
    recorder = RecipeRecorder(output)

    def states():
        return [(torch.zeros(256, 528, dtype=torch.uint8, device="hpu"),
                 torch.zeros(512, 512, dtype=torch.bfloat16, device="hpu"),
                 p) for p in position]

    weights = {"reference": states(), "candidate": states()}
    functions = {
        "reference": torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False),
        "candidate": torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False),
    }
    for arm, fn in functions.items():
        for _ in range(3):
            for value, row in zip(values, weights[arm], strict=True):
                fn(value, *row)
        torch.hpu.synchronize()

    # Both paths must publish exactly the same packed bytes, decoded BF16 row,
    # and completion value across the first circular wrap.
    for rows in weights.values():
        for packed, decoded, _ in rows:
            packed.zero_()
            decoded.zero_()
    completions = {arm: [fn(value, *row) for value, row in zip(values, weights[arm], strict=True)]
                   for arm, fn in functions.items()}
    torch.hpu.synchronize()
    mismatches = {"packed": 0, "decoded": 0, "completion": 0}
    for index in range(layers):
        rp, rd, _ = weights["reference"][index]
        cp, cd, _ = weights["candidate"][index]
        mismatches["packed"] += int((rp.cpu() != cp.cpu()).sum())
        mismatches["decoded"] += int((rd.cpu().view(torch.int16) != cd.cpu().view(torch.int16)).sum())
        mismatches["completion"] += int((completions["reference"][index].cpu()
                                          != completions["candidate"][index].cpu()).sum())
    if any(mismatches.values()):
        raise AssertionError(("logical SWA writer changed state", mismatches))

    replays = {arm: recorder.prepare(fn, values, weights[arm]) for arm, fn in functions.items()}
    for replay in replays.values():
        for _ in range(32):
            replay()
        torch.hpu.synchronize()
    bind_worker_helpers(0)
    events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
              for _ in range(32)]
    rounds = []
    for round_id in range(3):
        order = ("reference", "candidate") if round_id % 2 == 0 else ("candidate", "reference")
        for arm in order:
            host, device = [], []
            for begin, end in events:
                start = time.perf_counter_ns()
                begin.record()
                replays[arm]()
                end.record()
                end.synchronize()
                host.append((time.perf_counter_ns() - start) / 1e6)
                device.append(begin.elapsed_time(end))
            rounds.append({"round": round_id, "arm": arm,
                           "device_sweep": summarize(device),
                           "synchronized_host_sweep": summarize(host)})
            print(arm, round_id, sum(device) / len(device), flush=True)
    result = {
        "name": "dsv41-swa-logical-position",
        "scope": "20 PP-stage decoded SWA writers through first true consumer completion",
        "layers_per_sweep": layers,
        "position": 511,
        "bitwise_mismatches": mismatches,
        "rounds": rounds,
        "native_compute_info": {arm: replay.native.info() for arm, replay in replays.items()},
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for replay in replays.values():
        replay.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
