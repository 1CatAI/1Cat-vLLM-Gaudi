# SPDX-License-Identifier: Apache-2.0
"""Measure the production ratio-2 compressor history/pair-reduction chain.

The PP0 V4.1 stage owns three ratio-2 compressor source layers.  The
reference reproduces their C1 history update, two-row gather, softmax and
BF16 reduction.  The candidate executes the same state transition in one
TPC kernel per layer.  Projection and the downstream FP4 writer are outside
the boundary because neither changes in this candidate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from benchmark_deepseek_v41_projection_chains import summarize
from deepseek_v41_micro_replay import RecipeRecorder
import torch

from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers


LAYERS = 3
HISTORY_ROWS = 8


def reference(kv_history, score_history, kv, score, position):
    first = torch.bitwise_and(position, -2)
    ring = torch.bitwise_and(position, HISTORY_ROWS - 1).long()
    a = torch.bitwise_and(first, HISTORY_ROWS - 1).long()
    b = torch.bitwise_and(first + 1, HISTORY_ROWS - 1).long()
    kv_history.index_copy_(0, ring, kv)
    score_history.index_copy_(0, ring, score)
    gates = torch.stack((score_history[a], score_history[b]), 1).softmax(1)
    return (kv_history[a] * gates[:, 0] +
            kv_history[b] * gates[:, 1]).to(torch.bfloat16)


def candidate(kv_history, score_history, kv, score, position):
    return torch.ops.custom_op.custom_deepseek_v41_compressor_pair_bf16_gaudi2(
        kv_history, score_history, kv, score, position)


def chain(fn):
    def run(kv, score, kv_history, score_history, position):
        return tuple(fn(kh, sh, x, s, position)
                     for x, s, kh, sh in zip(
                         kv, score, kv_history, score_history, strict=True))
    return run


def clone_state(state):
    return tuple(tuple(value.clone() for value in row) for row in state)


def mismatches(left, right):
    result = {"latent_bf16_bits": 0, "kv_history": 0, "score_history": 0}
    for a, b in zip(left["outputs"], right["outputs"], strict=True):
        result["latent_bf16_bits"] += int(
            (a.view(torch.int16).cpu() != b.view(torch.int16).cpu()).sum())
    for name, index in (("kv_history", 0), ("score_history", 1)):
        for a, b in zip(left["state"], right["state"], strict=True):
            result[name] += int((a[index].cpu() != b[index].cpu()).sum())
    return result


def main() -> None:
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(20260921)
    recorder = RecipeRecorder(output)

    functions = {
        "reference": torch.compile(chain(reference), backend="hpu_backend",
                                   fullgraph=True, dynamic=False),
        "candidate": torch.compile(chain(candidate), backend="hpu_backend",
                                   fullgraph=True, dynamic=False),
    }
    kv = tuple(torch.randn(1, 512, dtype=torch.float32, device="hpu")
               for _ in range(LAYERS))
    score = tuple(torch.randn(1, 512, dtype=torch.float32, device="hpu")
                  for _ in range(LAYERS))
    base = tuple((torch.randn(HISTORY_ROWS, 512, dtype=torch.float32,
                              device="hpu"),
                  torch.randn(HISTORY_ROWS, 512, dtype=torch.float32,
                              device="hpu")) for _ in range(LAYERS))

    # Cover both parity paths with changing values before timing the odd path
    # represented by the archived position-511 trace.
    parity_checks = []
    timed_state = {}
    for logical in (510, 511):
        arms = {}
        for arm, fn in functions.items():
            state = clone_state(base)
            position = torch.tensor([logical], dtype=torch.int32, device="hpu")
            values = tuple(x + (logical - 509) * 0.03125 for x in kv)
            scores = tuple(x - (logical - 509) * 0.015625 for x in score)
            outputs = fn(values, scores,
                         tuple(row[0] for row in state),
                         tuple(row[1] for row in state), position)
            arms[arm] = {"outputs": outputs, "state": state}
            if logical == 511:
                timed_state[arm] = (values, scores, state, position)
        torch.hpu.synchronize()
        diff = mismatches(arms["reference"], arms["candidate"])
        parity_checks.append({"position": logical, "mismatches": diff})
        if any(diff.values()):
            raise AssertionError(("compressor pair changed state/output", logical, diff))

    replays = {}
    for arm, fn in functions.items():
        values, scores, state, position = timed_state[arm]
        replays[arm] = recorder.prepare(
            fn, [values],
            [(scores, tuple(row[0] for row in state),
              tuple(row[1] for row in state), position)])

    for replay in replays.values():
        for _ in range(32):
            replay()
        torch.hpu.synchronize()
    bind_worker_helpers(0)
    events = [(torch.hpu.Event(enable_timing=True),
               torch.hpu.Event(enable_timing=True)) for _ in range(32)]
    rounds = []
    for round_id in range(3):
        order = ("reference", "candidate") if round_id % 2 == 0 else (
            "candidate", "reference")
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
        "name": "dsv41-compressor-pair",
        "scope": "three PP0 ratio-2 history updates plus pair softmax/reduction through BF16 latent",
        "positions_checked": parity_checks,
        "timed_position": 511,
        "layers": LAYERS,
        "rounds": rounds,
        "native_compute_info": {arm: replay.native.info()
                                for arm, replay in replays.items()},
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for replay in replays.values():
        replay.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
