# SPDX-License-Identifier: Apache-2.0
"""Measure power-of-two CSA2 state addressing through the real FP4 writer.

The production V4.1 layout has three ratio-2 KV source layers on PP0 and one
ratio-1 source layer on PP1.  This diagnostic executes that exact four-source
state shape, including ratio-2 history mutation and the packed/decoded FP4
writer.  Projection, norm and RoPE are intentionally outside the boundary:
they are unchanged by this candidate.
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


PAGE_TOKENS = 128
HISTORY_ROWS = 8


def program(bitwise: bool):

    def run(kv, score, kv_history, score_history, main_cache, index_cache,
            index_value, position, block_table, decoded, ratio: int):
        if ratio == 2:
            if bitwise:
                first = torch.bitwise_and(position, -2)
                ring = torch.bitwise_and(position, HISTORY_ROWS - 1).long()
                a = torch.bitwise_and(first, HISTORY_ROWS - 1).long()
                b = torch.bitwise_and(first + 1, HISTORY_ROWS - 1).long()
            else:
                first = position - position.remainder(2)
                ring = position.remainder(HISTORY_ROWS).long()
                a = first.remainder(HISTORY_ROWS).long()
                b = (first + 1).remainder(HISTORY_ROWS).long()
            kv_history.index_copy_(0, ring, kv)
            score_history.index_copy_(0, ring, score)
            gates = torch.stack((score_history[a], score_history[b]), 1).softmax(1)
            latent = (kv_history[a] * gates[:, 0] +
                      kv_history[b] * gates[:, 1]).to(torch.bfloat16)
        else:
            latent = kv.to(torch.bfloat16)

        width = PAGE_TOKENS // ratio
        if bitwise:
            ratio_shift = ratio.bit_length() - 1
            width_shift = width.bit_length() - 1
            rows = torch.bitwise_right_shift(position, ratio_shift)
            visible = (torch.bitwise_and(position, ratio - 1) == ratio - 1)
            blocks = block_table.index_select(
                0, torch.bitwise_right_shift(rows, width_shift).long())
            physical = blocks * width + torch.bitwise_and(rows, width - 1)
            slots = torch.where(visible, physical,
                                torch.bitwise_and(rows, width - 1))
        else:
            rows = position // ratio
            visible = (position + 1).remainder(ratio) == 0
            blocks = block_table.index_select(0, (rows // width).long())
            physical = blocks * width + rows.remainder(width)
            slots = torch.where(visible, physical, rows.remainder(width))
        done = torch.ops.custom_op.custom_deepseek_v41_fp4_paged_decoded_write_bf16_gaudi2(
            main_cache, index_cache, latent.contiguous(),
            index_value.contiguous(), slots.to(torch.int32).contiguous(),
            rows.to(torch.int32).contiguous(), decoded)
        return latent, done

    return run


def main() -> None:
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(20260921)
    recorder = RecipeRecorder(output)
    functions = {
        "reference": torch.compile(program(False), backend="hpu_backend",
                                   fullgraph=True, dynamic=False),
        "candidate": torch.compile(program(True), backend="hpu_backend",
                                   fullgraph=True, dynamic=False),
    }
    ratios = (2, 2, 2, 1)
    inputs = [torch.randn(1, 512, dtype=torch.float32, device="hpu")
              for _ in ratios]

    def make_weights():
        rows = []
        for ratio in ratios:
            physical = 2 * PAGE_TOKENS // ratio
            # The production C1 decoded mirror keeps at least the fixed C512
            # bucket even when ratio-2 has only 256 currently visible rows.
            decoded_rows = 512
            rows.append((
                torch.randn(1, 512, dtype=torch.float32, device="hpu"),
                torch.randn(HISTORY_ROWS, 512, dtype=torch.float32,
                            device="hpu"),
                torch.randn(HISTORY_ROWS, 512, dtype=torch.float32,
                            device="hpu"),
                torch.zeros(physical, 288, dtype=torch.uint8, device="hpu"),
                torch.zeros(physical, 68, dtype=torch.uint8, device="hpu"),
                torch.randn(1, 128, dtype=torch.bfloat16, device="hpu"),
                torch.tensor([511], dtype=torch.int32, device="hpu"),
                torch.cat((torch.ones(1, dtype=torch.int32),
                           torch.zeros(8191, dtype=torch.int32))).to("hpu"),
                torch.zeros(decoded_rows, 512, dtype=torch.bfloat16,
                            device="hpu"),
                ratio,
            ))
        return rows

    weights = {arm: make_weights() for arm in functions}
    # Make both arms start from identical state and changing, nonzero inputs.
    for layer in range(len(ratios)):
        for index in (0, 1, 2, 5, 6, 7):
            weights["candidate"][layer][index].copy_(
                weights["reference"][layer][index])

    outputs = {}
    for arm, fn in functions.items():
        outputs[arm] = [fn(x, *w) for x, w in zip(inputs, weights[arm], strict=True)]
    torch.hpu.synchronize()
    mismatch = {"latent": 0, "completion": 0, "kv_history": 0,
                "score_history": 0, "main_cache": 0, "index_cache": 0,
                "decoded": 0}
    for layer in range(len(ratios)):
        for name, index in (("kv_history", 1), ("score_history", 2),
                            ("main_cache", 3), ("index_cache", 4),
                            ("decoded", 8)):
            a, b = weights["reference"][layer][index], weights["candidate"][layer][index]
            if a.dtype == torch.bfloat16:
                a, b = a.view(torch.int16), b.view(torch.int16)
            mismatch[name] += int((a.cpu() != b.cpu()).sum())
        for name, index in (("latent", 0), ("completion", 1)):
            a, b = outputs["reference"][layer][index], outputs["candidate"][layer][index]
            if a.dtype == torch.bfloat16:
                a, b = a.view(torch.int16), b.view(torch.int16)
            mismatch[name] += int((a.cpu() != b.cpu()).sum())
    if any(mismatch.values()):
        raise AssertionError(("CSA2 bitwise addressing changed state", mismatch))

    replays = {arm: recorder.prepare(fn, inputs, weights[arm])
               for arm, fn in functions.items()}
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
        "name": "dsv41-csa2-power2-position-state",
        "scope": "three ratio-2 history/writer chains plus one ratio-1 writer through completion",
        "ratios": list(ratios),
        "position": 511,
        "bitwise_mismatches": mismatch,
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
