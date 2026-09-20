# SPDX-License-Identifier: Apache-2.0
"""Compare 1M-capacity and hot-bucket ratio-2 index selection graphs.

The three chains model V4.1 PP0 Full index owners (layers 2, 8 and 14).  Both
arms read the same paged FP4 index cache and current positions.  The bounded
arm changes only the statically declared score/output extent of the already
separate hot recipe; long-context execution retains the capacity graph.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401,E402
import torch


RATIO = 2
LAYERS = 3
PAGE_ROWS = 128 // RATIO
FULL_CAPACITY = 1_048_576 // RATIO


def build(capacity: int):
    def chain(queries, weights, caches, pages, positions, candidates):
        selected = []
        for layer in range(LAYERS):
            scores, _ = torch.ops.custom_op.custom_deepseek_v41_index_scores_gaudi2(
                queries[layer], weights[layer], caches[layer], pages, positions,
                candidates, RATIO, 0, capacity)
            stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
                scores, positions, RATIO, 0, 0)
            selected.append(
                torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
                    scores, positions, candidates, stats, RATIO, 0, 0))
        return torch.cat(selected, 0)

    return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)


def measure(fn, inputs, warmups: int, repeats: int):
    for _ in range(warmups):
        fn(*inputs)
    torch.hpu.synchronize()
    samples, output = [], None
    for _ in range(repeats):
        begin = torch.hpu.Event(enable_timing=True)
        end = torch.hpu.Event(enable_timing=True)
        begin.record()
        output = fn(*inputs)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    ordered = sorted(samples)
    return output.cpu(), {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[int(.9 * len(ordered))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--visible", type=int, default=2052)
    parser.add_argument("--hot-capacity", type=int, default=1280)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=128)
    args = parser.parse_args()
    visible_rows = (args.visible + RATIO - 1) // RATIO
    if args.hot_capacity % 64 or not 512 <= visible_rows <= args.hot_capacity:
        raise ValueError("hot capacity must cover the aligned compressed prefix")

    import os
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    generator = torch.Generator(device="cpu").manual_seed(20260920)
    queries = torch.randn((LAYERS, 1, 32, 128), generator=generator).bfloat16()
    weights = (torch.randn((LAYERS, 1, 32), generator=generator) * .02).bfloat16()
    # Include the reserved null page plus every physical context page.
    cache_rows = (FULL_CAPACITY // PAGE_ROWS + 1) * PAGE_ROWS
    caches = torch.randint(0, 256, (LAYERS, cache_rows, 68),
                           generator=generator, dtype=torch.uint8)
    pages = torch.arange(1, FULL_CAPACITY // PAGE_ROWS + 1, dtype=torch.int32)
    positions = torch.tensor([args.visible - 1], dtype=torch.int32)
    candidates = torch.full((1, 2048), -1, dtype=torch.int32)
    inputs = tuple(value.to("hpu") for value in
                   (queries, weights, caches, pages, positions, candidates))

    full_output, full = measure(build(FULL_CAPACITY), inputs,
                                args.warmups, args.repeats)
    hot_output, hot = measure(build(args.hot_capacity), inputs,
                              args.warmups, args.repeats)
    payload = {
        "benchmark": "dsv41_ratio2_three_layer_hot_capacity",
        "visible_tokens": args.visible,
        "visible_compressed_rows": visible_rows,
        "full_capacity": FULL_CAPACITY,
        "hot_capacity": args.hot_capacity,
        "full_capacity_chain": full,
        "hot_capacity_chain": hot,
        "selected_bitwise_equal": bool(torch.equal(full_output, hot_output)),
        "selected_differences": int((full_output != hot_output).sum()),
        "mean_saved_ms": full["mean_ms"] - hot["mean_ms"],
        "median_saved_ms": full["median_ms"] - hot["median_ms"],
        "e2e_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
