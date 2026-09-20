# SPDX-License-Identifier: Apache-2.0
"""Compare the three production ratio-2 Full index selections.

The benchmark keeps the 1M paged-cache contract and includes the decoded-hot
row update in the candidate.  It stops after the ordered row IDs consumed by
MLA are materialized, so it measures the complete changed selection chain.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401
import torch

from vllm_gaudi.ops.deepseek_v41_indexer import INDEX_MME_HOT_TOKENS, runtime_index_select
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--visible", type=int, default=2052)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--waves", type=int, default=9)
    parser.add_argument("--wave-replays", type=int, default=32)
    args = parser.parse_args()
    ratio, layers = 2, 3
    hot_rows = INDEX_MME_HOT_TOKENS // ratio
    capacity = 1_048_576 // ratio
    if not 1 <= args.visible <= INDEX_MME_HOT_TOKENS:
        raise ValueError("visible tokens must fit the decoded hot bucket")

    torch.ops.load_library(__import__("os").environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    generator = torch.Generator(device="cpu").manual_seed(20260919)
    decoded_cpu = []
    packed_cpu = []
    for _ in range(layers):
        decoded = unpack_fp4(
            pack_fp4(torch.randn(hot_rows, 128, generator=generator).bfloat16(), 32), 128, 32)
        packed = torch.zeros(capacity, 68, dtype=torch.uint8)
        packed[:hot_rows] = pack_fp4(decoded, 32)
        decoded_cpu.append(decoded)
        packed_cpu.append(packed)
    query_cpu = unpack_fp4(
        pack_fp4(torch.randn(layers, 32, 128, generator=generator).bfloat16(), 32), 128, 32,
    ).reshape(layers, 1, 32, 128)
    weights_cpu = (torch.randn(layers, 1, 32, generator=generator) * .02).bfloat16()

    query, weights = query_cpu.to("hpu"), weights_cpu.to("hpu")
    packed = tuple(value.to("hpu") for value in packed_cpu)
    decoded = tuple(value.to("hpu") for value in decoded_cpu)
    pages = torch.arange(capacity // (128 // ratio), dtype=torch.int32, device="hpu")
    position = torch.tensor([args.visible - 1], dtype=torch.int32, device="hpu")
    candidates = torch.full((1, 2048), -1, dtype=torch.int32, device="hpu")
    write_row = torch.tensor([(args.visible - 1) // ratio], dtype=torch.int32, device="hpu")
    new_keys = tuple(value[(args.visible - 1) // ratio:(args.visible - 1) // ratio + 1].to("hpu")
                     for value in decoded_cpu)

    def make_chain(use_hot: bool):
        def chain(query_, weights_, packed0, packed1, packed2, hot0, hot1, hot2,
                  key0, key1, key2, pages_, position_, candidates_, write_row_):
            packed_inputs = (packed0, packed1, packed2)
            hot_inputs = (hot0, hot1, hot2)
            key_inputs = (key0, key1, key2)
            outputs = []
            for layer in range(layers):
                if use_hot:
                    hot_inputs[layer].index_copy_(0, write_row_.long(), key_inputs[layer])
                selected, _ = runtime_index_select(
                    query_[layer], weights_[layer], packed_inputs[layer], pages_, position_, candidates_,
                    ratio=ratio, capacity=capacity, reindex=False, publish_candidates=False,
                    decoded_hot=hot_inputs[layer] if use_hot else None)
                outputs.append(selected)
            return torch.cat(outputs, 0)
        return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)

    call_args = (query, weights, *packed, *decoded, *new_keys, pages, position, candidates, write_row)
    native, hot = make_chain(False), make_chain(True)

    def measure(function):
        for _ in range(args.warmups):
            output = function(*call_args)
        torch.hpu.synchronize()
        samples = []
        for _ in range(args.waves):
            begin = torch.hpu.Event(enable_timing=True)
            end = torch.hpu.Event(enable_timing=True)
            begin.record()
            for _ in range(args.wave_replays):
                output = function(*call_args)
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / args.wave_replays)
        return output.cpu(), {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        }

    native_output, native_time = measure(native)
    hot_output, hot_time = measure(hot)
    result = {
        "chain": "three ratio-2 Full selections through ordered MLA row IDs",
        "visible_tokens": args.visible,
        "capacity_tokens": 1_048_576,
        "native_packed": native_time,
        "decoded_hot_mme": hot_time,
        "selected_equal": bool(torch.equal(native_output, hot_output)),
        "selected_diff": int((native_output != hot_output).sum()),
        "saved_ms": native_time["mean_ms"] - hot_time["mean_ms"],
        "speedup": native_time["mean_ms"] / hot_time["mean_ms"],
        "end_to_end_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
