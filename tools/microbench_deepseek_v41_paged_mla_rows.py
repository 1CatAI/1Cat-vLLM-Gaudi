# SPDX-License-Identifier: Apache-2.0
"""Measure the production paged MLA MME cost as selected-row width changes.

This is deliberately a component benchmark: it uses the same Gaudi custom op,
packed SWA/main dtypes and index geometry as C1 decode, but does not claim an
end-to-end gain.  It is useful for deciding whether the 2K gap is dominated by
selected-row materialization/MME work before changing the serving graph.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401,E402
import torch


def _sync() -> None:
    torch.hpu.synchronize()


def _measure(rows: int, repeats: int, warmups: int) -> dict:
    # C1 production shapes: q=[tokens, local_heads, 512], SWA=[256,528] U8,
    # packed main=[physical rows,288] U8, row ids=[1,rows], attention ids
    # [1,512], and one length.  Keep the main allocation fixed so only the
    # selected SRAM decode/MME extent changes between cases.
    q = torch.randn((1, 32, 512), dtype=torch.bfloat16, device="hpu")
    swa = torch.zeros((256, 528), dtype=torch.uint8, device="hpu")
    main = torch.zeros((4096, 288), dtype=torch.uint8, device="hpu")
    row_ids = torch.arange(rows, dtype=torch.int32, device="hpu").reshape(1, rows)
    indices = (torch.arange(512, dtype=torch.int32, device="hpu") % rows).reshape(1, 512)
    sink = torch.zeros((32,), dtype=torch.float32, device="hpu")
    scale = torch.tensor([512.0 ** -0.5], dtype=torch.float32, device="hpu")
    lengths = torch.tensor([512], dtype=torch.int32, device="hpu")

    def invoke(q, swa, main, row_ids, indices, sink, scale, lengths):
        return torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2(
            q, swa, main, row_ids, indices, sink, scale, lengths
        )

    # The paged operator allocates its selected-KV workspace in the compiled
    # recipe.  Calling it eagerly cannot provide that recipe-owned SRAM and
    # would measure an invalid path, so compile exactly one fixed-row graph per
    # case, matching serving's static graph contract.
    compiled = torch.compile(invoke, backend="hpu_backend", fullgraph=True, dynamic=False)

    # Force the same compiled custom-op recipe to be materialized before timing.
    for _ in range(warmups):
        compiled(q, swa, main, row_ids, indices, sink, scale, lengths)
    _sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        compiled(q, swa, main, row_ids, indices, sink, scale, lengths)
        _sync()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return {
        "rows": rows,
        "repeats": repeats,
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": samples[len(samples) // 2],
        "p90_ms": samples[min(len(samples) - 1, int(len(samples) * 0.9))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 2048, 2560])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if not hasattr(torch, "hpu"):
        raise RuntimeError("HPU torch is required")
    lib = os.environ.get("VLLM_HPU_DSV4_TPC_OP_LIBRARY")
    if not lib:
        raise RuntimeError("VLLM_HPU_DSV4_TPC_OP_LIBRARY must point to the built extension")
    torch.ops.load_library(lib)
    if any(r <= 0 or r > 4096 or r % 64 for r in args.rows):
        raise ValueError("rows must be positive multiples of 64 and <=4096")
    results = [_measure(r, args.repeats, args.warmups) for r in args.rows]
    payload = {
        "benchmark": "dsv41_paged_mla_mme_selected_rows",
        "dtype": {"q": "bf16", "swa": "u8", "main": "u8", "ids": "i32", "accumulator": "f32"},
        "shapes": {"q": [1, 32, 512], "swa": [256, 528], "main": [4096, 288], "indices": [1, 512]},
        "results": results,
        "e2e_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    _sync()


if __name__ == "__main__":
    main()
