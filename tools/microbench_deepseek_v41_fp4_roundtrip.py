# SPDX-License-Identifier: Apache-2.0
"""Validate and time the fused V4.1 group-32 FP4 roundtrip on one HPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core as htcore  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_math import (  # noqa: E402
    _pack_fp4_torch,
    unpack_fp4,
)


def reference(value: torch.Tensor) -> torch.Tensor:
    return unpack_fp4(_pack_fp4_torch(value, 32), value.shape[-1], 32)


def source(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = (torch.randn(rows, 128, generator=generator) * 3).bfloat16()
    value[0, :32] = 0
    if rows > 1:
        value[1, :32] = -0.0
    if rows > 2:
        # Exact FP4 decision boundaries under a unit UE8M0 scale.
        value[2, :8] = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0],
            dtype=torch.bfloat16,
        )
        value[2, 8:16] = -value[2, :8]
    return value


def synchronize() -> None:
    htcore.mark_step()
    torch.hpu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()

    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    op = torch.ops.custom_op.custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2
    rows_by_tokens = {1: 16, 128: 2048, 512: 8192}
    results: dict[str, object] = {"cases": []}

    for tokens, rows in rows_by_tokens.items():
        host = source(rows, 41 + tokens)
        expected = reference(host)
        device = host.to("hpu")
        actual = op(device)
        synchronize()
        actual_host = actual.cpu()
        mismatch = int((actual_host.view(torch.int16) != expected.view(torch.int16)).sum())
        if mismatch:
            where = (actual_host.view(torch.int16) != expected.view(torch.int16)).nonzero()
            first = where[:16]
            diagnostic = [{
                "row": int(row),
                "column": int(column),
                "input": float(host[row, column]),
                "actual": float(actual_host[row, column]),
                "expected": float(expected[row, column]),
                "actual_bits": int(actual_host[row, column].view(torch.int16)),
                "expected_bits": int(expected[row, column].view(torch.int16)),
            } for row, column in first]
            print(json.dumps({"case": f"C{tokens}", "mismatches": mismatch,
                              "first": diagnostic}, indent=2))
            raise AssertionError(f"C{tokens}: {mismatch} BF16 encodings differ")

        # Warm the exact shape, then measure the complete submitted operation.
        for _ in range(5):
            actual = op(device)
        synchronize()
        samples = []
        for _ in range(args.iterations):
            begin = time.perf_counter_ns()
            actual = op(device)
            synchronize()
            samples.append((time.perf_counter_ns() - begin) / 1e6)
        results["cases"].append({
            "tokens": tokens,
            "rows": rows,
            "width": 128,
            "mismatches": mismatch,
            "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "iterations": len(samples),
        })

    # Reproduce the production C2048 query boundary: four fixed C512 native
    # calls in one compiled graph, with no generic unpack node.
    host = source(2048 * 16, 2048).reshape(2048, 16, 128)
    expected = reference(host.reshape(-1, 128)).reshape_as(host)

    def c2048(value: torch.Tensor) -> torch.Tensor:
        parts = []
        for start in range(0, 2048, 512):
            part = value[start:start + 512].reshape(-1, 128).contiguous()
            parts.append(op(part).reshape(512, 16, 128))
        return torch.cat(parts, 0)

    compiled = torch.compile(c2048, backend="hpu_backend", fullgraph=True, dynamic=False)
    device = host.to("hpu")
    first = compiled(device)
    synchronize()
    first_host = first.cpu()
    mismatch = int((first_host.view(torch.int16) != expected.view(torch.int16)).sum())
    if mismatch:
        raise AssertionError(f"compiled C2048: {mismatch} BF16 encodings differ")
    changed = (host + torch.tensor(0.5, dtype=torch.bfloat16)).to("hpu")
    changed_actual = compiled(changed)
    synchronize()
    changed_expected = reference(changed.cpu().reshape(-1, 128)).reshape_as(host)
    changed_host = changed_actual.cpu()
    changed_mismatch = int(
        (changed_host.view(torch.int16) != changed_expected.view(torch.int16)).sum()
    )
    if changed_mismatch:
        raise AssertionError(
            f"compiled C2048 changed input: {changed_mismatch} BF16 encodings differ"
        )
    results["compiled_c2048"] = {
        "mismatches": mismatch,
        "changed_input_mismatches": changed_mismatch,
        "shape": list(host.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
