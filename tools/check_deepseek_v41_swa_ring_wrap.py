# SPDX-License-Identifier: Apache-2.0
"""Validate the decoded SWA mirror across the 256-row C1 ring wrap.

The deployed paged writer accepts separate packed and decoded row indices.
This check demonstrates the production contract: both destinations must use
the circular row.  It also runs the former logical-row call as a causal
control, using the same values and native kernel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401,E402
import torch

from vllm_gaudi.ops.deepseek_v41_math import unpack_swa


def run(logical_decoded_row: bool) -> dict:
    packed = torch.zeros(256, 528, dtype=torch.uint8, device="hpu")
    decoded = torch.zeros(512, 512, dtype=torch.bfloat16, device="hpu")
    torch.manual_seed(20260920)
    values = [torch.randn(1, 512, dtype=torch.bfloat16, device="hpu")
              for _ in range(5)]
    positions = tuple(range(254, 259))
    completions = []
    for position, value in zip(positions, values):
        ring = torch.tensor([position % 256], dtype=torch.int32, device="hpu")
        decoded_row = torch.tensor([position if logical_decoded_row else position % 256],
                                   dtype=torch.int32, device="hpu")
        completions.append(
            torch.ops.custom_op.custom_deepseek_v41_swa_paged_decoded_write_bf16_gaudi2(
                packed, value, ring, decoded_row, decoded, 0))
    torch.hpu.synchronize()
    packed_host, decoded_host = packed.cpu(), decoded.cpu()
    canonical = unpack_swa(packed_host)
    rows = {}
    for row in (254, 255, 0, 1, 2):
        actual, expected = decoded_host[row], canonical[row]
        rows[str(row)] = {
            "bf16_mismatches": int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        }
    return {
        "mode": "logical_decoded_row" if logical_decoded_row else "ring_decoded_row",
        "positions": list(positions),
        "completion_rows": [entry.cpu().tolist() for entry in completions],
        "rows": rows,
        "total_bf16_mismatches": sum(item["bf16_mismatches"] for item in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    library = os.environ.get("VLLM_HPU_DSV4_TPC_OP_LIBRARY")
    if not library:
        raise RuntimeError("VLLM_HPU_DSV4_TPC_OP_LIBRARY is required")
    torch.ops.load_library(library)
    payload = {
        "benchmark": "dsv41_swa_decoded_ring_wrap",
        "scope": "native paged write at logical positions 254..258",
        "legacy_control": run(True),
        "fixed_contract": run(False),
    }
    payload["passed"] = (
        payload["legacy_control"]["total_bf16_mismatches"] > 0
        and payload["fixed_contract"]["total_bf16_mismatches"] == 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
