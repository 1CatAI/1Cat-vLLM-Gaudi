#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact device checks for fused short-search metadata, with changing inputs."""
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401
from vllm_gaudi.ops.deepseek_v41_paged_attention import _shared_prefix_attention_layout  # noqa: E402


def reference(selected, positions, block_table, ratio):
    width = 128 // ratio
    rows = selected.clamp_min(0)
    blocks = block_table.index_select(0, (rows.flatten() // width).long()).reshape(rows.shape)
    physical = blocks * width + rows.remainder(width)
    window = positions.unsqueeze(-1) - 127 + torch.arange(128, device=positions.device, dtype=torch.int32)
    window = torch.where(window >= 0, window.remainder(256), -1).int()
    swa = torch.arange(256, device=positions.device, dtype=torch.int32)
    return _shared_prefix_attention_layout(physical, selected, window, swa)


def main():
    destination = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch._dynamo.config.recompile_limit = 64
    old = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    ops = [
        torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2,
        torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r2_i32_gaudi2
    ]
    candidates = [torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False) for op in ops]
    torch.manual_seed(26)
    records = []
    for ratio in (1, 2):
        new = candidates[ratio - 1]
        for count in range(1, 7):
            positions = torch.tensor([0, 1, 127, 128, 255, 511][:count], dtype=torch.int32)
            selected = torch.arange(512, dtype=torch.int32).expand(count, -1)
            selected = torch.where(selected < ((positions + 1) // ratio)[:, None], selected, -1)
            cases = [(f"C{count}", selected, positions)]
            if count == 6:
                cases += [
                    ("empty", torch.full_like(selected, -1), positions.flip(0)),
                    ("repeated_reordered", torch.randint(-1, 512, selected.shape, dtype=torch.int32),
                     torch.tensor([1048575, 512, 256, 128, 127, 0], dtype=torch.int32)),
                ]
            for label, selected, positions in cases:
                blocks = torch.randperm(8192, dtype=torch.int32)
                args = [x.contiguous().to("hpu") for x in (selected, positions, blocks)]
                expected = [x.cpu() for x in old(*args, ratio)]
                actual = [x.cpu() for x in new(*args)]
                bad = [int((a != b).sum()) for a, b in zip(expected, actual, strict=True)]
                records.append(dict(ratio=ratio, case=label, count=count, mismatches=bad))
                (destination / "result.json").write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps(records[-1]), flush=True)
                if any(bad):
                    torch.save(
                        dict(selected=selected,
                             positions=positions,
                             blocks=blocks,
                             expected=expected,
                             actual=actual,
                             ratio=ratio), destination / "mismatch.pt")
                    raise RuntimeError("Fused prefix metadata differs from compiled reference")


if __name__ == "__main__":
    main()
