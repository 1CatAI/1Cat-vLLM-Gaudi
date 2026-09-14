# SPDX-License-Identifier: Apache-2.0
"""Exact candidate attention comparison; run with an explicit one-device lease."""

import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_paged_attention import (  # noqa: E402
    _selected_attention_layout, _shared_prefix_attention_layout,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-exp", action="store_true")
    parser.add_argument("--vector-scales", action="store_true")
    parser.add_argument("--sram-kv", action="store_true")
    parser.add_argument("--head-vector", action="store_true")
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(20260912)
    swa = pack_swa(torch.randn(256, 512).bfloat16()).to("hpu")
    main_cache = pack_fp4(torch.randn(1024, 512).bfloat16()).to("hpu")
    sink = torch.randn(32, device="hpu")
    scale = torch.tensor([512**-0.5], dtype=torch.float32, device="hpu")
    swa_offsets = torch.arange(256, dtype=torch.int32, device="hpu")
    selected_offsets = torch.arange(3072, dtype=torch.int32, device="hpu")

    def reference(q, physical, selected, window):
        rows, indices = _selected_attention_layout(physical, selected, window, swa_offsets, selected_offsets)
        return torch.ops.custom_op.custom_deepseek_v41_paged_attention_bf16_gaudi2(q, swa, main_cache, rows, indices,
                                                                                   sink, scale)

    def candidate(q, physical, selected, window):
        rows, indices, lengths = _shared_prefix_attention_layout(physical, selected, window, swa_offsets)
        op = (torch.ops.custom_op.custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2 if args.head_vector else
              torch.ops.custom_op.custom_deepseek_v41_paged_attention_sram_bf16_gaudi2 if args.sram_kv else
              torch.ops.custom_op.custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2 if args.vector_scales
              else torch.ops.custom_op.custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2 if args.
              packed_exp else torch.ops.custom_op.custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2)
        return op(q, swa, main_cache, rows, indices, sink, scale, lengths)

    old = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    new = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    for tokens, start, ratio in ((1, 0, 2), (2, 7, 2), (3, 126, 2), (4, 254, 2), (5, 127, 128), (6, 0, 2), (6, 126, 2),
                                 (6, 254, 2), (6, 1018, 2), (6, 127, 128)):
        positions = torch.arange(start, start + tokens, dtype=torch.int32)
        window = positions[:, None] - 127 + torch.arange(128, dtype=torch.int32)
        window = torch.where(window >= 0, window.remainder(256), -1)
        rows = torch.arange(512, dtype=torch.int32).expand(tokens, -1)
        selected = torch.where(rows < ((positions + 1) // ratio)[:, None], rows, -1).contiguous()
        physical = (selected.clamp_min(0) * 7 + 13).remainder(1024)
        q = torch.randn(tokens, 32, 512).bfloat16()
        inputs = [x.to("hpu") for x in (q, physical, selected, window)]
        expected, actual = old(*inputs).cpu(), new(*inputs).cpu()
        mismatch = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
        records.append(dict(tokens=tokens, start=start, ratio=ratio, bf16_mismatches=mismatch))
        (output / "result.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
        if mismatch:
            torch.save(dict(q=q, physical=physical, selected=selected, window=window, expected=expected, actual=actual),
                       output / "mismatch.pt")
            raise RuntimeError("Prefix KV candidate differs from existing BF16 attention")
    torch.hpu.synchronize()


if __name__ == "__main__":
    main()
