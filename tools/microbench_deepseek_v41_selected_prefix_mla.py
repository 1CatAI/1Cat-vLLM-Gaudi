# SPDX-License-Identifier: Apache-2.0
"""Measure removing the decoded C1 prefix-layout materialization.

The reference executes the production prefix-layout TPC and then the retained
MLA -> wo_a -> wo_b compound operator.  The candidate gives selection and
position directly to the same compound chain, deriving the fixed 128-row SWA
prefix in its gather producer.  Timing ends at the persistent BF16 wo_b output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_deepseek_v41_projection_chains import benchmark
from deepseek_v41_micro_replay import RecipeRecorder
import torch

from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
from vllm_gaudi.ops.deepseek_v41_math import rotary_table
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar


def reference(q, swa, main, selected, position, sink, scale, completion,
              block_table, woa, woa_scale, phase, wob, wob_scale):
    layout = torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2
    _, indices, lengths = layout(selected, position, block_table)
    return torch.ops.custom_op.custom_deepseek_v41_mla_woa_wob_fp8_roundtrip_gaudi2(
        q, swa, main, indices, sink, scale, lengths, completion, completion,
        0, main.shape[0], 256, woa, woa_scale, position, phase, wob,
        wob_scale)


def candidate(q, swa, main, selected, position, sink, scale, completion,
              block_table, woa, woa_scale, phase, wob, wob_scale):
    # block_table intentionally remains a graph operand so reference and
    # candidate use an identical benchmark interface. The direct decoded
    # mirror is logically addressed and does not need it.
    del block_table
    return torch.ops.custom_op.custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2(
        q, swa, main, selected, sink, scale, completion, completion, 0,
        main.shape[0], woa, woa_scale, position, phase, wob, wob_scale)


def make_selected(position: int, ratio: int) -> torch.Tensor:
    selected = torch.arange(512, dtype=torch.int32).reshape(1, 512)
    valid = (position + 1) // ratio
    selected[selected >= valid] = -1
    return selected.to("hpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("woa_sidecar", type=Path)
    parser.add_argument("dense_sidecar", type=Path)
    parser.add_argument("--position", type=int, default=511)
    args = parser.parse_args()

    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(20260921)
    recorder = RecipeRecorder(output)
    pp_rank = 1
    shard = PreparedV41Shard(args.prepared, pp_rank, 0)
    woa = WoaFP8Sidecar(args.woa_sidecar, shard)
    dense = DenseFP8Sidecar(args.dense_sidecar, shard)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    scaling = config["rope_scaling"]
    inputs, weights = [], []

    with torch.inference_mode():
        for layer in range(pp_rank * 20, (pp_rank + 1) * 20):
            ratio = int(config["compress_ratios"][layer])
            if ratio != 1:
                raise AssertionError((layer, ratio, "PP1 decoded prefix benchmark expects ratio one"))
            prefix = f"layers.{layer}.attn."
            table = rotary_table(
                64, 512, config["compress_rope_theta"],
                scaling["original_max_position_embeddings"], scaling["factor"],
                scaling["beta_fast"], scaling["beta_slow"])
            inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu")
            inputs.append(torch.randn(1, 32, 512, dtype=torch.bfloat16,
                                      device="hpu"))
            weights.append((
                # The production shared decoded-SWA allocation reserves a
                # 512-row stride per layer even though this fixed prefix uses
                # only its 256-row ring.
                torch.randn(512, 512, dtype=torch.bfloat16, device="hpu"),
                torch.randn(512 // ratio, 512, dtype=torch.bfloat16,
                            device="hpu"),
                make_selected(args.position, ratio),
                torch.tensor([args.position], dtype=torch.int32, device="hpu"),
                shard.tensor(prefix + "attn_sink", "hpu").float(),
                torch.tensor([512**-.5], dtype=torch.float32, device="hpu"),
                torch.zeros(1, dtype=torch.int32, device="hpu"),
                torch.arange(8, dtype=torch.int32, device="hpu"),
                woa.tensor(prefix + "wo_a.weight", "hpu"),
                woa.tensor(prefix + "wo_a.channel_scale", "hpu"),
                inverse,
                dense.tensor(prefix + "wo_b.weight", "hpu"),
                dense.tensor(prefix + "wo_b.channel_scale", "hpu"),
            ))

        # Values, not shapes, cover the short-prefix boundary on the PP1
        # ratio-one path represented by the archived trace.
        boundary_checks = []
        ratio, index = 1, 0
        for logical in (0, 127, 128, args.position):
            row = list(weights[index])
            row[2] = make_selected(logical, ratio)
            row[3] = torch.tensor([logical], dtype=torch.int32,
                                  device="hpu")
            wanted = reference(inputs[index], *row).cpu()
            actual = candidate(inputs[index], *row).cpu()
            different = int((wanted.view(torch.int16) !=
                             actual.view(torch.int16)).sum())
            boundary_checks.append({"ratio": ratio, "position": logical,
                                    "bf16_bit_mismatches": different})
            if different:
                raise AssertionError((ratio, logical, different))

        result = benchmark(
            f"selected-prefix-mla-p{pp_rank}-pos{args.position}",
            reference, candidate, inputs, weights, weights, output, 3,
            recorder)
        result.update({
            "scope": ("20 PP-stage decoded Attention layers from fixed-prefix "
                      "layout through persistent BF16 wo_b output"),
            "boundary_checks": boundary_checks,
            "candidate_removes": [
                "one prefix-layout TPC launch per layer",
                "materialized I32 [1,768] row ids",
                "materialized I32 [1,640] attention indices",
                "materialized I32 [1] lengths",
            ],
        })
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
