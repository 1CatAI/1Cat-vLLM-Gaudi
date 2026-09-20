# SPDX-License-Identifier: Apache-2.0
"""Diagnose fused V4.1 router score generation before normalization."""
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def main():
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    prepared = Path(os.environ["DSV41_PREPARED_DIR"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(410920)
    shard = PreparedV41Shard(prepared, 0, 0)
    rows = []
    with torch.inference_mode():
        for layer in range(20):
            prefix = f"layers.{layer}.ffn.gate."
            rows.append((
                torch.randn(1, 5120, dtype=torch.bfloat16, device="hpu"),
                shard.tensor(prefix + "weight", "hpu"),
                shard.tensor(prefix + "bias", "hpu"),
                shard.tensor(prefix + "bias_vl", "hpu"),
                torch.tensor([layer % 7 == 0], dtype=torch.bool, device="hpu"),
            ))

        def parent(x, weight, text, image, mask):
            logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                x.contiguous(), weight)
            scores = F.softplus(logits).sqrt()
            ids, _ = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2(
                scores, text, image, mask)
            return ids, scores

        def candidate(x, weight, text, image, mask):
            logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                x.contiguous(), weight)
            return torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                logits, text, image, mask)

        compiled_parent = torch.compile(parent, backend="hpu_backend", fullgraph=True,
                                        dynamic=False)
        compiled_candidate = torch.compile(candidate, backend="hpu_backend", fullgraph=True,
                                           dynamic=False)
        for _ in range(3):
            for row in rows:
                compiled_parent(*row)
                compiled_candidate(*row)
        torch.hpu.synchronize()

        checks = []
        for change in (1.0, 1.03125, 0.96875):
            for row in rows:
                row[0].mul_(change)
            for layer, row in enumerate(rows):
                parent_ids, parent_scores = tuple(v.cpu() for v in compiled_parent(*row))
                candidate_ids, candidate_selected = tuple(
                    v.cpu() for v in compiled_candidate(*row))
                parent_selected = parent_scores.gather(1, parent_ids.long())
                delta = (parent_selected - candidate_selected).abs()
                checks.append({
                    "change": change,
                    "layer": layer,
                    "ids_equal": torch.equal(parent_ids, candidate_ids),
                    "scores_bitwise_equal": torch.equal(parent_selected,
                                                           candidate_selected),
                    "score_max_abs": float(delta.max()),
                    "score_mean_abs": float(delta.mean()),
                })
        torch.hpu.synchronize()
        record = {
            "scope": "20 real PP0 gate matrices; selected raw scores before normalization",
            "checks": checks,
            "ids_all_equal": all(v["ids_equal"] for v in checks),
            "scores_all_bitwise_equal": all(v["scores_bitwise_equal"] for v in checks),
        }
        (output / "router-scores.json").write_text(json.dumps(record, indent=2) + "\n")
        bad = sum(not v["scores_bitwise_equal"] for v in checks)
        print(f"router selected scores: {bad}/60 differ", flush=True)


if __name__ == "__main__":
    main()
