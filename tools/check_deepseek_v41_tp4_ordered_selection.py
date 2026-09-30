# SPDX-License-Identifier: Apache-2.0
"""Cheap equivalence gate before copying the TP2 ordered selector into TP4.

This is a selection-contract diagnostic, not a complete-chain performance
benchmark. No model weights, distributed communication, or serving request.
"""
import json
import os
from pathlib import Path


def main():
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    torch.hpu.set_device(0)
    bind_worker_cpu(0)
    bind_worker_helpers(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = {"status": "running", "scope": __doc__, "checks": [], "timed": False,
              "model_requests": 0, "performance_credit": False}

    def save():
        (root / "SELECTION_CONTRACT.json").write_text(json.dumps(report, indent=2) + "\n")

    def current(scores, rows, positions):
        values = selected = None
        for offset in range(0, scores.shape[-1], 2048):
            values, selected = PagedCSA2Attention._merge_topk(
                values, selected, scores[:, offset:offset + 2048], rows[offset:offset + 2048], 512)
        valid = (selected >= 0) & (selected < positions.reshape(-1, 1) + 1)
        selected = torch.where(valid, selected, 1048576).sort(-1).values
        return torch.where(selected < 1048576, selected, -1).int()

    def ordered(scores, candidates, positions):
        metadata = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
            scores, positions, 1, 0, 0)
        return torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
            scores, positions, candidates, metadata, 1, 0, 0)

    current_call = torch.compile(current, backend="hpu_backend", fullgraph=True, dynamic=False)
    ordered_call = torch.compile(ordered, backend="hpu_backend", fullgraph=True, dynamic=False)
    candidates = torch.zeros((1, 2048), dtype=torch.int32, device="hpu")
    save()
    with torch.inference_mode():
        for columns, visible in ((10240, 8257), (20480, 16515)):
            rows = torch.arange(columns, dtype=torch.int32, device="hpu")
            positions = torch.tensor([visible - 1], dtype=torch.int32, device="hpu")
            torch.manual_seed(1729)
            for mode in ("bf16_positive_scores", "equal_cutoff"):
                if mode == "bf16_positive_scores":
                    source = torch.rand(1, columns).bfloat16().float() * 8
                else:
                    source = torch.ones(1, columns)
                source[:, visible:] = -torch.inf
                device_scores = source.to("hpu")
                expected = current_call(device_scores, rows, positions).cpu()
                actual = ordered_call(device_scores, candidates, positions).cpu()
                exact = torch.equal(expected, actual)
                row = {"columns": columns, "visible_rows": visible, "mode": mode,
                       "indices_exact": exact,
                       "different_positions": int((expected != actual).sum()),
                       "intersection_count": len(set(expected[0].tolist()) & set(actual[0].tolist())),
                       "first_current_ids": expected[0, :16].tolist(),
                       "first_ordered_ids": actual[0, :16].tolist()}
                report["checks"].append(row)
                print(json.dumps(row), flush=True)
                if not exact:
                    torch.save({"scores": source, "current": expected, "ordered": actual,
                                "positions": positions.cpu()}, root / "first_selection_difference.pt")
                    report.update(status="rejected_direct_copy_cutoff_tie_contract",
                                  decision="Do not build or time an incompatible complete model chain.")
                    save()
                    return
        report.update(status="selection_equivalence_passed",
                      decision="Complete producer-consumer timing still required.")
        save()


if __name__ == "__main__":
    main()
