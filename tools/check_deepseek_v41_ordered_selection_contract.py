# SPDX-License-Identifier: Apache-2.0
"""Exact key, worker-prefix and output checks for the native index selector."""

import json
import os
from pathlib import Path


def check_contract(torch, checks=None):
    from vllm_gaudi.ops.deepseek_v41_ordered_selection import ordered_index_ids

    generator = torch.Generator().manual_seed(7801)
    checks = [] if checks is None else checks
    cases = [(1, 1, [length], False, False) for length in (511, 512, 513, 32767, 32768, 32769, 65537)]
    cases += [(2, 2, [8257, 8258], False, False),
              (6, 1, [16377, 16384, 16385, 16515, 16516, 16517], False, True),
              (2, 1, [16515, 16516], True, False),
              (2, 1, [513, 1025], True, False)]
    for batch, ratio, visible, reindex, blocks in cases:
        width = 2048 if blocks else 512
        limits = [(v + 7) // 8 if blocks else min((v + 7) // 8, 2048) * 8 if reindex else v
                  for v in visible]
        columns = (max(limits) + 63) // 64 * 64
        positions = torch.tensor([v * ratio - 1 for v in visible], dtype=torch.int32)
        candidates = torch.arange(2048, dtype=torch.int32).repeat(batch, 1)
        for row, v in enumerate(visible):
            if reindex and v > 16384:
                candidates[row, -1] = (v - 1) // 8

        def select(scores, pos, pool, ratio=ratio, reindex=reindex, blocks=blocks):
            metadata = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
                scores, pos, ratio, int(reindex), int(blocks))
            ids = ordered_index_ids(scores, pos, pool, ratio, reindex=reindex, blocks=blocks)
            return metadata, ids

        # Independent contract shapes are separate cold programs, not runtime
        # shape changes of one serving graph.
        torch._dynamo.reset()
        call = torch.compile(select, backend="hpu_backend", fullgraph=True, dynamic=False)
        device_positions = positions.to("hpu")
        device_candidates = candidates.to("hpu")
        for pattern in ("random", "tied", "sparse_signed_zero"):
            scores = torch.randn(batch, columns, generator=generator).bfloat16().float()
            if pattern == "tied":
                scores.zero_()
            elif pattern == "sparse_signed_zero":
                scores.fill_(-torch.inf)
                scores[:, ::11] = -0.0
                scores[:, ::17] = 0.0
                scores[:, ::23] = torch.inf
            for row, (v, limit) in enumerate(zip(visible, limits, strict=True)):
                scores[row, limit:] = -torch.inf
                if reindex:
                    logical = (candidates[row, :, None] * 8 + torch.arange(8)).flatten()[:columns]
                    scores[row, logical >= v] = -torch.inf
                if blocks:
                    scores[row, (v - 1) // 8] = torch.inf

            expected_ids = torch.full((batch, width), -1, dtype=torch.int32)
            expected_metadata = torch.empty((batch, 50), dtype=torch.int32)
            bitplanes = {}
            active = []
            for row, (v, limit) in enumerate(zip(visible, limits, strict=True)):
                if v <= 512 or blocks and limit <= 2048:
                    expected_ids[row, :min(limit, width)] = torch.arange(min(limit, width))
                    continue
                active.append(row)
                values = scores[row, :limit]
                bits = (values.view(torch.int32).long() & 0xffffffff) >> 16
                keys = torch.where(bits > 32767, (~bits) & 65535, bits ^ 32768)
                keys = torch.where(values > -torch.inf, keys, 0)
                selected = torch.argsort(keys, descending=True, stable=True)[:width]
                selected = selected[keys[selected] > 0].sort().values
                mapped = (candidates[row, selected // 8] * 8 + selected % 8) if reindex else selected
                expected_ids[row, :selected.numel()] = mapped.int()
                cutoff = keys.sort(descending=True).values[width - 1] if (keys > 0).sum() >= width else 0
                greater = (keys > cutoff) & (keys > 0)
                equal = (keys == cutoff) & (keys > 0)
                padded = (limit + 127) // 128 * 128
                planes = []
                for predicate in (greater, equal):
                    bits = torch.zeros(padded, dtype=torch.int64)
                    bits[:limit] = predicate
                    planes.append((bits.reshape(-1, 32) << torch.arange(32)).sum(-1).int())
                bitplanes[row] = planes
                expected_metadata[row, 0] = cutoff
                expected_metadata[row, 1] = greater.sum()
                partitions = 16384 if reindex else limit
                for worker in range(24):
                    stop = min(partitions * worker // 24, limit)
                    expected_metadata[row, 2 + worker * 2] = greater[:stop].sum()
                    expected_metadata[row, 3 + worker * 2] = equal[:stop].sum()
            metadata, ids = call(scores.to("hpu"), device_positions, device_candidates)
            actual_ids, actual_metadata = ids.cpu(), metadata.cpu()
            assert torch.equal(actual_metadata[active, :50], expected_metadata[active]), (
                batch, ratio, visible, reindex, blocks, pattern, "metadata",
                actual_metadata[active, :50].tolist(), expected_metadata[active].tolist())
            if actual_metadata.shape[1] > 50:
                words = (columns + 127) // 128 * 4
                assert actual_metadata.shape[1] == 50 + words * 2
                for row in active:
                    for plane, expected in enumerate(bitplanes[row]):
                        actual = actual_metadata[row, 50 + plane * words:50 + plane * words + expected.numel()]
                        assert torch.equal(actual, expected), (visible, pattern, "bitplane", row, plane,
                                                               actual[:16].tolist(), expected[:16].tolist())
            assert torch.equal(actual_ids, expected_ids), (batch, ratio, visible, reindex, blocks, pattern,
                                                          "output", (actual_ids != expected_ids).nonzero()[:8])
            checks.append(dict(batch=batch, ratio=ratio, visible=visible, reindex=reindex, blocks=blocks,
                               pattern=pattern, ordered_ids_exact=True, prefix_metadata_exact=True))
            print(json.dumps(checks[-1]), flush=True)
    return checks


def main():
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", "0")
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    bind_worker_cpu(0)
    bind_worker_helpers(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    checks = []
    report = dict(status="running", checks=checks, gain_ledger_credit=False)
    try:
        with torch.inference_mode():
            check_contract(torch, checks)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="passed", exact_cases=len(checks))), flush=True)


if __name__ == "__main__":
    main()
