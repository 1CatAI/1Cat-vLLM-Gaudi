# SPDX-License-Identifier: Apache-2.0
"""Validate a fixed-capacity native indexer before loading model weights."""

import argparse
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--lengths", default="511,512,513,1023,1024,1025,4097,16385")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4

    torch.ops.load_library(str(args.library))
    torch.manual_seed(149)
    capacity = args.capacity
    q = unpack_fp4(pack_fp4(torch.randn(1, 32, 128).bfloat16(), 32), 128, 32)
    weights = (torch.randn(1, 32) * .02).bfloat16()
    # Keep CPU codec temporaries bounded even when testing a 1M allocation.
    keys = torch.empty(capacity, 128, dtype=torch.bfloat16)
    for start in range(0, capacity, 16384):
        stop = min(capacity, start + 16384)
        keys[start:stop] = unpack_fp4(pack_fp4(torch.randn(stop - start, 128).bfloat16(), 32), 128, 32)
    report = {"capacity": capacity, "cases": [], "qualified": False}

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    for ratio in (1, 2):
        page_rows = 128 // ratio
        pages = torch.randperm(capacity // page_rows).int() + 1
        physical = pages.repeat_interleave(page_rows).long() * page_rows
        physical += torch.arange(capacity) % page_rows
        packed = torch.zeros(capacity + page_rows, 68, dtype=torch.uint8)
        for start in range(0, capacity, 16384):
            stop = min(capacity, start + 16384)
            packed[physical[start:stop]] = pack_fp4(keys[start:stop], 32)
        candidates = torch.full((1, 2048), -1, dtype=torch.int32)
        hp_q, hp_w, hp_cache, hp_pages, hp_candidates = (x.to("hpu") for x in (q, weights, packed, pages, candidates))
        position = torch.zeros(1, dtype=torch.int32, device="hpu")

        def select(q, w, cache, table, pos, candidates, ratio=ratio):
            return runtime_index_select(q,
                                        w,
                                        cache,
                                        table,
                                        pos,
                                        candidates,
                                        ratio=ratio,
                                        capacity=capacity,
                                        reindex=args.reindex,
                                        publish_candidates=not args.reindex)

        compiled = torch.compile(select, backend="hpu_backend", fullgraph=True, dynamic=False)
        for visible in map(int, args.lengths.split(",")):
            if args.reindex:
                available = (visible + 7) // 8
                chosen = torch.randperm(available)[:2048].int()
                candidates.fill_(-1)
                candidates[0, :chosen.numel()] = chosen
                candidates = candidates[:, torch.randperm(2048)].contiguous()
                hp_candidates.copy_(candidates)
            position.copy_(torch.tensor([visible * ratio - 1], dtype=torch.int32))
            torch.hpu.synchronize()
            before = dict(metric_global("graph_compilation").stats())
            start = time.perf_counter()
            selected, blocks = compiled(hp_q, hp_w, hp_cache, hp_pages, position, hp_candidates)
            torch.hpu.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000
            after = dict(metric_global("graph_compilation").stats())
            selected = selected.cpu()
            blocks = blocks.cpu() if blocks is not None else None
            if visible <= 512:
                expected = torch.full((1, 512), -1, dtype=torch.int32)
                expected[0, :visible] = torch.arange(visible)
                expected_blocks = torch.full((1, 2048), -1, dtype=torch.int32)
                expected_blocks[0, :(visible + 7) // 8] = torch.arange((visible + 7) // 8)
                equal = torch.equal(selected, expected)
                blocks_equal = args.reindex or torch.equal(blocks, expected_blocks)
                max_error, score_equal = 0., True
            else:
                reference = (q.float().reshape(32, 128) @ keys[:visible].float().T).bfloat16().relu()
                reference = (reference * weights.flatten()[:, None]).reshape(2, 16, visible).sum(1).sum(0).float()
                rows = torch.arange(visible)
                if args.reindex:
                    rows = (candidates[0, :, None] * 8 + torch.arange(8)).flatten().long()
                    valid = (rows >= 0) & (rows < visible)
                    reference = reference[rows.clamp(0, visible - 1)].masked_fill(~valid, -torch.inf)
                expected = rows[reference.argsort(descending=True, stable=True)[:512]].sort().values.int()[None]
                equal = torch.equal(selected, expected)
                padded = torch.nn.functional.pad(reference, (0, -reference.numel() % 8), value=-torch.inf)
                block_scores = padded.reshape(-1, 8).amax(-1)
                block_scores[-1] = torch.inf
                nblocks = min(2048, block_scores.numel())
                chosen = block_scores.argsort(descending=True, stable=True)[:nblocks].sort().values.int()
                expected_blocks = torch.full((1, 2048), -1, dtype=torch.int32)
                expected_blocks[0, :nblocks] = chosen
                blocks_equal = args.reindex or torch.equal(blocks, expected_blocks)
                scores, _ = torch.ops.custom_op.custom_deepseek_v41_index_scores_gaudi2(
                    hp_q, hp_w, hp_cache, hp_pages, position, hp_candidates, ratio, int(args.reindex),
                    16384 if args.reindex else capacity)
                actual = scores.cpu()[:reference.numel()]
                finite = reference.isfinite()
                max_error = float((actual[finite] - reference[finite]).abs().max())
                score_equal = torch.equal(actual, reference)
            case = dict(ratio=ratio,
                        visible=visible,
                        wall_ms=wall_ms,
                        selected_equal=equal,
                        blocks_equal=blocks_equal,
                        scores_equal=score_equal,
                        max_score_error=max_error,
                        compilation_before=before,
                        compilation_after=after,
                        reindex=args.reindex,
                        first_selected=selected[0, :12].tolist(),
                        first_blocks=blocks[0, :12].tolist() if blocks is not None else None)
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            save()
            if not (equal and blocks_equal and score_equal):
                torch.save(
                    dict(query=q,
                         weights=weights,
                         keys=keys,
                         selected=selected,
                         expected=expected,
                         blocks=blocks,
                         expected_blocks=expected_blocks), args.output.with_suffix(".failure.pt"))
                raise RuntimeError("Indexer contract failed before performance qualification")
    report["qualified"] = True
    save()


if __name__ == "__main__":
    main()
