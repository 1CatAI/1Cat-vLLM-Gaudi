# SPDX-License-Identifier: Apache-2.0
"""Qualify bounded Reindex scores and the first packed Attention consumer."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--tiled-keys", action="store_true", help="Check affine SRAM key producer")
    parser.add_argument("--bounded", action="store_true", help="Check compact selection before native skipping")
    parser.add_argument("--candidate-pool-layout",
                        choices=("legacy-scattered", "full-producer"),
                        default="legacy-scattered",
                        help="Choose the producer contract for score comparisons")
    parser.add_argument("--reference-timing",
                        action="store_true",
                        help="Measure only a missing comparable component reference")
    parser.add_argument("--reference-result", type=Path)
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4
    from vllm_gaudi.ops.deepseek_v41_reindex_mme import (
        bounded_reindex_mme_select,
        reindex_mme_scores,
        reindex_mme_select,
    )
    from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
    from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla

    torch.set_num_threads(1)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(381)
    batch, capacity = args.batch, args.capacity
    report = dict(batch=batch,
                  capacity=capacity,
                  tile=args.tile,
                  tiled_keys=args.tiled_keys,
                  bounded=args.bounded,
                  candidate_pool_layout=args.candidate_pool_layout,
                  checks=[],
                  timing={},
                  qualified=False,
                  scope="Synthetic index/MLA inputs; no full-model or real-weight performance qualification")
    if args.reference_result:
        report["reference_result"] = str(args.reference_result.resolve())

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    def compile_entry(fn, name):
        entry = FunctionType(fn.__code__.replace(co_name=name),
                             fn.__globals__,
                             argdefs=fn.__defaults__,
                             closure=fn.__closure__)
        return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)

    def boundary(value):
        return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(value.reshape(1, -1)).reshape(value.shape)

    # Codec inputs use the production FP4 roundtrip; weights retain BF16.
    qhost = unpack_fp4(pack_fp4(torch.randn(batch, 32, 128).bfloat16(), 32), 128, 32)
    weights = (torch.randn(batch, 32) * .02).bfloat16().to("hpu")
    query = qhost.to("hpu")
    packed = pack_fp4(torch.randn(capacity + 128, 128).bfloat16(), 32).to("hpu")
    main = pack_fp4(torch.randn(capacity + 128, 512).bfloat16(), 16).to("hpu")
    swa = pack_swa(torch.randn(batch * 256, 512).bfloat16()).to("hpu")
    aq = torch.randn(batch, 32, 512).bfloat16().to("hpu")
    sink = torch.randn(32, device="hpu", dtype=torch.float32)
    scale = torch.tensor([512**-0.5], device="hpu")
    slots = torch.arange(batch, dtype=torch.int32, device="hpu")
    done = torch.zeros(batch, dtype=torch.int32, device="hpu")
    positions = torch.zeros(batch, dtype=torch.int32, device="hpu")
    candidates = torch.full((batch, 2048), -1, dtype=torch.int32, device="hpu")
    measured_args = None
    for ratio in (1, 2):
        page_rows = 128 // ratio
        pages = torch.stack([torch.randperm(capacity // page_rows) + 1 for _ in range(batch)]).int().to("hpu")

        def score_fn(q, w, cache, page, pos, pool, ratio=ratio):
            return reindex_mme_scores(q, w, cache, page, pos, pool, ratio, args.tile, tiled_keys=args.tiled_keys)

        score_program = compile_entry(score_fn, f"reindex_score_{ratio}")
        for change, visible in enumerate((513, 2048, 4096, 8193, capacity - 1, 511)):
            lengths = torch.tensor([visible - i * 3 for i in range(batch)], dtype=torch.int32)
            lengths[-1] = 0 if batch > 1 else visible
            positions.copy_(lengths * ratio - 1)
            pool = torch.full((batch, 2048), -1, dtype=torch.int32)
            for row in range(batch):
                count = (max(0, int(lengths[row])) + 7) // 8
                blocks = (torch.arange(count) if args.candidate_pool_layout == "full-producer" and count <= 2048 else
                          torch.randperm(count)[:2048]).int()
                pool[row, :blocks.numel()] = blocks
            if args.candidate_pool_layout == "legacy-scattered":
                pool = pool[:, torch.randperm(2048)].contiguous()
                pool[:, 17:23] = -1
            candidates.copy_(pool)
            if change:
                query.copy_(unpack_fp4(pack_fp4(torch.randn(batch, 32, 128).bfloat16(), 32), 128, 32))
            expected, _ = torch.ops.custom_op.custom_deepseek_v41_index_scores_gaudi2(
                query, weights, packed, pages, positions, candidates, ratio, 1, 16384)
            actual = score_program(query, weights, packed, pages, positions, candidates)
            # Scores are deliberately unwritten by the old entry for <=512.
            eligible = lengths > 512
            e, a = expected.cpu()[eligible], actual.cpu()[eligible]
            exact = torch.equal(e, a)
            case = dict(ratio=ratio, visible=visible, score_exact=exact, mismatch=int((e != a).sum()))
            if e.numel():
                finite = e.isfinite() & a.isfinite()
                case["finite_maxabs"] = float((e[finite] - a[finite]).abs().max())
            report["checks"].append(case)
            save()
            print(json.dumps(case), flush=True)
            if not exact:
                torch.save(
                    dict(expected=e,
                         actual=a,
                         query=query.cpu(),
                         weights=weights.cpu(),
                         positions=positions.cpu(),
                         candidates=pool), args.output.with_suffix(".failure.pt"))
                raise RuntimeError("Score contract mismatch; no speed qualification")

        def chain(q,
                  w,
                  cache,
                  page,
                  pos,
                  pool,
                  attn_q,
                  sw,
                  kv,
                  bias,
                  scaling,
                  request_slots,
                  completion,
                  candidate=False,
                  ratio=ratio):
            if candidate:
                if args.bounded:
                    ids = bounded_reindex_mme_select(q,
                                                     w,
                                                     cache,
                                                     page,
                                                     pos,
                                                     pool,
                                                     ratio=ratio,
                                                     tiled_keys=args.tiled_keys)
                else:
                    ids = reindex_mme_select(q,
                                             w,
                                             cache,
                                             page,
                                             pos,
                                             pool,
                                             ratio=ratio,
                                             tile=args.tile,
                                             tiled_keys=args.tiled_keys)
            else:
                ids = runtime_index_select(q, w, cache, page, pos, pool, ratio=ratio, capacity=capacity,
                                           reindex=True)[0]
            output = batch_packed_mla(attn_q,
                                      sw,
                                      kv,
                                      ids,
                                      page,
                                      pos,
                                      request_slots,
                                      bias,
                                      scaling,
                                      completion,
                                      completion,
                                      ratio=ratio)
            return ids, boundary(output)

        def old(*values):
            return chain(*values, candidate=False)

        def new(*values):
            return chain(*values, candidate=True)

        old_program = compile_entry(old, f"reindex_consumer_reference_{ratio}")
        new_program = compile_entry(new, f"reindex_consumer_candidate_{ratio}")
        values = (query, weights, packed, pages, positions, candidates, aq, swa, main, sink, scale, slots, done)
        for visible in (531, 2048, 4096, 8193):
            lengths = torch.tensor([visible - i * 2 for i in range(batch)], dtype=torch.int32)
            lengths[-1] = 0 if batch > 1 else visible
            positions.copy_(lengths * ratio - 1)
            pool.fill_(-1)
            for row in range(batch):
                count = (max(0, int(lengths[row])) + 7) // 8
                blocks = (torch.arange(count) if args.candidate_pool_layout == "full-producer" and count <= 2048 else
                          torch.randperm(count)[:2048]).int()
                pool[row, :blocks.numel()] = blocks
            candidates.copy_(pool if args.candidate_pool_layout ==
                             "full-producer" else pool[:, torch.randperm(2048)].contiguous())
            old_outputs, new_outputs = old_program(*values), new_program(*values)
            equal = [torch.equal(a.cpu(), b.cpu()) for a, b in zip(old_outputs, new_outputs, strict=True)]
            report["checks"].append(dict(ratio=ratio, visible=visible, consumer_exact=equal))
            save()
            assert all(equal), report["checks"][-1]
        measured_args = values, old_program, new_program

    if not args.checks_only:
        values, old_program, new_program = measured_args
        # New missing component reference includes the same actual MLA consumer.
        programs = {}
        for name, program in (("reference", old_program), ("candidate", new_program)):
            for _ in range(3):
                program(*values)
            torch.hpu.synchronize()
            outputs = program(*values)
            torch.hpu.synchronize()
            programs[name] = program
        for visible in (511, 531, 8193, capacity - 1):
            positions.copy_(torch.full((batch, ), visible * ratio - 1, dtype=torch.int32))
            pool.fill_(-1)
            for row in range(batch):
                blocks = torch.randperm((visible + 7) // 8)[:2048].int()
                pool[row, :blocks.numel()] = blocks
            candidates.copy_(pool[:, torch.randperm(2048)].contiguous())
            per_length = {}
            outputs_by_name = {}
            for name, program in programs.items():
                outputs_by_name[name] = program(*values)
                torch.hpu.synchronize()
                if name == "reference" and not args.reference_timing:
                    continue
                raw = []
                for _ in range(25):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    wall = time.perf_counter()
                    start.record()
                    outputs = program(*values)
                    stop.record()
                    stop.synchronize()
                    raw.append(dict(device_ms=start.elapsed_time(stop), wall_ms=(time.perf_counter() - wall) * 1000))
                per_length[name] = dict(samples=raw,
                                        median_device_ms=statistics.median(x["device_ms"] for x in raw),
                                        median_wall_ms=statistics.median(x["wall_ms"] for x in raw))
                torch.save([x.cpu() for x in outputs], args.output.with_name(f"{name}-consumer-{visible}.pt"))
                outputs_by_name[name] = outputs
            equality = [
                torch.equal(a.cpu(), b.cpu())
                for a, b in zip(outputs_by_name["reference"], outputs_by_name["candidate"], strict=True)
            ]
            per_length["consumer_exact"] = equality
            report["timing"][str(visible)] = per_length
            save()
            assert all(equality), per_length
    report["qualified"] = True
    save()


if __name__ == "__main__":
    main()
