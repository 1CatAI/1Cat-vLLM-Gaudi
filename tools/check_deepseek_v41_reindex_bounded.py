# SPDX-License-Identifier: Apache-2.0
"""Bounded native Reindex -> packed MLA capability and component diagnostic.

All eight tile recipes and fixed buffers are prepared before measurement. The
native loop selects a prefix using existing host position metadata; it never
reads candidate counts back. Full-emitter pools must contain unique block IDs.
This is not a production stage, real-weight, TP or end-to-end qualification.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8, choices=(1, 4, 8, 32, 64))
    p.add_argument("--ratio", type=int, default=1, choices=(1, 2))
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--native-tile", action="store_true", help="Use the production optional compound tile")
    args = p.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.ops.deepseek_v41_reindex_compact import unique_pool_tile_bound
    from vllm_gaudi.ops.deepseek_v41_reindex_mme import reindex_mme_select, _bf16_boundary
    from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4
    torch.set_num_threads(1)
    torch.manual_seed(381)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    report = {
        "scope": __doc__,
        "batch": args.batch,
        "ratio": args.ratio,
        "cases": [],
        "qualified": False,
        "native_tile": args.native_tile
    }
    save = lambda: args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    batch, ratio = args.batch, args.ratio
    recorder = RecipeRecorder(args.output.parent)

    def compile_one(fn, name):
        renamed = FunctionType(fn.__code__.replace(co_name=name),
                               fn.__globals__,
                               argdefs=fn.__defaults__,
                               closure=fn.__closure__)
        return torch.compile(renamed, backend="hpu_backend", fullgraph=True, dynamic=False)

    page_width = 128 // ratio
    rows_per_request = 16384
    physical_rows = batch * rows_per_request + 128
    packed = torch.empty((physical_rows, 68), dtype=torch.uint8, device="hpu")
    main = torch.empty((physical_rows, 288), dtype=torch.uint8, device="hpu")
    for start in range(0, physical_rows, 4096):
        n = min(4096, physical_rows - start)
        packed[start:start + n].copy_(pack_fp4(torch.randn(n, 128).bfloat16(), 32))
        main[start:start + n].copy_(pack_fp4(torch.randn(n, 512).bfloat16(), 16))
    pages = torch.zeros((batch, 8192), dtype=torch.int32)
    for request in range(batch):
        n = rows_per_request // page_width
        pages[request, :n] = torch.randperm(n) + request * n + 1
    pages = pages.to("hpu")
    query = unpack_fp4(pack_fp4(torch.randn(batch, 32, 128).bfloat16(), 32), 128, 32).to("hpu")
    weights = (torch.randn(batch, 32) * .02).bfloat16().to("hpu")
    aq = torch.randn(batch, 32, 512).bfloat16().to("hpu")
    swa = pack_swa(torch.randn(batch * 256, 512).bfloat16()).to("hpu")
    sink = torch.randn(32, dtype=torch.float32, device="hpu")
    scale = torch.tensor([512**-.5], device="hpu")
    slots = torch.randperm(batch).int().to("hpu")
    done = torch.zeros(batch, dtype=torch.int32, device="hpu")
    positions = torch.full((batch, ), 8192 * ratio, dtype=torch.int32, device="hpu")
    pool = torch.full((batch, 2048), -1, dtype=torch.int32, device="hpu")

    def initialize(candidate_pool, pos):
        compact, source, counts = torch.ops.custom_op.custom_deepseek_v41_reindex_compact_gaudi2(
            candidate_pool, pos, ratio)
        return compact, source, counts

    init_fn = compile_one(initialize, "reindex_init")
    init_fn(pool, positions)
    initialize_plan = recorder.prepare(init_fn, [pool], [(positions, )])
    compact, source, counts = initialize_plan.outputs
    plans = [initialize_plan]
    for tile_index in range(8):

        def tile(q, w, cache, page, pos, compact_pool, index=tile_index):
            blocks = compact_pool[:, index * 256:(index + 1) * 256]
            rows = (blocks[..., None] * 8 + torch.arange(8, device=q.device, dtype=torch.int32)).flatten(1)
            visible = ((pos + 1) // ratio).unsqueeze(-1)
            valid = (rows >= 0) & (rows < visible) & (visible > 512)
            rows = torch.where(valid, rows, -1).contiguous()
            if args.native_tile:
                values = torch.ops.custom_op.custom_deepseek_v41_reindex_tile_gaudi2(
                    q, w, cache, page, rows, ratio, index)
            else:
                keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(cache, page, rows, ratio)
                dots = _bf16_boundary(torch.bmm(q, keys.transpose(1, 2)))
                values = torch.ops.custom_op.custom_deepseek_v41_index_reduce_gaudi2(dots.contiguous(), w)
            return values.masked_fill(~valid, -torch.inf)

        fn = compile_one(tile, f"reindex_tile_{tile_index}")
        params = (weights, packed, pages, positions, compact)
        fn(query, *params)
        plans.append(recorder.prepare(fn, [query], [params]))

    def consume(attention_query, compact_pool, block_counts, pos, sw, kv, page, req_slots, bias, scaling, completion,
                score0, score1, score2, score3, score4, score5, score6, score7):
        ops = torch.ops.custom_op
        all_scores = torch.cat((score0, score1, score2, score3, score4, score5, score6, score7), -1)
        valid = torch.arange(16384, device=attention_query.device)[None] < block_counts[:, None] * 8
        score_buffer = all_scores.masked_fill(~valid, -torch.inf)
        stats = ops.custom_deepseek_v41_index_threshold_gaudi2(score_buffer, pos, ratio, 1, 0)
        ids = ops.custom_deepseek_v41_index_emit_gaudi2(score_buffer, pos, compact_pool, stats, ratio, 1, 0)
        ids = torch.where(ids >= 0, ids, 2147483647).sort(-1).values
        ids = torch.where(ids < 2147483647, ids, -1)
        result = batch_packed_mla(attention_query,
                                  sw,
                                  kv,
                                  ids,
                                  page,
                                  pos,
                                  req_slots,
                                  bias,
                                  scaling,
                                  completion,
                                  completion,
                                  ratio=ratio)
        return ids, _bf16_boundary(result)

    consumer = compile_one(consume, "reindex_consumer")
    params = (compact, counts, positions, swa, main, pages, slots, sink, scale, done, *(part.outputs[0]
                                                                                        for part in plans[1:]))
    consumer(aq, *params)
    plans.append(recorder.prepare(consumer, [aq], [params]))

    def reference(q, w, cache, page, pos, candidate_pool, attn_q, sw, kv, req_slots, bias, scaling, completion):
        ids = reindex_mme_select(q, w, cache, page, pos, candidate_pool, ratio=ratio)
        result = batch_packed_mla(attn_q,
                                  sw,
                                  kv,
                                  ids,
                                  page,
                                  pos,
                                  req_slots,
                                  bias,
                                  scaling,
                                  completion,
                                  completion,
                                  ratio=ratio)
        return ids, _bf16_boundary(result)

    ref = compile_one(reference, "reindex_existing_contract")
    ref_params = (weights, packed, pages, positions, pool, aq, swa, main, slots, sink, scale, done)
    ref(query, *ref_params)
    reference_plan = recorder.prepare(ref, [query], [ref_params])
    native_chain = [plan.native for plan in plans]
    run = lambda active: recorder.module.replay_bounded_tiles(native_chain, active)
    report["recipes_per_part"] = [plan.recipes for plan in plans]
    graph_files = lambda: sorted(str(x) for x in (args.output.parent / ".graph_dumps").glob("*-PostGraph*"))
    report["compiled_graphs_before_changes"] = graph_files()
    report["allocated_bytes_after_prepare"] = torch.hpu.memory_allocated()
    # No timing of the old full-model baseline or missing concurrency tiers.
    for visible in (8193, 531, 511, 2048, 2049, 16383, 0, 531):
        lengths = [max(0, visible - r * 3) for r in range(batch)]
        host_pos = [length * ratio - 1 for length in lengths]
        host_pool = torch.full((batch, 2048), -1, dtype=torch.int32)
        for request, length in enumerate(lengths):
            block_ids = torch.randperm((length + 7) // 8)[:2048].int()
            host_pool[request, :len(block_ids)] = block_ids
        host_pool = host_pool[:, torch.randperm(2048)].contiguous()
        host_pool[:, 17:23] = -1
        pool.copy_(host_pool)
        positions.copy_(torch.tensor(host_pos, dtype=torch.int32))
        query.copy_(unpack_fp4(pack_fp4(torch.randn(batch, 32, 128).bfloat16(), 32), 128, 32))
        expected = [x.cpu() for x in ref(query, *ref_params)]
        active = unique_pool_tile_bound(host_pos, ratio)
        before = [part.info()[1] for part in native_chain]
        run(active)
        torch.hpu.synchronize()
        actual = [x.cpu() for x in plans[-1].outputs]
        after = [part.info()[1] for part in native_chain]
        exact = [torch.equal(a, b) for a, b in zip(actual, expected, strict=True)]
        increments = [b - a for a, b in zip(before, after, strict=True)]
        expected_increments = [1] + [int(i < active) for i in range(8)] + [1]
        case = {
            "visible": visible,
            "active_tiles": active,
            "exact": exact,
            "native_replay_increments": increments,
            "expected_increments": expected_increments
        }
        report["cases"].append(case)
        save()
        if not all(exact) or increments != expected_increments:
            torch.save(
                {
                    "actual": actual,
                    "expected": expected,
                    "pool": host_pool,
                    "positions": host_pos,
                    "counts": counts.cpu(),
                    "tile_scores": [x.outputs[0].cpu() for x in plans[1:9]]
                }, args.output.with_suffix(".failure.pt"))
            raise RuntimeError(f"Bounded Reindex/consumer contract failed: {case}")
        # Same native chain at eight tiles is a missing local mechanism
        # reference, not the retained monolithic production performance.
        for name, active_count in (("retained_algorithm_native_component", None), ("same_chain_all_tiles", 8),
                                   ("bounded", active)):
            invoke = reference_plan if active_count is None else lambda n=active_count: run(n)
            for _ in range(3):
                invoke()
            torch.hpu.synchronize()
            samples = []
            for _ in range(args.samples):
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                wall = time.perf_counter()
                begin.record()
                invoke()
                end.record()
                end.synchronize()
                samples.append({"device_ms": begin.elapsed_time(end), "wall_ms": (time.perf_counter() - wall) * 1000})
            case[name] = {
                "samples": samples,
                "median_device_ms": statistics.median(s["device_ms"] for s in samples),
                "median_wall_ms": statistics.median(s["wall_ms"] for s in samples)
            }
        save()
        print(json.dumps({
            "visible": visible,
            "tiles": active,
            "exact": exact,
            "reference_ms": case["retained_algorithm_native_component"]["median_device_ms"],
            "bounded_ms": case["bounded"]["median_device_ms"]
        }),
              flush=True)
    report["mechanism_passed"] = True
    report["native_info"] = [part.info() for part in native_chain]
    report["reference_native_info"] = reference_plan.native.info()
    report["compiled_graphs_after_changes"] = graph_files()
    report["new_graphs_during_changes"] = sorted(
        set(report["compiled_graphs_after_changes"]) - set(report["compiled_graphs_before_changes"]))
    report["peak_allocated_bytes"] = torch.hpu.max_memory_allocated()
    reference_plan.close()
    for plan in reversed(plans):
        plan.close()
    torch.distributed.destroy_process_group()
    save()


if __name__ == "__main__":
    main()
