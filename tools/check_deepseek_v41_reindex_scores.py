# SPDX-License-Identifier: Apache-2.0
"""Compare the current C6 Reindex tile with the existing fused paged-key gather."""
import argparse
import json
import os
from pathlib import Path
import time


def score(query, weights, packed, pages, positions, rows, native_keys, mirror=None,
          native_reduce=False, ratio=1, local_heads=8):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
    if mirror is not None:
        from vllm_gaudi.ops.deepseek_v41_index_mirror import per_query_mirror_keys
        keys = per_query_mirror_keys(mirror, rows)
    elif native_keys:
        keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(packed, pages, rows, ratio)
    else:
        safe = rows.clamp_min(0)
        width = 128 // ratio
        physical = pages[(safe.flatten() // width).long()].reshape(safe.shape) * width + (safe % width)
        selected = packed.index_select(0, physical.flatten().long()).reshape(*physical.shape, 68)
        keys = unpack_fp4(selected, 128, 32)
    products = torch.bmm(query.contiguous(), keys.transpose(1, 2).contiguous())
    if native_reduce:
        return torch.ops.custom_op.custom_deepseek_v41_prefill_index_reduce_gaudi2(products.contiguous(),
                                                                                   weights.contiguous(),
                                                                                   positions.contiguous(),
                                                                                   rows.contiguous(), ratio,
                                                                                   local_heads)
    products = products.relu() * weights.unsqueeze(-1)
    reduced = products.reshape(query.shape[0], 32 // local_heads, local_heads, -1).sum(2).sum(1)
    visible = ((positions + 1) // ratio).unsqueeze(-1)
    return reduced.float().masked_fill((rows < 0) | (rows >= visible), -torch.inf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-keys", action="store_true")
    parser.add_argument("--native-reduction", action="store_true",
                        help="Both arms use the qualified mirror; fuse only head reduction/mask and consume TopK")
    parser.add_argument("--wide-reduction", action="store_true",
                        help="Compare eight native score tiles with one full per-query BMM/reduction")
    parser.add_argument("--streamed-selection", action="store_true",
                        help="Use the maintained per-tile TopK and sequential merge consumer")
    parser.add_argument("--local-heads", type=int, choices=(8, 16), default=8)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=1)
    opts = parser.parse_args()
    if opts.native_reduction and not opts.mirror_keys:
        parser.error("Native reduction comparison requires --mirror-keys in both arms")
    if opts.wide_reduction and not opts.native_reduction:
        parser.error("Wide reduction requires the qualified native reduction parent")
    if opts.streamed_selection and not opts.native_reduction:
        parser.error("Streamed selection comparison requires --native-reduction")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG",
                                                              "").replace("{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_helpers(rank)
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank,
                  status="running",
                  cases=[],
                  formal_qualification=False,
                  scope="C6 per-query paged Reindex score tile, not full round or model TPOT")
    try:
        with torch.inference_mode():
            torch.manual_seed(9306)
            query = torch.randn(6, 32, 128, dtype=torch.bfloat16).to("hpu")
            weights = (torch.randn(6, 32) * .02).bfloat16().to("hpu")
            source = torch.randn(32768 + 128, 128, dtype=torch.bfloat16)
            packed = pack_fp4(source, 32).contiguous().to("hpu")
            width = 128 // opts.ratio
            pages = torch.arange(1, 32768 // width + 1, dtype=torch.int32).to("hpu")
            positions = torch.arange(16384, 16390, dtype=torch.int32).to("hpu")
            all_rows = (torch.arange(6 * 16384).reshape(6, 16384) * 13 % 32768).int()
            all_rows[:, ::113] = -1
            all_rows = all_rows.to("hpu")
            logical = torch.arange(32768, dtype=torch.int32, device="hpu")
            physical = pages[(logical // width).long()] * width + (logical % width)
            mirror = None
            if opts.mirror_keys:
                from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
                mirror = unpack_fp4(packed.index_select(0, physical.long()), 128, 32).contiguous()
                # The native decoder normalizes both FP4 zero encodings.
                mirror = torch.where(mirror == 0, 0, mirror)
            updated_rows = positions // opts.ratio
            updated_slots = pages[(updated_rows // width).long()] * width + (updated_rows % width)
            updates = packed.index_select(0, updated_slots.long()).clone()
            functions = []
            for arm in (0, 1):

                def body(query, weights, packed, pages, positions, rows, mirror, updates, arm=arm):
                    packed.index_copy_(0, updated_slots.long(), updates)
                    selected_mirror = mirror if opts.mirror_keys and (arm or opts.native_reduction) else None
                    if selected_mirror is not None:
                        from vllm_gaudi.ops.deepseek_v41_index_mirror import refresh_index_mirror_rows
                        refresh_index_mirror_rows(packed, pages, updated_rows, mirror, opts.ratio)
                    if opts.streamed_selection:
                        from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
                        wide = None
                        if opts.wide_reduction and arm:
                            wide = score(query, weights, packed, pages, positions, rows, True, selected_mirror,
                                         native_reduce=True, ratio=opts.ratio, local_heads=opts.local_heads)
                        best_scores = best_rows = None
                        pieces = []
                        for start in range(0, rows.shape[-1], 2048):
                            current_rows = rows[:, start:start + 2048]
                            current_scores = (wide[:, start:start + 2048] if wide is not None else
                                              score(query, weights, packed, pages, positions,
                                                    current_rows.contiguous(), True, selected_mirror,
                                                    native_reduce=True, ratio=opts.ratio,
                                                    local_heads=opts.local_heads))
                            best_scores, best_rows = PagedCSA2Attention._merge_topk(
                                best_scores, best_rows, current_scores, current_rows, 512)
                            pieces.append(current_scores)
                        return torch.cat(pieces, -1), best_rows
                    if opts.wide_reduction and arm:
                        scores = score(query, weights, packed, pages, positions, rows, True, selected_mirror,
                                       native_reduce=True, ratio=opts.ratio, local_heads=opts.local_heads)
                        return scores, scores.topk(512, dim=-1, sorted=False).indices
                    scores = torch.cat([
                        score(query,
                              weights,
                              packed,
                              pages,
                              positions,
                              rows[:, start:start + 2048].contiguous(),
                              bool(arm) or opts.mirror_keys,
                              selected_mirror,
                              native_reduce=opts.native_reduction and (bool(arm) or opts.wide_reduction),
                              ratio=opts.ratio,
                              local_heads=opts.local_heads) for start in range(0, rows.shape[1], 2048)
                    ], -1)
                    if opts.native_reduction:
                        return scores, scores.topk(512, dim=-1, sorted=False).indices
                    return scores

                functions.append(torch.compile(body, backend="hpu_backend", fullgraph=True, dynamic=False))
            args = (query, weights, packed, pages, positions, all_rows, mirror, updates)
            for _ in range(2):
                values = [fn(*args) for fn in functions]
                values = [tuple(v.cpu() for v in out) if isinstance(out, tuple) else (out.cpu(), ) for out in values]
                for a, b in zip(*values, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
                    if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                        raise AssertionError("Reindex scores/TopK changed at the byte level")
                query.add_(.0625)
                weights.mul_(.5)
            for fn in functions:
                for _ in range(64):
                    fn(*args)
            torch.hpu.synchronize()
            for case in (0, 1, 0, 1, 0, 1):
                fn = functions[case]
                start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start.record()
                host = time.perf_counter_ns()
                for _ in range(64):
                    fn(*args)
                stop.record()
                stop.synchronize()
                report["cases"].append(
                    dict(native_keys=bool(case),
                         mirror_keys=opts.mirror_keys and (bool(case) or opts.native_reduction),
                         native_reduction=opts.native_reduction and (bool(case) or opts.wide_reduction),
                         wide_reduction=opts.wide_reduction and bool(case),
                         local_heads=opts.local_heads,
                         ratio=opts.ratio,
                         rows=6,
                         candidate_columns=16384,
                         tiles=8,
                         device_ms=start.elapsed_time(stop) / 64,
                         host_ms=(time.perf_counter_ns() - host) / 1e6 / 64))
            report.update(status="passed", exact_two_input_versions=True, byte_exact=True,
                          endpoint="scores and streamed TopK consumer" if opts.streamed_selection else
                          "scores and TopK consumer" if opts.native_reduction else "scores")
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / f"reindex-scores-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
