# SPDX-License-Identifier: Apache-2.0
"""Measure runtime CSA2 selection through its actual MME MLA consumer."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capacity", type=int, default=32768)
    p.add_argument("--visible", type=int, default=1025)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--native-replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
    from vllm_gaudi.ops.deepseek_v41_paged_attention import _selected_attention_layout, PagedCSA2Attention

    torch.ops.load_library(str(args.library))
    recorder = None
    if args.native_replay:
        from deepseek_v41_micro_replay import RecipeRecorder
        recorder = RecipeRecorder(args.output.parent)
    torch.manual_seed(25)
    capacity, visible = args.capacity, args.visible
    q = torch.randn(1, 32, 128).bfloat16().to("hpu")
    w = (torch.randn(1, 32) * .02).bfloat16().to("hpu")
    cache = torch.randint(0, 256, (capacity + 128, 68), dtype=torch.uint8)
    cache[:, 64:] = 125
    cache = cache.to("hpu")
    pages = (torch.randperm(capacity // 128).int() + 1).to("hpu")
    pos = torch.tensor([visible - 1], dtype=torch.int32, device="hpu")
    candidates = torch.full((1, 2048), -1, dtype=torch.int32, device="hpu")
    main_kv = torch.randint(0, 256, (capacity + 128, 288), dtype=torch.uint8)
    main_kv[:, 256:] = 56
    main_kv = main_kv.to("hpu")
    swa = torch.randint(0, 127, (256, 528), dtype=torch.uint8)
    swa[:, 512:] = 124
    swa = swa.to("hpu")
    attn_q = torch.randn(1, 32, 512).bfloat16().to("hpu")
    sink = torch.zeros(32, device="hpu")
    scale = torch.tensor([512**-.5], device="hpu")
    offsets = torch.arange(512, dtype=torch.int32, device="hpu")
    swa_offsets = torch.arange(256, dtype=torch.int32, device="hpu")
    window_offsets = torch.arange(128, dtype=torch.int32, device="hpu")
    search = max(512, 1 << (visible - 1).bit_length())
    rows = torch.arange(search, dtype=torch.int32, device="hpu")

    def reference(query, weights, packed, table, position):
        if visible <= 512:
            return torch.where(offsets[None] <= position[:, None], offsets[None], -1)
        best, selected = None, None
        for start in range(0, search, 2048):
            current = rows[start:start + 2048]
            physical = table[(current // 128).long()] * 128 + current % 128
            keys = unpack_fp4(packed[physical.long()], 128, 32)
            scores = (torch.einsum("thd,nd->thn", query, keys).relu() * weights[:, :, None])
            scores = scores.reshape(1, 2, 16, -1).sum(2).sum(1).float()
            scores = scores.masked_fill(current[None] > position[:, None], -torch.inf)
            best, selected = PagedCSA2Attention._merge_topk(best, selected, scores, current, 512)
        return selected.sort(-1).values.int()

    def chain(query, weights, packed, table, position, pool, main, swa, attention_query):
        if args.reference:
            selected = reference(query, weights, packed, table, position)
        else:
            selected, _ = runtime_index_select(query, weights, packed, table, position, pool,
                                               ratio=1, capacity=capacity)
        safe = selected.clamp_min(0)
        physical = table[(safe.flatten() // 128).long()].reshape(safe.shape) * 128 + safe % 128
        window = position[:, None] - 127 + window_offsets[None]
        window = torch.where(window >= 0, window % 256, -1).int()
        row_ids, indices = _selected_attention_layout(physical, selected, window, swa_offsets, offsets)
        lengths = torch.full((1,), 640, dtype=torch.int32, device=query.device)
        result = torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2(
            attention_query, swa, main, row_ids, indices, sink, scale, lengths)
        return result, selected

    fixed = (q, w, cache, pages, pos, candidates, main_kv, swa, attn_q)
    compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    start = time.perf_counter()
    replay = None
    if recorder is not None:
        replay = recorder.prepare(compiled, [q], [fixed[1:]])
    for _ in range(3):
        if replay is None:
            result = compiled(*fixed)
        else:
            replay()
            result = replay.outputs
    torch.hpu.synchronize()
    warm_ms = (time.perf_counter() - start) * 1000
    before = dict(metric_global("graph_compilation").stats())
    wall, device = [], []
    for repeat in range(24):
        pos.copy_(torch.tensor([visible - 1 - repeat % 2], dtype=torch.int32))
        torch.hpu.synchronize()
        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        if replay is None:
            result = compiled(*fixed)
        else:
            replay()
            result = replay.outputs
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        device.append(begin.elapsed_time(end))
    report = dict(capacity=capacity, visible=visible, reference=args.reference,
                  native_replay=args.native_replay, recipes=None if replay is None else replay.recipes, warm_ms=warm_ms,
                  wall_ms=wall, device_ms=device, wall_median_ms=statistics.median(wall),
                  device_median_ms=statistics.median(device), output_finite=bool(result[0].cpu().isfinite().all()),
                  compilation_before=before, compilation_after=dict(metric_global("graph_compilation").stats()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if replay is not None:
        replay.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
