# SPDX-License-Identifier: Apache-2.0
"""Complete C2-C6 logical MLA vs static union codec/QK/softmax/PV.

Synthetic overlap sweep, not production selected-row statistics or TPOT.
Includes all union metadata, packed-row movement, decoding and consumers.
"""
import argparse
import json
import os
from pathlib import Path
import time


def vector_attention(q, swa, main, selection, positions, pages, sink, scale, lengths, ratio):
    import torch
    width = 128 // ratio
    absolute = positions.unsqueeze(-1) - 127 + torch.arange(128, dtype=torch.int32, device=q.device)
    logical = selection.clamp_min(0)
    page_ids = torch.div(logical, width, rounding_mode="floor")
    valid_page = (selection >= 0) & (page_ids < pages.numel())
    physical = (
        pages.index_select(0,
                           page_ids.clamp_max(pages.numel() - 1).reshape(-1).long()).reshape_as(selection) * width +
        (logical & (width - 1))).clamp_min(0)
    physical = torch.where(valid_page & (physical < main.shape[0]), physical + 256, -1)
    swa_ids = torch.where(absolute >= 0, absolute & 255, -1)
    row_ids = torch.cat((swa_ids, physical), -1).reshape(1, -1).int().contiguous()
    local = torch.cat((absolute >= 0, selection >= 0), -1)
    indices = torch.arange(q.shape[0] * 640, dtype=torch.int32, device=q.device).reshape(q.shape[0], 640)
    indices = torch.where(local, indices, -1).contiguous()
    return torch.ops.custom_op.custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2(
        q, swa, main, row_ids, indices, sink, scale, lengths)


def union_attention(q, swa, main, selection, positions, pages, sink, scale, lengths, ratio):
    import torch
    width = 128 // ratio
    logical = selection.reshape(-1)
    page_ids = torch.div(logical.clamp_min(0), width, rounding_mode="floor")
    page_valid = (logical >= 0) & (page_ids < pages.numel())
    physical = (pages.index_select(0,
                                   page_ids.clamp_max(pages.numel() - 1).long()) * width + (logical.clamp_min(0) &
                                                                                            (width - 1))).clamp_min(0)
    physical = torch.where(page_valid & (physical < main.shape[0]), physical, 2147483647)
    ordered, order = physical.sort()
    first = torch.cat((torch.ones(1, dtype=torch.bool, device=q.device), ordered[1:] != ordered[:-1]))
    ordinal = first.int().cumsum(0, dtype=torch.int32) - 1
    # Equal scatter destinations always have equal values. Shape stays fixed
    # even though the number of unique experts/rows is device data.
    union = torch.full_like(ordered, 2147483647).scatter(0, ordinal.long(), ordered)
    inverse = torch.empty_like(order).scatter(0, order, ordinal.long()).reshape_as(selection)
    packed = main.index_select(0, union.clamp_max(main.shape[0] - 1).long())
    packed = torch.where((union != 2147483647).unsqueeze(-1), packed, 0).contiguous()
    decoded = torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(packed, pages, union, 0)
    swa_rows = torch.arange(256, dtype=torch.int32, device=q.device).reshape(1, -1)
    window, _ = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2(swa, swa[:1].contiguous(), swa_rows,
                                                                                True)
    cache = torch.cat((window, decoded), 0)
    absolute = positions.unsqueeze(-1) - 127 + torch.arange(128, dtype=torch.int32, device=q.device)
    swa_ids = torch.where(absolute >= 0, absolute & 255, -1)
    # Invalid physical rows are zero KV but remain locally selected, matching
    # logical MLA's independent address validity and softmax mask contract.
    main_ids = torch.where(selection >= 0, inverse + 256, -1)
    ids = torch.cat((swa_ids, main_ids), -1).int().contiguous()
    return torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(q, cache, ids, sink, scale, lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--warm", type=int, default=32)
    parser.add_argument("--vector-gather", action="store_true")
    parser.add_argument("--native-vector", action="store_true")
    parser.add_argument("--native-union", action="store_true")
    parser.add_argument("--native-slotmap", action="store_true")
    parser.add_argument("--native-replay", action="store_true")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--edge-cases", action="store_true")
    parser.add_argument("--shared-main",
                        action="store_true",
                        help="Four layer consumers with identical KV/selection ownership")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG",
                                                              "").replace("{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    if args.native_union or args.native_slotmap:
        torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    bind_worker_helpers(rank)
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="running",
                  scope="synthetic selected-row overlap, complete MLA consumer",
                  cases=[],
                  end_to_end_qualified=False,
                  vector_gather=args.vector_gather,
                  native_vector=args.native_vector,
                  native_union=args.native_union,
                  native_slotmap=args.native_slotmap,
                  shared_main=args.shared_main)
    recorder = None
    if args.native_replay:
        from deepseek_v41_micro_replay import RecipeRecorder

        recorder = RecipeRecorder(root)
    try:
        with torch.inference_mode():
            torch.manual_seed(10306)
            q = torch.randn(args.tokens, args.heads, 512).bfloat16().to("hpu")
            if os.environ.get("GRAPH_VISUALIZATION") == "1":
                from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

                torch.hpu.synchronize()
                configure_post_graph_directory(root / "graphs" / "rank0")
            swa = pack_swa(torch.randn(256, 512).bfloat16()).to("hpu")
            main_cache = pack_fp4(torch.randn(32768, 512).bfloat16(), 16).to("hpu")
            layer_queries = torch.stack([q + i * .0625 for i in range(4)])
            layer_swa = torch.stack([pack_swa(torch.randn(256, 512).bfloat16()).to("hpu") for _ in range(4)])
            sink = torch.randn(args.heads).float().to("hpu")
            scale = torch.tensor([512**-.5], device="hpu")
            lengths = torch.full((args.tokens, ), 640, dtype=torch.int32, device="hpu")
            for ratio in (1, 2):
                pages = torch.arange(256 * ratio, dtype=torch.int32).to("hpu")
                positions = torch.arange(16384, 16384 + args.tokens, dtype=torch.int32).to("hpu")
                for common in (0, 384, 512):
                    rows = torch.arange(512).int().repeat(args.tokens, 1)
                    for token in range(args.tokens):
                        rows[token, common:] += token * (512 - common)
                    rows[:, -3:] = -1
                    rows[0, -4] = 65536  # Valid selection of an invalid address remains zero/mask=1.
                    if args.edge_cases:
                        rows[:, 10] = rows[:, 9]  # Within-query duplicates.
                        rows[:, 20], rows[:, 21] = rows[:, 21].clone(), rows[:, 20].clone()
                        rows[:, 30] = -1  # An invalid hole, not only an invalid tail.
                        positions = torch.arange(-1, args.tokens - 1, dtype=torch.int32).to("hpu")
                        lengths = torch.tensor([0, 64, 128, 129, 639, 640][:args.tokens],
                                               dtype=torch.int32, device="hpu")
                        pages[0] = -1
                    rows = rows.to("hpu")
                    functions = []
                    for arm in (0, 1):

                        def body(q,
                                 swa,
                                 main_cache,
                                 rows,
                                 positions,
                                 pages,
                                 sink,
                                 scale,
                                 lengths,
                                 arm=arm,
                                 ratio=ratio):
                            if args.shared_main:
                                values, shared_rows, shared_mask = [], None, None
                                for layer in range(4):
                                    query, window = q[layer].contiguous(), swa[layer].contiguous()
                                    if arm and layer:
                                        output = torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2(
                                            query, window, shared_rows, shared_mask, positions, sink, scale, lengths)
                                    elif arm:
                                        output, shared_rows, shared_mask = (
                                            torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(
                                                query, window, main_cache, rows, positions, pages, sink, scale, lengths,
                                                ratio))
                                    else:
                                        output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                                            query, window, main_cache, rows, positions, pages, sink, scale, lengths,
                                            ratio, True)
                                    values.append(output)
                                return torch.cat(values, 0)
                            if arm:
                                if args.native_slotmap:
                                    return torch.ops.custom_op.custom_deepseek_v41_logical_mla_slotmap_gaudi2(
                                        q, swa, main_cache, rows, positions, pages, sink, scale, lengths, ratio, True)
                                if args.native_union:
                                    return torch.ops.custom_op.custom_deepseek_v41_logical_mla_union_gaudi2(
                                        q, swa, main_cache, rows, positions, pages, sink, scale, lengths, ratio, True)
                                if args.native_vector:
                                    return torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                                        q, swa, main_cache, rows, positions, pages, sink, scale, lengths, ratio, True)
                                fn = vector_attention if args.vector_gather else union_attention
                                return fn(q, swa, main_cache, rows, positions, pages, sink, scale, lengths, ratio)
                            return torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                                q, swa, main_cache, rows, positions, pages, sink, scale, lengths, ratio,
                                args.native_union or args.native_slotmap)

                        functions.append(torch.compile(body, backend="hpu_backend", fullgraph=True, dynamic=False))
                    query_input = layer_queries if args.shared_main else q
                    swa_input = layer_swa if args.shared_main else swa
                    operands = query_input, swa_input, main_cache, rows, positions, pages, sink, scale, lengths
                    for _ in range(2):
                        result = [fn(*operands).cpu() for fn in functions]
                        if not torch.equal(result[0], result[1]):
                            raise AssertionError(
                                dict(ratio=ratio,
                                     common=common,
                                     different=int((result[0] != result[1]).sum()),
                                     max_abs=float((result[0].float() - result[1].float()).abs().max())))
                        query_input.add_(.0625)
                    for fn in functions:
                        for _ in range(args.warm):
                            fn(*operands)
                    torch.hpu.synchronize()
                    replays = None
                    if recorder is not None:
                        replays = [recorder.prepare(fn, [operands[0]], [tuple(operands[1:])]) for fn in functions]
                        for function, replay in zip(functions, replays, strict=True):
                            expected = function(*operands).cpu()
                            replay()
                            torch.hpu.synchronize()
                            actual = replay.outputs[0].cpu()
                            if not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
                                raise AssertionError("Native replay differs from its compiled MLA arm")
                    timings = []
                    for arm in (0, 1, 0, 1, 0, 1):
                        start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        start.record()
                        wall = time.perf_counter_ns()
                        for _ in range(args.steps):
                            if replays is None:
                                functions[arm](*operands)
                            else:
                                replays[arm]()
                        stop.record()
                        stop.synchronize()
                        timings.append(
                            dict(arm=arm,
                                 device_ms=start.elapsed_time(stop) / args.steps,
                                 host_ms=(time.perf_counter_ns() - wall) / 1e6 / args.steps))
                    unique = int(torch.unique(rows.cpu()[rows.cpu() >= 0]).numel())
                    report["cases"].append(
                        dict(ratio=ratio,
                             common_rows=common,
                             unique_rows=unique,
                             exact_two_inputs=True,
                             timings=timings))
                    torch.hpu.synchronize()
                    if replays is not None:
                        for replay in replays:
                            replay.close()
            report.update(status="passed")
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / "mla-union.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
