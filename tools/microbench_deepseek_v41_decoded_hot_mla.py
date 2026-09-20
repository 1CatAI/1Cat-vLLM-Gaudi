# SPDX-License-Identifier: Apache-2.0
"""Compare packed-paged and decoded-hot C1 MLA through the real consumer.

Both arms consume the same checkpoint-format SWA/FP4 roundtrip values and the
same native paged prefix layout.  Timing ends after the BF16 MLA result is
materialized and synchronized; this is a component result, not an E2E claim.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401,E402
import torch

from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa


def sync() -> None:
    torch.hpu.synchronize()


def stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "repeats": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": ordered[len(ordered) // 2],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .9))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": ordered,
    }


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    return {
        "bf16_mismatches": int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
        "max_absolute_error": float(difference.max()),
        "rmse": float(difference.square().mean().sqrt()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=1)
    parser.add_argument("--position", type=int, default=2047)
    parser.add_argument("--main-rows", type=int, default=2560)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--prefix-comparison-only", action="store_true",
                        help="Only compare general versus indices-only prefix metadata through MLA")
    args = parser.parse_args()
    if args.position // args.ratio >= args.main_rows:
        raise ValueError("main mirror must cover the visible logical compressed row")
    library = os.environ.get("VLLM_HPU_DSV4_TPC_OP_LIBRARY")
    if not library:
        raise RuntimeError("VLLM_HPU_DSV4_TPC_OP_LIBRARY is required")
    torch.ops.load_library(library)
    torch.manual_seed(20260919 + args.ratio)

    # Quantize once on CPU so both HPU arms consume identical canonical bytes.
    host_swa_packed = pack_swa(torch.randn(256, 512, dtype=torch.bfloat16))
    host_main_packed = pack_fp4(torch.randn(args.main_rows, 512, dtype=torch.bfloat16), 16)
    host_swa_decoded = torch.zeros(512, 512, dtype=torch.bfloat16)
    host_swa_decoded[:256].copy_(unpack_swa(host_swa_packed))
    host_main_decoded = unpack_fp4(host_main_packed, 512, 16)

    visible_rows = (args.position + 1) // args.ratio
    selected = ((torch.arange(512, dtype=torch.int32) * 37 + 13) % visible_rows).reshape(1, 512)
    host_selected = selected.clone()
    host_selected_cache = torch.cat((host_swa_decoded[:256],
                                     host_main_decoded.index_select(0, selected.flatten().long())), 0)
    window = torch.arange(args.position - 127, args.position + 1,
                          dtype=torch.int32).remainder(256).reshape(1, 128)
    # Identity page assignment: the packed row list contains the selected
    # physical rows compactly, whereas decoded-hot indices address logical
    # main rows directly after the 256-row SWA ring.
    row_ids = torch.cat((torch.arange(256, dtype=torch.int32),
                         host_selected.flatten() + 256)).reshape(1, 768)
    packed_indices = torch.cat((window,
                                torch.arange(256, 768, dtype=torch.int32).reshape(1, 512)), -1)
    decoded_indices = torch.cat((window, host_selected + 256), -1)
    lengths = torch.tensor([640], dtype=torch.int32)
    position = torch.tensor([args.position], dtype=torch.int32)
    page_width = 128 // args.ratio
    block_table = torch.arange((args.main_rows + page_width - 1) // page_width,
                               dtype=torch.int32)
    q = torch.randn(1, 32, 512, dtype=torch.bfloat16)
    sink = torch.randn(32, dtype=torch.float32)
    scale = torch.tensor([512.0**-.5], dtype=torch.float32)
    swa_done = torch.zeros(16, dtype=torch.int32)
    main_done = torch.zeros(36, dtype=torch.int32)

    tensors = [q, host_swa_packed, host_main_packed, host_swa_decoded,
               host_main_decoded, host_selected_cache, row_ids, packed_indices,
               decoded_indices, lengths, host_selected, position, block_table,
               sink, scale, swa_done, main_done]
    q, swa_packed, main_packed, swa_decoded, main_decoded, selected_cache, row_ids, packed_indices, decoded_indices, lengths, selected, position, block_table, sink, scale, swa_done, main_done = [
        value.contiguous().to("hpu") for value in tensors
    ]
    layout = (torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r1_i32_gaudi2
              if args.ratio == 1 else
              torch.ops.custom_op.custom_deepseek_v41_prefix_layout_r2_i32_gaudi2)

    def packed(q, swa, main, rows, indices, lengths, sink, scale):
        return torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2(
            q, swa, main, rows, indices, sink, scale, lengths)

    def decoded(q, swa, main, indices, lengths, sink, scale, swa_done, main_done):
        return torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
            q, swa, main, indices, sink, scale, lengths, swa_done, main_done,
            0, args.main_rows, 256)

    def decoded_with_layout(q, swa, main, selected, position, blocks,
                            sink, scale, swa_done, main_done):
        _, indices, lengths = layout(selected, position, blocks)
        return torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
            q, swa, main, indices, sink, scale, lengths, swa_done, main_done,
            0, args.main_rows, 256)

    def decoded_with_indices(q, swa, main, selected, position,
                             sink, scale, swa_done, main_done):
        indices, lengths = torch.ops.custom_op.custom_deepseek_v41_prefix_indices_i32_gaudi2(
            selected, position)
        return torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
            q, swa, main, indices, sink, scale, lengths, swa_done, main_done,
            0, args.main_rows, 256)

    def selected_mla(q, cache, indices, lengths, sink, scale):
        return torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(
            q, cache, indices, sink, scale, lengths)

    packed_graph = torch.compile(packed, backend="hpu_backend", fullgraph=True, dynamic=False)
    decoded_graph = torch.compile(decoded, backend="hpu_backend", fullgraph=True, dynamic=False)
    decoded_layout_graph = torch.compile(decoded_with_layout, backend="hpu_backend", fullgraph=True, dynamic=False)
    decoded_indices_graph = torch.compile(decoded_with_indices, backend="hpu_backend", fullgraph=True, dynamic=False)
    selected_graph = torch.compile(selected_mla, backend="hpu_backend", fullgraph=True, dynamic=False)
    packed_inputs = (q, swa_packed, main_packed, row_ids, packed_indices, lengths, sink, scale)
    decoded_inputs = (q, swa_decoded, main_decoded, decoded_indices, lengths,
                      sink, scale, swa_done, main_done)
    decoded_layout_inputs = (q, swa_decoded, main_decoded, selected, position,
                             block_table, sink, scale, swa_done, main_done)
    decoded_indices_inputs = (q, swa_decoded, main_decoded, selected, position,
                              sink, scale, swa_done, main_done)
    if args.prefix_comparison_only:
        for _ in range(args.warmups):
            decoded_layout_graph(*decoded_layout_inputs)
            sync()
            decoded_indices_graph(*decoded_indices_inputs)
            sync()
        layout_actual = decoded_layout_graph(*decoded_layout_inputs).cpu()
        indices_actual = decoded_indices_graph(*decoded_indices_inputs).cpu()
        sync()
        correctness = {
            "decoded_indices_only_vs_layout": error(indices_actual, layout_actual),
            "finite": bool(indices_actual.isfinite().all()),
        }
        timings = {"decoded_hot_native_layout": [],
                   "decoded_hot_indices_only": []}
        arms = (("decoded_hot_native_layout", decoded_layout_graph, decoded_layout_inputs),
                ("decoded_hot_indices_only", decoded_indices_graph, decoded_indices_inputs))
        for iteration in range(args.repeats):
            for name, graph, inputs in (arms if iteration % 2 == 0 else reversed(arms)):
                start = time.perf_counter_ns()
                graph(*inputs)
                sync()
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
        reference = stats(timings["decoded_hot_native_layout"])
        candidate = stats(timings["decoded_hot_indices_only"])
        payload = {
            "benchmark": "dsv41_decoded_hot_prefix_metadata_through_mla",
            "scope": "selected IDs through synchronized BF16 MLA consumer",
            "ratio": args.ratio,
            "position": args.position,
            "main_rows": args.main_rows,
            "correctness": correctness,
            "decoded_hot_native_layout": reference,
            "decoded_hot_indices_only": candidate,
            "indices_only_saved_mean_ms": reference["mean_ms"] - candidate["mean_ms"],
            "indices_only_saved_p50_ms": reference["p50_ms"] - candidate["p50_ms"],
            "e2e_claim": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        return
    for _ in range(args.warmups):
        packed_graph(*packed_inputs)
        sync()
        decoded_graph(*decoded_inputs)
        sync()
        decoded_layout_graph(*decoded_layout_inputs)
        sync()
        decoded_indices_graph(*decoded_indices_inputs)
        sync()

    expected = packed_graph(*packed_inputs).cpu()
    actual = decoded_graph(*decoded_inputs).cpu()
    layout_actual = decoded_layout_graph(*decoded_layout_inputs).cpu()
    indices_actual = decoded_indices_graph(*decoded_indices_inputs).cpu()
    selected_actual = selected_graph(q, selected_cache, packed_indices, lengths, sink, scale).cpu()
    sync()
    host_full_cache = torch.cat((host_swa_decoded[:256], host_main_decoded), 0)
    chosen = host_full_cache.index_select(0, decoded_indices.cpu().flatten().long()).reshape(1, 640, 512).float()
    scores = torch.bmm(q.cpu().float(), chosen.transpose(1, 2)) * scale.cpu()
    scores = torch.cat((scores, sink.cpu().reshape(1, 32, 1)), -1).softmax(-1)[..., :-1]
    cpu_reference = torch.bmm(scores, chosen).bfloat16()
    correctness = {
        **error(actual, expected),
        "finite": bool(actual.isfinite().all()),
        "decoded_native_layout_vs_packed": error(layout_actual, expected),
        "decoded_indices_only_vs_layout": error(indices_actual, layout_actual),
        "packed_vs_selected_mla": error(expected, selected_actual),
        "decoded_vs_selected_mla": error(actual, selected_actual),
        "packed_vs_cpu_reference": error(expected, cpu_reference),
        "decoded_vs_cpu_reference": error(actual, cpu_reference),
        "selected_vs_cpu_reference": error(selected_actual, cpu_reference),
    }
    if correctness["bf16_mismatches"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"expected": expected, "actual": actual}, args.output.with_suffix(".mismatch.pt"))

    timings: dict[str, list[float]] = {"packed_paged": [], "decoded_hot": [],
                                      "decoded_hot_native_layout": [],
                                      "decoded_hot_indices_only": []}
    # Alternate order to reduce drift; each sample includes the first real
    # consumer completion and host synchronization.
    arms = (("packed_paged", packed_graph, packed_inputs),
            ("decoded_hot", decoded_graph, decoded_inputs),
            ("decoded_hot_native_layout", decoded_layout_graph, decoded_layout_inputs),
            ("decoded_hot_indices_only", decoded_indices_graph, decoded_indices_inputs))
    for iteration in range(args.repeats):
        for name, graph, inputs in (arms if iteration % 2 == 0 else reversed(arms)):
            start = time.perf_counter_ns()
            graph(*inputs)
            sync()
            timings[name].append((time.perf_counter_ns() - start) / 1e6)

    packed_stats, decoded_stats = stats(timings["packed_paged"]), stats(timings["decoded_hot"])
    decoded_layout_stats = stats(timings["decoded_hot_native_layout"])
    decoded_indices_stats = stats(timings["decoded_hot_indices_only"])
    payload = {
        "benchmark": "dsv41_c1_paged_vs_decoded_hot_complete_mla",
        "scope": "selected KV supply + QK + softmax + PV + BF16 output; metadata already materialized",
        "ratio": args.ratio,
        "position": args.position,
        "main_rows": args.main_rows,
        "selected_rows": 512,
        "correctness": correctness,
        "packed_paged": packed_stats,
        "decoded_hot": decoded_stats,
        "decoded_hot_native_layout": decoded_layout_stats,
        "decoded_hot_indices_only": decoded_indices_stats,
        "p50_delta_ms": decoded_stats["p50_ms"] - packed_stats["p50_ms"],
        "p50_speedup": packed_stats["p50_ms"] / decoded_stats["p50_ms"],
        "native_layout_p50_delta_ms": decoded_layout_stats["p50_ms"] - packed_stats["p50_ms"],
        "native_layout_p50_speedup": packed_stats["p50_ms"] / decoded_layout_stats["p50_ms"],
        "indices_only_saved_p50_ms": decoded_layout_stats["p50_ms"] - decoded_indices_stats["p50_ms"],
        "e2e_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
