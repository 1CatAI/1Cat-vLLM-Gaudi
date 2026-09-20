# SPDX-License-Identifier: Apache-2.0
"""A/B exact CSA2 hot-prefix selection chains on real HPU.

The benchmark models the five ratio-1 owners used by V4.1 decode: one Full
owner publishes candidate blocks and four Reindex owners consume them.  It is
a component result and makes no end-to-end claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401,E402
import torch

from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select


def score(query, weights, decoded, positions):
    values = torch.einsum("bhd,nd->bhn", query, decoded)
    terms = (values.to(torch.bfloat16).relu() * weights.unsqueeze(-1)).to(torch.bfloat16)
    values = terms.reshape(query.shape[0], 2, 16, -1).sum(2, dtype=torch.bfloat16)
    values = values.sum(1, dtype=torch.bfloat16).float()
    rows = torch.arange(decoded.shape[0], dtype=torch.int32, device=query.device)
    return values.masked_fill(rows.unsqueeze(0) >= positions.unsqueeze(-1) + 1,
                              -torch.inf).contiguous()


def split_select(values, positions, candidates, reindex, blocks):
    stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
        values, positions, 1, reindex, blocks)
    return torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
        values, positions, candidates, stats, 1, reindex, blocks)


def build(direct_hot: bool, skip_block_publish: bool = False):
    def chain(queries, weights, decoded, positions, candidates):
        scores = [score(queries[i], weights[i], decoded[i], positions) for i in range(5)]
        selected = [split_select(scores[0], positions, candidates, 0, 0)]
        # At <16K visible rows the Full candidate list is the causal block
        # prefix.  Retain its exact output in this benchmark even when the
        # Reindex selectors take the equivalent identity mapping.
        if skip_block_publish:
            # Hot Reindex consumers below 16K use the causal identity prefix,
            # so no downstream consumer observes the published block tensor.
            blocks = candidates
        else:
            placeholders = torch.empty((positions.shape[0], 320), dtype=torch.float32,
                                       device=positions.device)
            blocks = split_select(placeholders, positions, candidates, 0, 1)
        for layer in range(1, 5):
            selected.append(split_select(scores[layer], positions, blocks,
                                         0 if direct_hot else 1, 0))
        return torch.cat(selected, 0), blocks[:, :320]

    return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)


def build_runtime_hot():
    """Exercise the production runtime_index_select hot-bucket branch."""

    def chain(queries, weights, decoded, positions, candidates):
        cache = torch.empty((2, 128, 68), dtype=torch.uint8, device=positions.device)
        pages = torch.zeros((1,), dtype=torch.int32, device=positions.device)
        selected = []
        for layer in range(5):
            rows, _ = runtime_index_select(
                queries[layer], weights[layer], cache, pages, positions, candidates,
                ratio=1, capacity=1048576, reindex=layer > 0,
                publish_candidates=layer == 0, decoded_hot=decoded[layer])
            selected.append(rows)
        return torch.cat(selected, 0), candidates[:, :320]

    return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)


def build_native_reduce():
    """Replace the generic post-MME expression with one BF16 TPC reducer."""

    def chain(queries, weights, decoded, positions, candidates):
        selected = []
        for layer in range(5):
            raw = torch.einsum("bhd,nd->bhn", queries[layer], decoded[layer]).contiguous()
            scores = torch.ops.custom_op.custom_deepseek_v41_index_reduce_bf16_gaudi2(
                raw, weights[layer], positions, 1)
            stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
                scores, positions, 1, 0, 0)
            selected.append(torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
                scores, positions, candidates, stats, 1, 0, 0))
        return torch.cat(selected, 0), candidates[:, :320]

    return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)


def build_reduce_only():
    def chain(query, weights, decoded, positions):
        raw = torch.einsum("bhd,nd->bhn", query, decoded).contiguous()
        return (torch.ops.custom_op.custom_deepseek_v41_index_reduce_bf16_gaudi2(
            raw, weights, positions, 1), )

    return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)


def build_select_only():
    def chain(scores, positions, candidates):
        stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
            scores, positions, 1, 0, 0)
        return (torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
            scores, positions, candidates, stats, 1, 0, 0), )

    return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)


def measure(function, inputs, warmups, repeats):
    for _ in range(warmups):
        function(*inputs)
    torch.hpu.synchronize()
    samples = []
    result = None
    for _ in range(repeats):
        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        begin.record()
        result = function(*inputs)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    ordered = sorted(samples)
    return tuple(value.cpu() for value in result), {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[int(.9 * len(ordered))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--visible", type=int, default=2052)
    parser.add_argument("--capacity", type=int, default=2560)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=128)
    args = parser.parse_args()
    if not 513 <= args.visible <= args.capacity or args.capacity % 64:
        raise ValueError("expected an aligned hot bucket with >512 visible rows")

    import os
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    generator = torch.Generator(device="cpu").manual_seed(20260919)
    if not 1 <= args.batch <= 64:
        raise ValueError("batch must be in [1, 64]")
    queries = torch.randn((5, args.batch, 32, 128), generator=generator).bfloat16()
    weights = (torch.randn((5, args.batch, 32), generator=generator) * .02).bfloat16()
    decoded = torch.randn((5, args.capacity, 128), generator=generator).bfloat16()
    positions = torch.tensor([args.visible - 1 - (row % 7) for row in range(args.batch)],
                             dtype=torch.int32)
    candidates = torch.full((args.batch, 2048), -1, dtype=torch.int32)
    inputs = tuple(value.to("hpu") for value in (queries, weights, decoded, positions, candidates))

    reference, split = measure(build(False), inputs, args.warmups, args.repeats)
    split_direct_output, split_direct = measure(
        build(True), inputs, args.warmups, args.repeats)
    split_direct_no_publish_output, split_direct_no_publish = measure(
        build(True, True), inputs, args.warmups, args.repeats)
    runtime_hot_output, runtime_hot = measure(
        build_runtime_hot(), inputs, args.warmups, args.repeats)
    native_reduce_output, native_reduce = measure(
        build_native_reduce(), inputs, args.warmups, args.repeats)
    reduce_only_output, reduce_only = measure(
        build_reduce_only(),
        (inputs[0][0], inputs[1][0], inputs[2][0], inputs[3]),
        args.warmups, args.repeats)
    select_only_output, select_only = measure(
        build_select_only(), (reduce_only_output[0].to("hpu"), inputs[3], inputs[4]),
        args.warmups, args.repeats)
    reference_scores = score(inputs[0][0], inputs[1][0], inputs[2][0], inputs[3]).cpu()
    raw_scores = torch.einsum("bhd,nd->bhn", inputs[0][0], inputs[2][0]).contiguous()
    candidate_scores = torch.ops.custom_op.custom_deepseek_v41_index_reduce_bf16_gaudi2(
        raw_scores, inputs[1][0], inputs[3], 1).cpu()
    valid = args.visible
    result = {
        "benchmark": "dsv41_ratio1_five_layer_select",
        "visible": args.visible,
        "batch": args.batch,
        "capacity": args.capacity,
        "chain": "five MME scores, one Full block publication, four Reindex selections",
        "split_threshold_emit": split,
        "split_direct_hot": split_direct,
        "split_direct_hot_no_block_publish": split_direct_no_publish,
        "production_runtime_hot": runtime_hot,
        "native_reduce": native_reduce,
        "one_owner_mme_reduce": reduce_only,
        "one_owner_threshold_emit": select_only,
        "split_direct_selected_equal": bool(torch.equal(reference[0], split_direct_output[0])),
        "split_direct_selected_diff": int((reference[0] != split_direct_output[0]).sum()),
        "split_direct_blocks_equal": bool(torch.equal(reference[1], split_direct_output[1])),
        "split_direct_no_publish_selected_equal": bool(
            torch.equal(reference[0], split_direct_no_publish_output[0])),
        "split_direct_no_publish_selected_diff": int(
            (reference[0] != split_direct_no_publish_output[0]).sum()),
        "production_runtime_hot_selected_equal": bool(
            torch.equal(reference[0], runtime_hot_output[0])),
        "production_runtime_hot_selected_diff": int(
            (reference[0] != runtime_hot_output[0]).sum()),
        "native_reduce_selected_equal": bool(
            torch.equal(reference[0], native_reduce_output[0])),
        "native_reduce_selected_diff": int(
            (reference[0] != native_reduce_output[0]).sum()),
        "select_only_matches_native_first_owner": bool(torch.equal(
            native_reduce_output[0][:args.batch], select_only_output[0])),
        "native_reduce_valid_scores_bitwise_equal": bool(torch.equal(
            reference_scores[:, :valid].view(torch.int32),
            candidate_scores[:, :valid].view(torch.int32))),
        "native_reduce_valid_scores_max_abs": float(
            (reference_scores[:, :valid] - candidate_scores[:, :valid]).abs().max()),
        "reference_first": reference[0][0, :16].tolist(),
        "split_direct_saved_ms": split["mean_ms"] - split_direct["mean_ms"],
        "split_direct_no_publish_saved_ms": (
            split["mean_ms"] - split_direct_no_publish["mean_ms"]),
        "production_runtime_hot_saved_ms": split["mean_ms"] - runtime_hot["mean_ms"],
        "native_reduce_saved_vs_runtime_ms": runtime_hot["mean_ms"] - native_reduce["mean_ms"],
        "e2e_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
