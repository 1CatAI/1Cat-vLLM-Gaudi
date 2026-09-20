# SPDX-License-Identifier: Apache-2.0
"""Compare the native decoded-index score kernel with the eager MME score path.

This is a component benchmark only.  It uses the fixed C1 hot-cache geometry
and the same threshold/emit consumers as serving; it makes no end-to-end claim.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401,E402
import torch


def _sync() -> None:
    torch.hpu.synchronize()


def _timed(fn, args, warmups: int, repeats: int) -> dict:
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    for _ in range(warmups):
        compiled(*args)
    _sync()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        compiled(*args)
        _sync()
        samples.append((time.perf_counter_ns() - begin) / 1e6)
    samples.sort()
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "samples_ms": samples,
    }


def _sync_cpu(value):
    """Materialize a compiled result without timing the transfer."""
    if isinstance(value, tuple):
        return tuple(_sync_cpu(item) for item in value)
    return value.cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--visible", type=int, default=2048)
    ap.add_argument("--capacity", type=int, default=2560)
    ap.add_argument("--ratio", type=int, choices=(1, 2), default=1)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    visible_rows = (args.visible + args.ratio - 1) // args.ratio
    if args.capacity % 64 or visible_rows > args.capacity:
        raise ValueError("capacity must cover visible compressed rows and be a multiple of 64")
    torch.ops.load_library(__import__("os").environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(20260919)
    q = torch.randn((1, 32, 128), dtype=torch.bfloat16, device="hpu")
    weights = (torch.randn((1, 32), dtype=torch.float32, device="hpu") * .02).bfloat16()
    cache = torch.randn((args.capacity, 128), dtype=torch.bfloat16, device="hpu")
    page_rows = 128 // args.ratio
    pages = torch.arange((args.capacity + page_rows - 1) // page_rows,
                         dtype=torch.int32, device="hpu")
    candidates = torch.full((1, 2048), -1, dtype=torch.int32, device="hpu")
    position = torch.tensor([args.visible - 1], dtype=torch.int32, device="hpu")
    ratio = args.ratio

    def native_score(q_, w_, c_, p_, pos_, cand_):
        return torch.ops.custom_op.custom_deepseek_v41_index_scores_decoded_gaudi2(
            q_, w_, c_, p_, pos_, cand_, ratio, 0, args.capacity)

    def eager_score(q_, w_, c_, p_, pos_, cand_):
        visible = ((pos_.to(torch.int32) + 1) // ratio).unsqueeze(-1)
        score = torch.einsum("bhd,nd->bhn", q_, c_)
        score = (score.to(torch.bfloat16).relu() * w_.unsqueeze(-1)).to(torch.bfloat16)
        score = score.reshape(1, 2, 16, -1).sum(2, dtype=torch.bfloat16)
        score = score.sum(1, dtype=torch.bfloat16).float()
        rows = torch.arange(c_.shape[0], dtype=torch.int32, device=c_.device)
        return score.masked_fill(rows.unsqueeze(0) >= visible, -torch.inf).contiguous(), \
            torch.empty((1, (c_.shape[0] // 8 + 63) // 64 * 64), dtype=torch.float32, device=c_.device)

    def native_chain(q_, w_, c_, p_, pos_, cand_):
        score, blocks = native_score(q_, w_, c_, p_, pos_, cand_)
        stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
            score, pos_, ratio, 0, 0)
        selected = torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
            score, pos_, cand_, stats, ratio, 0, 0)
        return selected, blocks

    def eager_chain(q_, w_, c_, p_, pos_, cand_):
        score, blocks = eager_score(q_, w_, c_, p_, pos_, cand_)
        stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
            score, pos_, ratio, 0, 0)
        selected = torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
            score, pos_, cand_, stats, ratio, 0, 0)
        return selected, blocks

    def eager_score_only(q_, w_, c_, p_, pos_, cand_):
        return eager_score(q_, w_, c_, p_, pos_, cand_)[0]

    def threshold_only(score_, pos_):
        return torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
            score_, pos_, ratio, 0, 0)

    def emit_only(score_, pos_, cand_, stats_):
        return torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
            score_, pos_, cand_, stats_, ratio, 0, 0)

    def builtin_topk_chain(q_, w_, c_, p_, pos_, cand_):
        score = eager_score(q_, w_, c_, p_, pos_, cand_)[0]
        # The production consumer requires increasing logical-row order, not
        # score order.  ``sorted=False`` avoids an unnecessary score sort; the
        # selected row IDs are ordered once afterwards.
        selected = torch.topk(score, 512, dim=-1, largest=True, sorted=False).indices
        return selected.sort(dim=-1).values.to(torch.int32)

    def deterministic_topk_chain(q_, w_, c_, p_, pos_, cand_):
        score = eager_score(q_, w_, c_, p_, pos_, cand_)[0]
        # Scores are exactly BF16 after the two checkpoint reduction
        # boundaries.  Encode the BF16 total order plus the required
        # smaller-row tie break into one int32 key, then use the optimized
        # generic top-k implementation without changing candidate semantics.
        bits = torch.bitwise_and(score.to(torch.bfloat16).view(torch.int16).to(torch.int32),
                                 0xffff)
        ordered = torch.where(bits > 0x7fff, torch.bitwise_and(torch.bitwise_not(bits), 0xffff),
                              torch.bitwise_xor(bits, 0x8000))
        rows = torch.arange(score.shape[-1], dtype=torch.int32, device=score.device)
        keys = ordered * (score.shape[-1] + 1) + (score.shape[-1] - 1 - rows)
        selected = torch.topk(keys, 512, dim=-1, largest=True, sorted=False).indices
        return selected.sort(dim=-1).values.to(torch.int32)

    call_args = (q, weights, cache, pages, position, candidates)
    eager_score_compiled = torch.compile(eager_score_only,
                                         backend="hpu_backend",
                                         fullgraph=True,
                                         dynamic=False)
    score_for_select = eager_score_compiled(*call_args)
    _sync()
    threshold_compiled = torch.compile(threshold_only,
                                       backend="hpu_backend",
                                       fullgraph=True,
                                       dynamic=False)
    stats_for_emit = threshold_compiled(score_for_select, position)
    _sync()
    native = _timed(native_chain, call_args, args.warmups, args.repeats)
    eager = _timed(eager_chain, call_args, args.warmups, args.repeats)
    score_timing = _timed(eager_score_only, call_args, args.warmups, args.repeats)
    threshold_timing = _timed(threshold_only, (score_for_select, position), args.warmups,
                              args.repeats)
    emit_timing = _timed(emit_only, (score_for_select, position, candidates, stats_for_emit),
                         args.warmups, args.repeats)
    builtin = _timed(builtin_topk_chain, call_args, args.warmups, args.repeats)
    deterministic = _timed(deterministic_topk_chain, call_args, args.warmups, args.repeats)

    eager_result = torch.compile(eager_chain,
                                 backend="hpu_backend",
                                 fullgraph=True,
                                 dynamic=False)(*call_args)[0]
    builtin_result = torch.compile(builtin_topk_chain,
                                   backend="hpu_backend",
                                   fullgraph=True,
                                   dynamic=False)(*call_args)
    deterministic_result = torch.compile(deterministic_topk_chain,
                                         backend="hpu_backend",
                                         fullgraph=True,
                                         dynamic=False)(*call_args)
    eager_cpu, builtin_cpu, deterministic_cpu = _sync_cpu(
        (eager_result, builtin_result, deterministic_result))
    selected_equal = torch.equal(eager_cpu, builtin_cpu)
    deterministic_equal = torch.equal(eager_cpu, deterministic_cpu)
    result = {
        "benchmark": "dsv41_decoded_index_score_chain",
        "visible": args.visible,
        "capacity": args.capacity,
        "ratio": ratio,
        "shape": {"q": [1, 32, 128], "cache": [args.capacity, 128]},
        "native_decoded_score_plus_select": native,
        "eager_mme_score_plus_select": eager,
        "eager_mme_score_only": score_timing,
        "native_threshold_only": threshold_timing,
        "native_emit_only": emit_timing,
        "builtin_topk_plus_row_sort": builtin,
        "builtin_selected_equal": selected_equal,
        "deterministic_topk_plus_row_sort": deterministic,
        "deterministic_selected_equal": deterministic_equal,
        "e2e_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
