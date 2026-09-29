# SPDX-License-Identifier: Apache-2.0
"""Bounded prefill MLA through inverse RoPE, real wo_a and wo_b consumers.

This component excludes indexer/compressor and TP/PP, and cannot qualify full
prefill throughput. Use one missing TPC reference per tile, then candidates.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--native-replay", action="store_true")
    parser.add_argument("--production-split",
                        action="store_true",
                        help="Compiled MLA followed by eager output consumers, as in prefill")
    parser.add_argument("--query-tokens",
                        type=int,
                        help="Validate the production bounded helper at a full prefill size")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=12)
    args = parser.parse_args()
    tokens = args.query_tokens or args.tile
    if not 1 <= tokens <= 8192:
        parser.error("query tokens must be in [1,8192]")
    if args.production_split and args.native_replay:
        parser.error("production prefill split does not use whole-chain native replay")
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm_gaudi.ops.deepseek_v41_math import _apply_rope_torch, quantize_activation, rotary_table
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.set_num_threads(1)
    torch.manual_seed(4118)
    torch.ops.load_library(str(args.library))
    recorder = None
    if args.native_replay:
        from deepseek_v41_micro_replay import RecipeRecorder
        recorder = RecipeRecorder(args.output.parent)
    shard = PreparedV41Shard(args.prepared, args.layer // 20, 0)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    prefix = f"layers.{args.layer}.attn."
    woa = shard.dense(prefix + "wo_a.weight", "hpu").reshape(4, 1024, 4096).transpose(1, 2).contiguous()
    wob = shard.dense(prefix + "wo_b.weight", "hpu")
    sink = shard.tensor(prefix + "attn_sink", "hpu").float()
    scaling = config["rope_scaling"]
    table = rotary_table(64, 8192, config["compress_rope_theta"], scaling["original_max_position_embeddings"],
                         scaling["factor"], scaling["beta_fast"], scaling["beta_slow"]).to("hpu")
    q = torch.randn(tokens, 32, 512).bfloat16().to("hpu")
    cache = torch.randn(8192, 512).bfloat16().to("hpu")
    ids = torch.randint(0, 8192, (tokens, 640), dtype=torch.int32)
    ids[0].fill_(-1)
    if tokens > 1:
        ids[1, 256:] = -1
    ids = ids.to("hpu")
    positions = torch.arange(tokens, dtype=torch.int32, device="hpu")
    lengths = torch.full((tokens, ), 640, dtype=torch.int32, device="hpu")
    scale = torch.tensor([512**-.5], device="hpu")

    def attention(q, cache, ids, sink, scale, lengths):
        if args.reference:
            return torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(q, cache, ids, sink, scale)[0]
        if args.query_tokens:
            from vllm_gaudi.ops.deepseek_v41_paged_attention import bounded_prefill_mla
            return bounded_prefill_mla(q, cache, ids, sink, scale, args.tile)
        return torch.ops.custom_op.custom_deepseek_v41_prefill_mla_mme_gaudi2(q, cache, ids, sink, scale, lengths)

    def consumer(value, positions, table, woa, wob):
        rotated = _apply_rope_torch(value, positions, table, True).reshape(-1, 4, 4096)
        projected = torch.einsum("tgd,gdr->tgr", rotated, woa).flatten(1)
        return F.linear(quantize_activation(projected), wob)

    def chain(q, cache, ids, sink, scale, lengths, positions, table, woa, wob):
        return consumer(attention(q, cache, ids, sink, scale, lengths), positions, table, woa, wob)

    fixed = (q, cache, ids, sink, scale, lengths, positions, table, woa, wob)
    with torch.inference_mode():
        compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
        if args.production_split:
            compiled_attention = torch.compile(attention, backend="hpu_backend", fullgraph=True, dynamic=False)
            compiled = lambda *inputs: consumer(compiled_attention(*inputs[:6]), *inputs[6:])
        replay = recorder.prepare(compiled, [q], [fixed[1:]]) if recorder else None
        invoke = (lambda: (replay(), replay.outputs[0])[1]) if replay else lambda: compiled(*fixed)
        for _ in range(3):
            result = invoke()
        torch.hpu.synchronize()
        before = dict(metric_global("graph_compilation").stats())
        wall, device, checks, resources = [], [], [], []
        for repeat in range(args.repeat):
            q.copy_(torch.randn(tokens, 32, 512).bfloat16())
            torch.hpu.synchronize()
            resident = torch.hpu.memory_allocated()
            torch.hpu.reset_peak_memory_stats()
            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            result = invoke()
            end.record()
            end.synchronize()
            wall.append((time.perf_counter() - start) * 1000)
            device.append(begin.elapsed_time(end))
            resources.append({
                "resident_bytes": resident,
                "peak_bytes": torch.hpu.max_memory_allocated(),
                "additional_peak_bytes": torch.hpu.max_memory_allocated() - resident
            })
            if repeat in (0, args.repeat - 1):
                actual = result.cpu()
                ordinary = compiled(*fixed).cpu() if replay else chain(*fixed).cpu()
                checks.append({
                    "max_abs": float((actual.float() - ordinary.float()).abs().max()),
                    "finite": bool(actual.isfinite().all()),
                    "bitwise_equal": torch.equal(actual, ordinary)
                })
                torch.save({
                    "input": q.cpu(),
                    "output": actual,
                    "ordinary": ordinary
                }, args.output.parent / f"outputs-{repeat}.pt")
        # Check boundary rows without a multi-GiB CPU oracle temporary.
        chosen = sorted({0, min(1, tokens - 1), min(args.tile - 1, tokens - 1), min(args.tile, tokens - 1), tokens - 1})
        local = attention(*fixed[:6]).cpu()[chosen]
        qc, kc, ic, sc = q.cpu()[chosen].float(), cache.cpu().float(), ids.cpu()[chosen].long(), sink.cpu()
        selected = kc[ic.clamp_min(0)]
        scores = torch.einsum("thd,tkd->thk", qc, selected) * 512**-.5
        scores.masked_fill_(ic[:, None] < 0, -torch.inf)
        probability = torch.cat((scores, sc[None, :, None].expand(len(chosen), -1, -1)), -1).softmax(-1)[..., :-1]
        oracle = torch.einsum("thk,tkd->thd", probability, selected).bfloat16()
        torch.testing.assert_close(local, oracle, rtol=.02, atol=.002)
        report = dict(tile=args.tile,
                      query_tokens=tokens,
                      reference=args.reference,
                      native_replay=args.native_replay,
                      production_split=args.production_split,
                      timed_memory=resources,
                      wall_ms=wall,
                      device_ms=device,
                      wall_median_ms=statistics.median(wall),
                      device_median_ms=statistics.median(device),
                      checks=checks,
                      attention_oracle_max_abs=float((local.float() - oracle.float()).abs().max()),
                      compilation_before=before,
                      compilation_after=dict(metric_global("graph_compilation").stats()),
                      peak_bytes=torch.hpu.max_memory_allocated(),
                      actual_shape=list(result.shape))
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if replay:
            replay.close()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
