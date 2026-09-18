# SPDX-License-Identifier: Apache-2.0
"""Measure the regular large-M V4.1 N256 MoE prefill path.

This is deliberately a bounded microbenchmark.  It loads a small set of real
experts from one prepared TP/PP shard, calls the same BF16 N256 compound op
used by the model, and reports separate M buckets.  It does not qualify
end-to-end serving latency or model quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 3, 128, 8192])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fp8-fused", action="store_true")
    parser.add_argument("--fp8",
                        action="store_true",
                        help="use the non-fused N256 FP8 body, which supports large-M prefill")
    parser.add_argument("--tile",
                        type=int,
                        default=0,
                        help="split a large prefill batch into fixed token tiles before calling the native op")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.experts <= 384:
        raise ValueError("experts must be within the prepared 384-expert range")

    # The caller supplies the isolated runtime, kernel database and native
    # extension through the same environment used by the server.  Avoid the
    # full server bootstrap here so this bounded diagnostic does not require
    # loading the independent Engram host module.
    import torch
    import habana_frameworks.torch.core  # noqa: F401

    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert
    from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    lut = mxfp4_bf16_lut("hpu")
    selected = list(range(args.experts))

    def prepare_projection(name: str):
        q_source = shard.catalog[f"layers.0.ffn.experts.{name}_q16"]
        s_source = shard.catalog[f"layers.0.ffn.experts.{name}_s16"]
        q_rows, s_rows, channels = [], [], []
        for expert in selected:
            q, s = read_expert(q_source, expert), read_expert(s_source, expert)
            prepared_q, prepared_s, channel, _ = prepare_expert(q, s)
            q_rows.append(prepared_q)
            s_rows.append(prepared_s)
            channels.append(channel)

        def move(values, *, bf16=False):
            array = np.stack(values)
            tensor = torch.from_numpy(array.view(np.int16))
            return (tensor.view(torch.bfloat16) if bf16 else tensor).to("hpu")

        return move(q_rows), move(s_rows), move(channels, bf16=True)

    q13, s13, c13 = prepare_projection("w13")
    q2, s2, c2 = prepare_projection("w2")
    if args.fp8_fused and args.fp8:
        raise ValueError("choose --fp8-fused or --fp8")
    op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
          if args.fp8_fused else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2
          if args.fp8 else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2)

    def make_fn(tokens: int):

        def fn(x, ids, route):
            if args.fp8_fused:
                return op(x, ids, route, q13, q2, s13, s2, lut, c13, c2, True)
            if args.fp8:
                return op(x, ids, route, q13, q2, s13, s2, lut, c13, c2, True)
            return op(x, ids, route, q13, q2, s13, s2, lut, True)

        return fn

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "single-rank native N256 BF16 prefill MoE; real prepared experts; no TP/PP",
        "prepared": str(args.prepared),
        "experts": args.experts,
        "tokens": args.tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "fp8_fused": args.fp8_fused,
        "fp8": args.fp8,
        "tile": args.tile,
        "records": [],
    }
    torch.manual_seed(20260916)
    with torch.inference_mode():
        for tokens in args.tokens:
            x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="hpu")
            ids = torch.randint(0, args.experts, (tokens, 6), dtype=torch.int32, device="hpu")
            route = torch.softmax(torch.randn(tokens, 6, device="hpu"), -1).mul_(1.5).float()
            raw_fn = make_fn(tokens)
            if args.tile and tokens > args.tile:
                # This models a graph compiler's bounded internal tile while
                # preserving the scheduler's single max_num_batched_tokens
                # transaction.  It is intentionally explicit in the
                # microbenchmark; no C1/C6 request loop is hidden here.
                def tiled_fn(x, ids, route, tokens=tokens, raw_fn=raw_fn):
                    pieces = []
                    for begin in range(0, tokens, args.tile):
                        end = min(tokens, begin + args.tile)
                        pieces.append(raw_fn(x[begin:end], ids[begin:end], route[begin:end]))
                    return torch.cat(pieces, dim=0)

                fn = torch.compile(tiled_fn, backend="hpu_backend", fullgraph=True, dynamic=False)
            else:
                fn = torch.compile(raw_fn, backend="hpu_backend", fullgraph=True, dynamic=False)
            output = fn(x, ids, route)
            torch.hpu.synchronize()
            if not torch.isfinite(output.float()).all().item():
                raise RuntimeError(f"non-finite native output at M={tokens}")
            for _ in range(args.warmup):
                x.add_(0.001)
                fn(x, ids, route)
            torch.hpu.synchronize()
            events = []
            host = []
            for _ in range(args.repeat):
                begin = torch.hpu.Event(enable_timing=True)
                end = torch.hpu.Event(enable_timing=True)
                start_ns = time.perf_counter_ns()
                begin.record()
                fn(x, ids, route)
                end.record()
                end.synchronize()
                events.append(float(begin.elapsed_time(end)))
                host.append((time.perf_counter_ns() - start_ns) / 1e6)
            record = {
                "tokens": tokens,
                "shape": [tokens, 5120],
                "device_ms": events,
                "host_ms": host,
                "device_mean_ms": float(np.mean(events)),
                "host_mean_ms": float(np.mean(host)),
                "output_shape": list(output.shape),
            }
            result["records"].append(record)
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
