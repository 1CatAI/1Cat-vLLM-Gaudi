#!/usr/bin/env python3
"""Probe the stock Habana MXFP4 fused-MoE at V4.1 prefill shapes.

The production rank files retain the Q16/S16 decode layout.  This probe reads
only a bounded expert prefix, reverses that lossless layout on CPU, and keeps
the resulting checkpoint-format MXFP4 tensors on one HPU.  It deliberately
does not load a model or allocate KV cache so prefill shape qualification can
run while the serving model remains resident on other modules.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v4_mxfp4 import (
    restore_mxfp4_scale_u8,
    restore_mxfp4_u8,
)

DTYPES = {
    "I16": torch.int16,
    "BF16": torch.bfloat16,
}


def read_experts(shard: PreparedV41Shard, name: str, count: int) -> torch.Tensor:
    source = shard.catalog[name]
    if source.dtype not in DTYPES or not 1 <= count <= source.shape[0]:
        raise ValueError(f"unsupported bounded tensor read: {name}")
    elements = count * math.prod(source.shape[1:])
    nbytes = elements * torch.empty((), dtype=DTYPES[source.dtype]).element_size()
    with shard.path.open("rb") as stream:
        stream.seek(source.offset)
        raw = bytearray(nbytes)
        if stream.readinto(raw) != nbytes:
            raise RuntimeError(f"rank file ended while reading {name}")
    return torch.frombuffer(raw, dtype=DTYPES[source.dtype]).clone().reshape(count, *source.shape[1:])


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=(16, 128, 512))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import habana_frameworks.torch.core as htcore
    import vllm_gaudi  # noqa: F401 - registers HPU operators
    from vllm_gaudi.extension.ops import _mxfp4_fused_fwd, _mxfp4_hpu_backend

    shard = PreparedV41Shard(args.prepared, 0, 0)
    prefix = "layers.0.ffn.experts."
    started = time.perf_counter()
    w13 = restore_mxfp4_u8(read_experts(shard, prefix + "w13_q16", args.experts)).to("hpu")
    w2 = restore_mxfp4_u8(read_experts(shard, prefix + "w2_q16", args.experts)).to("hpu")
    s13 = restore_mxfp4_scale_u8(read_experts(shard, prefix + "w13_s16", args.experts)).to("hpu")
    s2 = restore_mxfp4_scale_u8(read_experts(shard, prefix + "w2_s16", args.experts)).to("hpu")
    torch.hpu.synchronize()
    load_seconds = time.perf_counter() - started

    weights13 = tuple(w13[index] for index in range(args.experts))
    weights2 = tuple(w2[index] for index in range(args.experts))
    scales13 = tuple(s13[index] for index in range(args.experts))
    scales2 = tuple(s2[index] for index in range(args.experts))
    compiled = torch.compile(_mxfp4_fused_fwd, backend=_mxfp4_hpu_backend, fullgraph=True, dynamic=False)

    results = []
    for tokens in args.tokens:
        torch.manual_seed(17 + tokens)
        hidden = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="hpu")
        ids = (torch.arange(tokens * 6, dtype=torch.int32, device="hpu").reshape(tokens, 6).remainder(args.experts))
        routing = torch.rand(tokens, 6, dtype=torch.bfloat16, device="hpu")
        routing /= routing.sum(-1, keepdim=True)

        def invoke(hidden=hidden, ids=ids, routing=routing):
            return compiled(hidden, ids, routing, weights13, weights2, scales13, scales2, 32, "silu", 0,
                            args.experts - 1, 0, 0)

        output = None
        for _ in range(args.warmup):
            output = invoke()
            torch.hpu.synchronize()
        elapsed = []
        for _ in range(args.repeats):
            begin = time.perf_counter_ns()
            output = invoke()
            torch.hpu.synchronize()
            elapsed.append((time.perf_counter_ns() - begin) / 1e6)
        assert output is not None
        finite = bool(torch.isfinite(output.float()).all().cpu())
        checksum = float(output.float().sum().cpu())
        results.append({
            "tokens": tokens,
            "shape": list(output.shape),
            "finite": finite,
            "checksum": checksum,
            "mean_ms": statistics.fmean(elapsed),
            "p50_ms": percentile(elapsed, 0.50),
            "p95_ms": percentile(elapsed, 0.95),
            "samples_ms": elapsed,
        })
        htcore.mark_step()

    report = {
        "schema_version": 1,
        "purpose": "V4.1 large-M prefill qualification of stock Habana MXFP4 FusedMoE",
        "prepared": str(args.prepared),
        "rank": "pp0-tp0",
        "layer": 0,
        "experts_loaded": args.experts,
        "weight_shapes": {
            "w13": list(w13.shape),
            "w2": list(w2.shape),
            "w13_scale": list(s13.shape),
            "w2_scale": list(s2.shape),
        },
        "load_seconds": load_seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
