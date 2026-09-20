#!/usr/bin/env python3
"""Validate and time bounded Q16 -> stock MXFP4 prefill composition."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v4_mxfp4 import restore_mxfp4_scale_u8, restore_mxfp4_u8


DTYPES = {"I16": torch.int16, "BF16": torch.bfloat16}


def read_experts(shard, name, count):
    source = shard.catalog[name]
    dtype = DTYPES[source.dtype]
    nbytes = count * math.prod(source.shape[1:]) * torch.empty((), dtype=dtype).element_size()
    with shard.path.open("rb") as stream:
        stream.seek(source.offset)
        raw = bytearray(nbytes)
        if stream.readinto(raw) != nbytes:
            raise RuntimeError(f"truncated {name}")
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(count, *source.shape[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-chunk", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chain", type=int, default=1,
                        help="enqueue this many dependent MoE layers before each synchronize")
    parser.add_argument("--n256", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--backend", choices=("mxfp4", "hpu"), default="mxfp4")
    parser.add_argument("--implementation", choices=("mxfp4", "bf16"), default="mxfp4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import vllm_gaudi  # noqa: F401
    if args.implementation == "bf16":
        torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    from vllm_gaudi.extension.ops import _mxfp4_fused_fwd, _mxfp4_hpu_backend
    from vllm_gaudi.ops.deepseek_v41_prefill_moe import (
        q16_chunked_bf16_moe,
        q16_chunked_mxfp4_moe,
    )
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut

    shard = PreparedV41Shard(args.prepared, 0, 0)
    prefix = "layers.0.ffn.experts."
    names = ("w13_q16", "w2_q16", "w13_s16", "w2_s16")
    cpu_q13, cpu_q2, cpu_s13, cpu_s2 = [read_experts(shard, prefix + name, args.experts) for name in names]
    torch.manual_seed(7)
    x = torch.randn(args.tokens, 5120, dtype=torch.bfloat16, device="hpu")
    ids = torch.arange(args.tokens * 6, dtype=torch.int32, device="hpu").reshape(args.tokens, 6) % args.experts
    routing = torch.rand(args.tokens, 6, dtype=torch.float32, device="hpu")
    routing /= routing.sum(-1, keepdim=True)

    def views(tensor):
        return tuple(tensor[index] for index in range(tensor.shape[0]))

    standard = None
    if not args.skip_reference:
        standard_cpu = (restore_mxfp4_u8(cpu_q13), restore_mxfp4_u8(cpu_q2),
                        restore_mxfp4_scale_u8(cpu_s13), restore_mxfp4_scale_u8(cpu_s2))
        standard = tuple(map(views, (value.to("hpu") for value in standard_cpu)))
    if args.n256:
        import numpy as np
        from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert

        converted = []
        for q_source, s_source in ((cpu_q13, cpu_s13), (cpu_q2, cpu_s2)):
            q_values, s_values = [], []
            for expert in range(args.experts):
                q, scales, _, _ = prepare_expert(q_source[expert].numpy(), s_source[expert].view(torch.uint16).numpy())
                q_values.append(q)
                s_values.append(scales)
            converted.extend((torch.from_numpy(np.stack(q_values)), torch.from_numpy(np.stack(s_values))))
        cpu_q13, cpu_s13, cpu_q2, cpu_s2 = converted
    q13, q2, s13, s2 = (cpu_q13.to("hpu"), cpu_q2.to("hpu"), cpu_s13.to("hpu"), cpu_s2.to("hpu"))
    full = torch.compile(_mxfp4_fused_fwd, backend=_mxfp4_hpu_backend, fullgraph=True, dynamic=False)
    chunked_backend = _mxfp4_hpu_backend if args.backend == "mxfp4" else "hpu_backend"
    implementation = q16_chunked_mxfp4_moe if args.implementation == "mxfp4" else q16_chunked_bf16_moe
    chunked = torch.compile(implementation, backend=chunked_backend, fullgraph=True, dynamic=False)
    expected = (None if standard is None else
                full(x, ids, routing.to(torch.bfloat16), *standard, 32, "silu", 0, args.experts - 1, 0, 0))
    if hasattr(torch.hpu, "reset_peak_memory_stats"):
        torch.hpu.reset_peak_memory_stats()
    lookup = mxfp4_bf16_lut("hpu")
    extra = () if args.implementation == "mxfp4" else (lookup, )
    actual = x
    for _ in range(args.chain):
        actual = chunked(actual, ids, routing, q13, q2, s13, s2, *extra,
                         expert_chunk=args.expert_chunk)
    torch.hpu.synchronize()
    exact = None if expected is None else bool(torch.equal(expected.cpu(), actual.cpu()))
    maximum = None if expected is None else float((expected.float() - actual.float()).abs().max().cpu())
    samples = []
    for _ in range(args.repeats):
        begin = time.perf_counter_ns()
        actual = x
        for _ in range(args.chain):
            actual = chunked(actual, ids, routing, q13, q2, s13, s2, *extra,
                             expert_chunk=args.expert_chunk)
        torch.hpu.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1e6)
    report = {
        "experts": args.experts,
        "expert_chunk": args.expert_chunk,
        "tokens": args.tokens,
        "chain": args.chain,
        "backend": args.backend,
        "implementation": args.implementation,
        "resident_layout": "n256" if args.n256 else "q16-v2",
        "exact_vs_full_stock": exact,
        "max_abs_error": maximum,
        "mean_ms": statistics.fmean(samples),
        "samples_ms": samples,
        "finite": bool(torch.isfinite(actual.float()).all().cpu()),
        "peak_allocated_bytes": int(torch.hpu.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.hpu.max_memory_reserved()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
