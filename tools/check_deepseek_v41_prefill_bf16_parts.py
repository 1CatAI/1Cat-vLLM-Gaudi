#!/usr/bin/env python3
"""Compile the standard-dtype pieces of the V4.1 prompt MoE bridge."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut


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


def decode(ids, q, s, lookup):
    return torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
        ids, q, s, lookup, False)


def bf16_moe(x, ids, routing, w13, w2):
    return torch.ops.hpu.mixture_of_experts(
        hidden_states=x,
        expert_routing_table=ids,
        router_weights=routing,
        w12=tuple(w13[index] for index in range(w13.shape[0])),
        w3=tuple(w2[index] for index in range(w2.shape[0])),
        permuted_weights=True,
        activation="silu",
        experts_min=0,
        experts_max=w13.shape[0] - 1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--part", choices=("decode", "moe"), required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import vllm_gaudi  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    shard = PreparedV41Shard(args.prepared, 0, 0)
    prefix = "layers.0.ffn.experts."
    q13_cpu = read_experts(shard, prefix + "w13_q16", args.experts)
    s13_cpu = read_experts(shard, prefix + "w13_s16", args.experts)
    if args.part == "decode":
        import numpy as np
        from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert
        q_values, s_values = [], []
        for expert in range(args.experts):
            q, scales, _, _ = prepare_expert(q13_cpu[expert].numpy(),
                                              s13_cpu[expert].view(torch.uint16).numpy())
            q_values.append(q)
            s_values.append(scales)
        q13_cpu = torch.from_numpy(np.stack(q_values))
        s13_cpu = torch.from_numpy(np.stack(s_values))
    q13 = q13_cpu.to("hpu")
    s13 = s13_cpu.to("hpu")
    lookup = mxfp4_bf16_lut("hpu")
    local_ids = torch.arange(args.experts, dtype=torch.int32, device="hpu").reshape(1, -1)

    if args.part == "decode":
        program = torch.compile(decode, backend="hpu_backend", fullgraph=True, dynamic=False)
        result = program(local_ids, q13, s13, lookup)
    else:
        x = torch.randn(args.tokens, 5120, dtype=torch.bfloat16, device="hpu")
        ids = (torch.arange(args.tokens * 6, dtype=torch.int64, device="hpu")
               .reshape(args.tokens, 6) % args.experts)
        routing = torch.rand(args.tokens, 6, dtype=torch.bfloat16, device="hpu")
        routing = routing / routing.sum(-1, keepdim=True)
        # Standard fused-MoE weight contract is [N,K].
        w13 = torch.randn(args.experts, 4608, 5120, dtype=torch.bfloat16, device="hpu")
        w2 = torch.randn(args.experts, 5120, 2304, dtype=torch.bfloat16, device="hpu")
        program = torch.compile(bf16_moe, backend="hpu_backend", fullgraph=True, dynamic=False)
        result = program(x, ids, routing, w13, w2)
    torch.hpu.synchronize()
    report = {
        "part": args.part,
        "shape": list(result.shape),
        "dtype": str(result.dtype),
        "finite": bool(torch.isfinite(result.float()).all().cpu()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
