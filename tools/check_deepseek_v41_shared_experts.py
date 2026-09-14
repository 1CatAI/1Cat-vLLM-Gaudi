# SPDX-License-Identifier: Apache-2.0
"""Changed C6 expert dataflow: real weights, native BF16 reference, no timing baseline.

Run under run_deepseek_v41.py --devices 1 with the candidate runtime profile.
The launcher owns the device and fingerprints the exact runtime/source.
"""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shared_experts import expert_owners, shared_expert_moe  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--implementation", choices=("shared", "k128", "n512"), default="shared")
    parser.add_argument("--sram-kv-chain", action="store_true")
    parser.add_argument("--head-vector", action="store_true")
    parser.add_argument("--kv-pack-chain", action="store_true")
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    shard = PreparedV41Shard(args.checkpoint, 0, 0)
    weights = SimpleNamespace()
    for name in ("w13_q16", "w2_q16", "w13_s16", "w2_s16"):
        spec = shard.catalog["layers.0.ffn.experts." + name]
        shape = (12, *spec.shape[1:])
        with shard.path.open("rb") as stream:
            stream.seek(spec.offset)
            storage = bytearray(math.prod(shape) * 2)
            if stream.readinto(storage) != len(storage):
                raise ValueError("Short prepared weight read")
        tensor = torch.frombuffer(storage, dtype=torch.int16).reshape(shape)
        if name.endswith("s16"):
            tensor = tensor.view(torch.bfloat16)
        setattr(weights, name, tensor.to("hpu"))
    shard.check_identity()
    lookup = mxfp4_bf16_lut(torch.device("hpu"))
    old = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2
    if args.sram_kv_chain:
        from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
        torch.manual_seed(20260912)
        swa = pack_swa(torch.randn(256, 512).bfloat16()).to("hpu")
        main_cache = pack_fp4(torch.randn(1024, 512).bfloat16()).to("hpu")
        row_ids = torch.cat((torch.arange(256), torch.arange(512) * 7 % 1024 + 256)).int().unsqueeze(0).to("hpu")
        indices = torch.cat((torch.arange(128), torch.arange(512) + 256)).int().expand(6, -1).contiguous().to("hpu")
        lengths = torch.arange(200, 206, dtype=torch.int32, device="hpu")
        sink = torch.randn(32, device="hpu")
        scale = torch.tensor([512**-0.5], device="hpu")

    def attention_input(value, use_sram):
        if not args.sram_kv_chain:
            return value
        q = torch.nn.functional.pad(value.reshape(6, 10, 512), (0, 0, 0, 22))
        op = (torch.ops.custom_op.custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2
              if use_sram and args.head_vector else torch.ops.custom_op.custom_deepseek_v41_paged_attention_sram_bf16_gaudi2 if use_sram
              else torch.ops.custom_op.custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2)
        current_swa = swa
        if args.kv_pack_chain:
            from vllm_gaudi.ops.deepseek_v41_math import _pack_swa_torch
            incoming = q[:, 0, :].contiguous()
            encoded = (torch.ops.custom_op.custom_deepseek_v41_swa_pack_bf16_gaudi2(incoming)
                       if use_sram else _pack_swa_torch(incoming))
            slots = torch.arange(6, device=value.device, dtype=torch.int64)
            current_swa = swa.index_copy(0, slots, encoded)
        attended = op(q, current_swa, main_cache, row_ids, indices, sink, scale, lengths)
        return attended[:, :10, :].reshape(6, 5120).contiguous()


    def reference(value, ids, routing):
        value = attention_input(value, False)
        return old(value, ids, routing, weights.w13_q16, weights.w2_q16,
                   weights.w13_s16, weights.w2_s16, lookup, True)

    def candidate(value, ids, routing):
        value = attention_input(value, True)
        if args.implementation in ("k128", "n512"):
            op = (torch.ops.custom_op.custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2
                  if args.implementation == "n512" else
                  torch.ops.custom_op.custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2)
            return op(
                value, ids, routing, weights.w13_q16, weights.w2_q16,
                weights.w13_s16, weights.w2_s16, lookup, True)
        return shared_expert_moe(value, ids, routing, weights, lookup, True)

    reference = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    candidate = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(20260912)
    records = []
    for step in range(2):
        value = (torch.randn(6, 5120) * (1 + step)).bfloat16()
        ids = ((torch.arange(36).reshape(6, 6) * (step + 1) + step) % 12).int()
        if step:
            # Duplicate experts within a token must retain independent W2
            # input rounding. Also keep the established invalid-ID zero path.
            ids[:, 2] = ids[:, 0]
            ids[1, 1] = -1
        routing = torch.rand(6, 6)
        routing = routing / routing.sum(-1, keepdim=True) * 1.5
        inputs = [x.to("hpu") for x in (value, ids, routing)]
        expected = reference(*inputs).cpu()
        actual = candidate(*inputs).cpu()
        active, rows = expert_owners(ids)
        mismatches = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
        record = {"step": step, "implementation": args.implementation, "sram_kv_chain": args.sram_kv_chain,
                  "active_experts": int(active.sum()), "slots": 36,
                  "unique_scatter_rows": rows.unique().numel(), "bf16_mismatches": mismatches,
                  "max_abs_error": (actual.float() - expected.float()).abs().max().item()}
        records.append(record)
        torch.save(dict(value=value, ids=ids, routing=routing, expected=expected, actual=actual),
                   output / f"real-weight-{step}.pt")
        (output / "result.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if mismatches:
            raise RuntimeError("Shared C6 expert output differs from native BF16 reference")
    torch.hpu.synchronize()


if __name__ == "__main__":
    main()
