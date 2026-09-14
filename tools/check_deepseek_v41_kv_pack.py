# SPDX-License-Identifier: Apache-2.0
"""Compare pure KV encoder bytes and their compiled attention consumer."""
import argparse
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401
from vllm_gaudi.ops.deepseek_v41_math import _pack_fp4_torch, _pack_swa_torch  # noqa: E402


def main():
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-mismatches", action="store_true")
    args = parser.parse_args()
    records = []
    # Independent codec/shape checks share Dynamo code objects in this diagnostic.
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(17)
    cases = {
        "all_bf16_encodings": torch.arange(65536).to(torch.int16).view(torch.bfloat16).reshape(128, 512),
        "changed_c6": torch.randn(6, 512).bfloat16(),
        "signed_zero": torch.tensor([0, -32768], dtype=torch.int16).repeat(1536).view(torch.bfloat16).reshape(6, 512),
    }
    special = torch.zeros(32, 512, dtype=torch.bfloat16)
    for offset in range(32):
        special[offset, :] = float("nan")
        special[offset, offset] = 1.0
        special[offset, 32 + offset] = float("inf")
        special[offset, 64:96] = 1.0
        special[offset, 64 + offset] = float("nan")
    cases["mixed_nonfinite"] = special
    ops = torch.ops.custom_op
    codecs = (
        ("swa", _pack_swa_torch, ops.custom_deepseek_v41_swa_pack_bf16_gaudi2),
        ("fp4_g16", lambda x: _pack_fp4_torch(x, 16), ops.custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2),
        ("fp4_g32", lambda x: _pack_fp4_torch(x, 32), ops.custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2),
    )
    for name, reference, candidate in codecs:
        old = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
        new = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
        for case, value in cases.items():
            value = value.to("hpu")
            expected, actual = old(value).cpu(), new(value).cpu()
            mismatches = int((expected != actual).sum())
            records.append(dict(codec=name, case=case, bytes=actual.numel(), byte_mismatches=mismatches))
            (root / "result.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(records[-1]), flush=True)
            if mismatches:
                torch.save(dict(value=value.cpu(), expected=expected, actual=actual),
                           root / f"mismatch-{name}-{case}.pt")
                if not args.collect_mismatches:
                    raise RuntimeError("Native KV codec bytes differ from the compiled reference")

    sink = torch.randn(32, device="hpu")
    scale = torch.tensor([512**-0.5], device="hpu")
    indices = torch.arange(6, dtype=torch.int32, device="hpu").expand(6, -1).contiguous()
    lengths = torch.arange(1, 7, dtype=torch.int32, device="hpu")
    cache = torch.zeros(256, 528, dtype=torch.uint8, device="hpu")
    main_cache = torch.zeros(512, 288, dtype=torch.uint8, device="hpu")
    rows = torch.arange(256, dtype=torch.int32, device="hpu").unsqueeze(0)
    slots = torch.arange(6, dtype=torch.int64, device="hpu")

    def chain(query, values, use_native):
        packed = (ops.custom_deepseek_v41_swa_pack_bf16_gaudi2(values) if use_native else _pack_swa_torch(values))
        updated = cache.index_copy(0, slots, packed)
        return ops.custom_deepseek_v41_paged_attention_sram_bf16_gaudi2(query, updated, main_cache, rows, indices, sink,
                                                                        scale, lengths)

    old = torch.compile(lambda q, v: chain(q, v, False), backend="hpu_backend", fullgraph=True, dynamic=False)
    new = torch.compile(lambda q, v: chain(q, v, True), backend="hpu_backend", fullgraph=True, dynamic=False)
    for step in range(2):
        q = torch.randn(6, 32, 512).bfloat16().to("hpu")
        values = torch.randn(6, 512).bfloat16().to("hpu")
        expected, actual = old(q, values).cpu(), new(q, values).cpu()
        mismatch = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
        records.append(dict(chain_step=step, bf16_mismatches=mismatch))
        (root / "result.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
        assert mismatch == 0


if __name__ == "__main__":
    main()
