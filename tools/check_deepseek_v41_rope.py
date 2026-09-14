#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device checks for V4.1's fused RoPE and unchanged BF16 consumer."""
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401
from vllm_gaudi.ops.deepseek_v41_math import _apply_rope_torch, apply_rope, rotary_table  # noqa: E402


def main():
    destination = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(25)
    table = rotary_table(64, 1024, 10000).to("hpu")
    old = torch.compile(_apply_rope_torch, backend="hpu_backend", fullgraph=True, dynamic=False)
    new = torch.compile(apply_rope, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []

    def compare(label, values, positions, inverse):
        expected = old(values, positions, table, inverse).cpu()
        actual = new(values, positions, table, inverse).cpu()
        bad = expected.view(torch.int16) != actual.view(torch.int16)
        record = dict(case=label,
                      shape=list(values.shape),
                      inverse=inverse,
                      mismatches=int(bad.sum()),
                      prefix_mismatches=int(bad[..., :-64].sum()))
        records.append(record)
        (destination / "result.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if bad.any():
            torch.save(
                dict(values=values.cpu(),
                     positions=positions.cpu(),
                     table=table.cpu(),
                     expected=expected,
                     actual=actual,
                     inverse=inverse), destination / "mismatch.pt")
            raise RuntimeError("Native RoPE differs from the compiled production expression")

    for count in range(1, 7):
        shape = (count, 512) if count % 2 else (count, 32, 512)
        positions = torch.tensor([0, 1, 127, 128, 511, 1023][:count], dtype=torch.int32, device="hpu")
        values = torch.randn(shape).bfloat16().to("hpu")
        for inverse in (False, True):
            compare(f"C{count}", values, positions, inverse)
    # Inputs and positions change while the same C6 graph is replayed.
    for step in range(2):
        values = torch.randn(6, 32, 512).bfloat16()
        # All BF16 encodings in the copied prefix must remain unchanged.
        bits = torch.arange(values[..., :-64].numel()).remainder(65536).to(torch.int16)
        values[..., :-64] = bits.view(torch.bfloat16).reshape(6, 32, 448)
        compare(f"prefix_bits_replay_{step}", values.to("hpu"),
                torch.tensor([1023, 511, 128, 127, 1, step], dtype=torch.int32, device="hpu"), bool(step))
    values = torch.randn(6, 128).bfloat16().to("hpu")
    positions = torch.arange(6, device="hpu", dtype=torch.int32)
    compare("index_width128", values, positions, False)

    weight = torch.randn(512, 64).bfloat16().to("hpu")

    def chain(x, positions, native):
        q = apply_rope(x, positions, table) if native else _apply_rope_torch(x, positions, table)
        q = apply_rope(q, positions, table, True) if native else _apply_rope_torch(q, positions, table, True)
        return torch.matmul(q, weight)

    old_chain = torch.compile(lambda x, p: chain(x, p, False), backend="hpu_backend", fullgraph=True, dynamic=False)
    new_chain = torch.compile(lambda x, p: chain(x, p, True), backend="hpu_backend", fullgraph=True, dynamic=False)
    x = torch.randn(6, 32, 512).bfloat16().to("hpu")
    expected, actual = old_chain(x, positions).cpu(), new_chain(x, positions).cpu()
    mismatches = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
    records.append(dict(case="connected_bf16_mme", mismatches=mismatches))
    (destination / "result.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records[-1]), flush=True)
    assert not mismatches


if __name__ == "__main__":
    main()
