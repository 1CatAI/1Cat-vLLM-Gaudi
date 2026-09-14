# SPDX-License-Identifier: Apache-2.0
"""Check independent K-partitioned decoding against the unchanged device path."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_native_moe import DECODE, HPU, MOE, mxfp4_bf16_lut, prepared, torch

KDECODE = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2
KMOE = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2


@HPU
@pytest.mark.parametrize("k", [128, 1152, 5120])
@pytest.mark.parametrize("normal", [False, True])
def test_decoder_partition_boundaries_and_encoding_contract(k, normal):
    torch._dynamo.reset()
    packed = np.tile(np.arange(16, dtype=np.uint8) * 17, (2, 256, k // 32))
    scales = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (2, 256, k // 32)).copy()
    if normal:
        scales = scales.clip(2, 254)
    q, s = prepared(packed, scales, "hpu")
    lookup = mxfp4_bf16_lut(torch.device("hpu"))
    candidate = torch.compile(KDECODE, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(DECODE, backend="hpu_backend", fullgraph=True, dynamic=False)
    ids = torch.empty((1, 6), dtype=torch.int32, device="hpu")
    for order in ([0, 1, 1, -1, 2, 0], [1, 0, -2, 1, 0, 0], [0, 0, 1, 1, 1, 0]):
        ids.copy_(torch.tensor([order], dtype=torch.int32))
        args = ids, q, s, lookup, normal
        expected = reference(*args).cpu().view(torch.int16)
        actual = candidate(*args).cpu().view(torch.int16)
        assert torch.equal(actual, expected)


@HPU
def test_complete_moe_full_k_and_changing_routing_exact():
    torch._dynamo.reset()
    torch.manual_seed(413)
    # Real TP2 matrix geometry, distinct experts, all nibble values, changing
    # scales and live routing. Neither path materializes CPU BF16 weights.
    shapes = ((2, 18, 163840), (2, 40, 36864))
    q13, q2 = [torch.randint(-32768, 32768, shape, dtype=torch.int16).to("hpu") for shape in shapes]
    scales = []
    for shape in shapes:
        bits = torch.randint(110, 120, (*shape[:2], shape[2] // 8), dtype=torch.int16) * 128
        scales.append(bits.view(torch.bfloat16).to("hpu"))
    lookup = mxfp4_bf16_lut(torch.device("hpu"))
    candidate = torch.compile(KMOE, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(MOE, backend="hpu_backend", fullgraph=True, dynamic=False)
    x = torch.empty(1, 5120, dtype=torch.bfloat16, device="hpu")
    ids = torch.empty(1, 6, dtype=torch.int32, device="hpu")
    routing = torch.empty(1, 6, device="hpu")
    records = []
    for step in range(4):
        x.copy_((torch.randn(1, 5120) * (step + 1)).bfloat16())
        ids.copy_(((torch.arange(6) + step) % 2).reshape(1, 6).int())
        route = torch.rand(1, 6)
        routing.copy_(route / route.sum(-1, keepdim=True) * 1.5)
        args = x, ids, routing, q13, q2, *scales, lookup, True
        expected = reference(*args).cpu().view(torch.int16)
        actual = candidate(*args).cpu().view(torch.int16)
        records.append({"step": step, "different_bf16_values": int((expected != actual).sum())})
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    (evidence / "expert-k128-output.json").write_text(json.dumps(records, indent=2) + "\n")
    assert all(record["different_bf16_values"] == 0 for record in records), records


def test_k128_moe_meta_rejects_non_c1():
    args = [
        torch.empty(2, 5120, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 6, dtype=torch.int32, device="meta"),
        torch.empty(2, 6, dtype=torch.float32, device="meta"),
        torch.empty(2, 18, 163840, dtype=torch.int16, device="meta"),
        torch.empty(2, 40, 36864, dtype=torch.int16, device="meta"),
        torch.empty(2, 18, 20480, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 40, 4608, dtype=torch.bfloat16, device="meta"),
        torch.empty(128, dtype=torch.bfloat16, device="meta")
    ]
    with pytest.raises(RuntimeError, match="C1"):
        KMOE(*args, True)
