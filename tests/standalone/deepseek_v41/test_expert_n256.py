# SPDX-License-Identifier: Apache-2.0
"""N256 layout, live expert addressing and full-K device contracts."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_native_moe import DECODE, HPU, MOE, mxfp4_bf16_lut, torch
from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, restore_expert

NDECODE = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2
FDECODE = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2
NMOE = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2
FMOE = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2
OLD_FP8 = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_fp8_gaudi2


def matrix(n, k, seed, experts=2):
    rng = np.random.default_rng(seed)
    q = rng.integers(-32768, 32768, (experts, n // 128, k * 32), dtype=np.int16)
    # Different output rows and K groups exercise both planes and channel scales.
    codes = rng.integers(110, 117, (experts, n // 128, k // 32, 128), dtype=np.uint16)
    s = (codes << 7).reshape(experts, n // 128, k * 4)
    return q, s


def device_weights(q, s):
    prepared = [prepare_expert(a, b) for a, b in zip(q, s, strict=True)]
    for a, b, (nq, ns, _, _) in zip(q, s, prepared, strict=True):
        rq, rs = restore_expert(nq, ns)
        assert np.array_equal(a, rq) and np.array_equal(b, rs)
    nq, ns, channel = [np.stack([row[i] for row in prepared]) for i in range(3)]

    def move(array, bf16=False):
        value = torch.from_numpy(array.view(np.int16))
        return (value.view(torch.bfloat16) if bf16 else value).to("hpu")

    return move(q), move(s, True), move(nq), move(ns), move(channel, True)


@HPU
@pytest.mark.parametrize("k", [128, 1152, 5120])
def test_decode_order_live_ids_and_exact_bytes(k):
    torch._dynamo.reset()
    q, s, nq, ns, channel = device_weights(*matrix(256, k, 15))
    lookup = mxfp4_bf16_lut("hpu")
    old = torch.compile(DECODE, backend="hpu_backend", fullgraph=True, dynamic=False)
    new = torch.compile(NDECODE, backend="hpu_backend", fullgraph=True, dynamic=False)
    fp8 = torch.compile(FDECODE, backend="hpu_backend", fullgraph=True, dynamic=False)
    ids = torch.empty(1, 6, device="hpu", dtype=torch.int32)
    c = channel.cpu().float().reshape(2, 256)
    for order in ([0, 1, -1, 0, 2, 1], [1, 0, 1, -2, 0, 0]):
        ids.copy_(torch.tensor([order], dtype=torch.int32))
        expected = old(ids, q, s, lookup, True).cpu()
        actual = new(ids, nq, ns, lookup, True).cpu()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        normalized = expected.float()
        for slot, expert in enumerate(order):
            if 0 <= expert < 2:
                normalized[slot].div_(c[expert])
        expected_fp8 = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
        actual_fp8 = fp8(ids, nq, ns, lookup, True).cpu().view(torch.uint8)
        assert torch.equal(actual_fp8, expected_fp8)


@HPU
def test_normal_scale_lookup_all_encodings_and_invalid_experts():
    if os.environ.get("VLLM_HPU_DSV41_N256_NORMAL_BF16") != "1":
        pytest.skip("Select the normal-scale candidate before loading its native extension")
    # Every nibble paired with all 253 normal scale codes. The second plane
    # belongs to FP8 and must have no influence on BF16 decode.
    codes = 2 + np.arange(256, dtype=np.uint16) % 253
    raw = np.stack([np.full((128, 128), nibble | (nibble << 4), dtype=np.uint8) for nibble in range(16)])
    q = torch.from_numpy(raw.view(np.int16).reshape(16, 1, 8192)).to("hpu")
    planes = np.zeros((16, 4, 512), dtype=np.uint8)
    planes[..., :256] = codes.astype(np.uint8)
    planes[..., 256:] = np.arange(256, dtype=np.uint8)
    s = torch.from_numpy(planes.view(np.int16).reshape(16, 1, 1024)).to("hpu")
    ids = torch.tensor([list(range(16)) + [-1, 16]], dtype=torch.int32, device="hpu")
    lut = mxfp4_bf16_lut("hpu")
    expected = NDECODE(ids, q, s, lut, False).cpu()
    actual = NDECODE(ids, q, s, lut, True).cpu()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(actual[-2:]) == 0


@HPU
def test_complete_moe_rounding_and_bf16_prefill():
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    q13, s13, nq13, ns13, c13 = device_weights(*matrix(2304, 5120, 41))
    q2, s2, nq2, ns2, c2 = device_weights(*matrix(5120, 1152, 42))
    lut = mxfp4_bf16_lut("hpu")
    torch.manual_seed(73)
    old = torch.compile(MOE, backend="hpu_backend", fullgraph=True, dynamic=False)
    compatible = torch.compile(NMOE, backend="hpu_backend", fullgraph=True, dynamic=False)
    fp8 = torch.compile(FMOE, backend="hpu_backend", fullgraph=True, dynamic=False)
    old_fp8 = torch.compile(OLD_FP8, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    for tokens in (1, 3):
        for step in range(2):
            x = torch.randn(tokens, 5120).bfloat16().to("hpu")
            ids = ((torch.arange(tokens * 6) + step) % 2).reshape(tokens, 6).int().to("hpu")
            r = torch.rand(tokens, 6)
            r = (r / r.sum(-1, keepdim=True) * 1.5).to("hpu")
            a = old(x, ids, r, q13, q2, s13, s2, lut, True).cpu()
            b = compatible(x, ids, r, nq13, nq2, ns13, ns2, lut, True).cpu()
            exact = torch.equal(a.view(torch.int16), b.view(torch.int16))
            record = {"tokens": tokens, "step": step, "bf16_compatibility_exact": exact}
            if tokens == 1:
                # The old FP8 algorithm is a component reference, not a quality baseline.
                expected = old_fp8(x, ids, r, q13, q2, s13, s2, lut, c13.reshape(2, 18, 128), c2.reshape(2, 40, 128),
                                   True).cpu()
                result = fp8(x, ids, r, nq13, nq2, ns13, ns2, lut, c13, c2, True).cpu()
                record["fp8_algorithm_exact"] = torch.equal(expected.view(torch.int16), result.view(torch.int16))
                difference = expected.float() - result.float()
                record["fp8_max_absolute_error"] = difference.abs().max().item()
                record["fp8_relative_rms_error"] = (difference.square().mean().sqrt() /
                                                    expected.float().square().mean().sqrt()).item()
                ordinary = FMOE(x, ids, r, nq13, nq2, ns13, ns2, lut, c13, c2, True).cpu()
                record["ordinary_compiled_exact"] = torch.equal(ordinary.view(torch.int16), result.view(torch.int16))
                torch.save({
                    "expected": expected,
                    "actual": result
                },
                           Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"fp8-output-{step}.pt")
            records.append(record)
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "numerical-contract.json").write_text(json.dumps(records, indent=2))
    # Compare two different composite graphs as a bounded numerical diagnostic.
    # Exact weight bytes and N256 ordinary/compiled execution are independent
    # contracts; this tolerance is not model-quality qualification.
    assert all(row["bf16_compatibility_exact"] and row.get("ordinary_compiled_exact", True)
               and row.get("fp8_relative_rms_error", 0) < 0.005 for row in records), records


@HPU
def test_fused_quant_clamp_and_live_experts():
    torch._dynamo.reset()
    _, _, q13, s13, c13 = device_weights(*matrix(2304, 5120, 141))
    _, _, q2, s2, c2 = device_weights(*matrix(5120, 1152, 142))
    lut = mxfp4_bf16_lut("hpu")
    op = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    torch.manual_seed(173)
    for step, gain in enumerate((0.0, 1.0, 1000.0)):
        x = (torch.randn(1, 5120) * gain).bfloat16().to("hpu")
        ids = ((torch.arange(6) + step) % 2).reshape(1, 6).int().to("hpu")
        r = torch.softmax(torch.randn(1, 6), -1).mul_(1.5).to("hpu")
        operands = (x, ids, r, q13, q2, s13, s2, lut, c13, c2, True)
        reference = FMOE(*operands).cpu()
        ordinary = op(*operands).cpu()
        actual = compiled(*operands).cpu()
        exact = torch.equal(ordinary.view(torch.int16), actual.view(torch.int16))
        error = actual.float() - reference.float()
        relative = (error.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-30)).item()
        records.append({
            "gain": gain,
            "ordinary_compiled_exact": exact,
            "relative_rms_error": relative,
            "max_absolute_error": error.abs().max().item(),
            "reference_exact": torch.equal(reference.view(torch.int16), actual.view(torch.int16))
        })
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "fused-quant-contract.json").write_text(json.dumps(records, indent=2))
    assert all(row["ordinary_compiled_exact"] and row["relative_rms_error"] < 0.001 for row in records), records
