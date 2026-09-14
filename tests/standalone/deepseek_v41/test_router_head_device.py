# SPDX-License-Identifier: Apache-2.0
"""Actual TPC selection and mixed-output GEMM contracts on a leased device."""
import json
import os
from pathlib import Path

import pytest

from test_native_moe import HPU, torch

ROUTER = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2
HEAD = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2


@HPU
@pytest.mark.parametrize("tokens", [1, 3, 129, 512])
def test_router_score_bias_ties_and_changing_inputs(tokens):
    torch.manual_seed(641)
    compiled = torch.compile(ROUTER, backend="hpu_backend", fullgraph=True, dynamic=False)
    args = [
        torch.empty(tokens, 384, device="hpu"),
        torch.empty(384, device="hpu"),
        torch.empty(384, device="hpu"),
        torch.empty(tokens, device="hpu", dtype=torch.bool)
    ]
    records = []
    for step in range(5):
        scores = torch.nn.functional.softplus(torch.randn(tokens, 384) * 30).sqrt()
        text, image = torch.randn(384), torch.randn(384)
        mask = ((torch.arange(tokens) + step) % 2).bool()
        if step == 1:
            scores.fill_(1)
            text.zero_()
            image.zero_()
        if step == 2:
            scores.zero_()
        if step == 3:
            scores.fill_(1e-18)
            text.fill_(-1e10)
            image.fill_(1e10)
        if step == 4:
            text.fill_(-float("inf"))
            image.fill_(float("inf"))
        choice = scores + torch.where(mask[:, None], image, text)
        ids = torch.argsort(choice, descending=True, stable=True)[:, :6]
        original = scores.gather(1, ids)
        reference = original / (original.sum(-1, keepdim=True) + 1e-20) * 1.5
        for target, source in zip(args, [scores, text, image, mask], strict=True):
            target.copy_(source)
        actual_ids, weights = (x.cpu() for x in compiled(*args))
        assert torch.equal(actual_ids, ids.int())
        assert all(len(set(row)) == 6 for row in actual_ids.tolist())
        torch.testing.assert_close(weights, reference, atol=2e-7, rtol=2e-6)
        ordinary = tuple(x.cpu() for x in ROUTER(*args))
        assert torch.equal(actual_ids, ordinary[0]) and torch.equal(weights, ordinary[1])
        records.append({"step": step, "max_weight_error": float((weights - reference).abs().max())})
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"router-{tokens}.json").write_text(json.dumps(records, indent=2))


@HPU
@pytest.mark.parametrize("tokens,width", [(1, 64640), (3, 1024), (129, 1024)])
def test_head_bf16_inputs_accumulate_and_return_fp32(tokens, width):
    torch.manual_seed(1541)
    # Dot products of small BF16 integers have exact FP32 sums here. This
    # distinguishes FP32 logits from a BF16 result subsequently cast to FP32.
    weight_cpu = torch.randint(-2, 3, (width, 5120)).bfloat16()
    weight = weight_cpu.to("hpu")
    hidden = torch.empty(tokens, 5120, device="hpu", dtype=torch.bfloat16)
    compiled = torch.compile(HEAD, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    for step in range(2):
        cpu = torch.randint(-2, 3, (tokens, 5120)).bfloat16()
        hidden.copy_(cpu)
        actual = compiled(hidden, weight).cpu()
        expected = torch.nn.functional.linear(cpu.float(), weight_cpu.float())
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        assert torch.any(actual != actual.bfloat16().float()), "test must exercise low FP32 mantissa bits"
        assert torch.equal(HEAD(hidden, weight).cpu(), actual)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        torch.testing.assert_close(torch.log_softmax(actual, -1), torch.log_softmax(expected, -1), atol=0, rtol=0)
        best = actual.topk(2).values
        records.append({"step": step, "smallest_top2_margin": float((best[:, 0] - best[:, 1]).min())})
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"head-{tokens}-{width}.json").write_text(json.dumps(records, indent=2))
