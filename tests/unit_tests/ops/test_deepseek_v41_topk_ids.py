# SPDX-License-Identifier: Apache-2.0
"""Preserve stock TopK tie order while carrying caller-owned row IDs."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def topk_ids():
    if os.getenv("DSV41_TEST_HPU") != "1":
        pytest.skip("Requires the qualified HPU runtime and an exclusive device lease")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    return torch.ops.custom_op.custom_deepseek_v41_topk_ids_gaudi2


def reference(scores, rows, width):
    values, offsets = scores.topk(width, dim=-1, sorted=False)
    return values, rows.gather(1, offsets)


def assert_same(actual, expected):
    for a, b in zip(actual, expected):
        assert torch.equal(a.cpu().view(torch.int32), b.cpu().view(torch.int32))


@pytest.mark.parametrize("tokens,columns,width", [
    (1, 1024, 512),
    (7, 13, 7),
    (128, 2048, 512),
    (257, 1024, 512),
    (4096, 1024, 512),
    (33, 2304, 2048),
    (7, 256, 256),
])
def test_dynamic_payload_and_ties(topk_ids, tokens, columns, width):
    torch.manual_seed(7203)
    for generation in range(2):
        scores = torch.randn(tokens, columns).bfloat16().float()
        scores[:, :6] = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan"), -1.])
        rows = torch.randint(-1, 100000, scores.shape, dtype=torch.int32)
        if generation:
            scores.zero_()
            rows = rows.flip(-1).clone()
        values, ids = scores.to("hpu"), rows.to("hpu")
        assert_same(topk_ids(values, ids, width), reference(values, ids, width))


def test_compiled_merge_consumes_fresh_inputs(topk_ids):

    def merge(a, a_ids, b, b_ids):
        return topk_ids(torch.cat((a, b), -1), torch.cat((a_ids, b_ids), -1), 512)

    compiled = torch.compile(merge, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(8021)
    for generation in range(3):
        values = torch.randn(33, 1024).bfloat16().float()
        ids = torch.randint(-1, 90000, values.shape, dtype=torch.int32)
        if generation == 1:
            values.fill_(-torch.inf)
        a, b = (part.clone().to("hpu") for part in values.chunk(2, -1))
        a_ids, b_ids = (part.clone().to("hpu") for part in ids.chunk(2, -1))
        actual = compiled(a, a_ids, b, b_ids)
        expected = reference(torch.cat((a, b), -1), torch.cat((a_ids, b_ids), -1), 512)
        assert_same(actual, expected)


def test_invalid_metadata(topk_ids):
    x = torch.empty(7, 1024, device="meta")
    rows = torch.empty(7, 1024, dtype=torch.int32, device="meta")
    for scores, ids, width in ((x, rows.long(), 512), (x, rows, 1025), (x, rows, 0), (x, rows[:1], 7)):
        with pytest.raises(RuntimeError, match="Index TopK"):
            topk_ids(scores, ids, width)
