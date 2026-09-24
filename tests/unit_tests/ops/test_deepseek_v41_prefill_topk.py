# SPDX-License-Identifier: Apache-2.0
"""Native BF16-valued score selection, including deterministic cutoff ties."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def native_topk():
    if os.getenv("DSV41_TEST_HPU") != "1":
        pytest.skip("Requires the qualified HPU runtime and an exclusive device lease")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    return torch.ops.custom_op.custom_deepseek_v41_prefill_score_topk_gaudi2


def reference(score, width):
    selected = score.argsort(dim=-1, descending=True, stable=True)[:, :width].sort(-1).values
    return score.gather(1, selected), selected.int()


@pytest.mark.parametrize("batch,columns,width", [
    (1, 64, 1),
    (1, 64, 64),
    (7, 256, 64),
    (7, 256, 256),
    (33, 2048, 512),
    (127, 2304, 2048),
    (513, 4096, 2048),
    (8191, 64, 7),
])
def test_shapes_and_changing_scores(native_topk, batch, columns, width):
    torch.manual_seed(1729)
    for generation in range(2):
        scores = torch.randn(batch, columns).bfloat16().float()
        scores[:, :6] = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan"), -1.])
        if generation:
            scores = scores.roll(1, -1)
        expected = reference(scores, width)
        actual = native_topk(scores.to("hpu"), width)
        assert torch.equal(actual[0].cpu().view(torch.int32), expected[0].view(torch.int32))
        assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("mode", ["zero", "negative_zero", "negative_inf", "all_bf16_codes"])
def test_cutoff_ties_and_encodings(native_topk, mode):
    if mode == "all_bf16_codes":
        scores = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().reshape(16, 4096)
    else:
        scores = torch.full((7, 4096), {"zero": 0., "negative_zero": -0., "negative_inf": -float("inf")}[mode])
    expected = reference(scores, 2048)
    actual = native_topk(scores.to("hpu"), 2048)
    assert torch.equal(actual[0].cpu().view(torch.int32), expected[0].view(torch.int32))
    assert torch.equal(actual[1].cpu(), expected[1])


def test_compiled_replay_consumes_changed_scores(native_topk):

    def select(x):
        return native_topk(x, 512)

    compiled = torch.compile(select, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(5021)
    slot = torch.empty(33, 2048, device="hpu")
    for _ in range(3):
        source = torch.randn(33, 2048).bfloat16().float()
        slot.copy_(source)
        actual = compiled(slot)
        expected = reference(source, 512)
        assert torch.equal(actual[0].cpu(), expected[0])
        assert torch.equal(actual[1].cpu(), expected[1])


def test_reject_invalid_metadata(native_topk):
    for columns, width in ((63, 1), (4160, 512), (128, 129), (4096, 2049)):
        with pytest.raises(RuntimeError, match="Prefill top-k"):
            native_topk(torch.empty(7, columns, device="meta"), width)
