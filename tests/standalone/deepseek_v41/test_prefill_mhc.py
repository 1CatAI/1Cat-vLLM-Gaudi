# SPDX-License-Identifier: Apache-2.0
"""Ordered prompt mixing contracts; device tests require a leased module."""
import os

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import hc_post
from vllm_gaudi.models.deepseek_v41_program import _prefill_hc_post


def test_prompt_switch_preserves_cpu_reference_and_decode(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_MHC_POST", "1")
    x = torch.randn(3, 5120).bfloat16()
    r = torch.randn(3, 4, 5120).bfloat16()
    p, c = torch.rand(3, 4), torch.rand(3, 4, 4)
    expected = ((c.unsqueeze(-1) * r.float().unsqueeze(2)).sum(1) + x.float().unsqueeze(1) * p.unsqueeze(-1)).bfloat16()
    assert torch.equal(hc_post(x, r, p, c), expected)
    assert torch.equal(_prefill_hc_post(x, r, p, c), expected)


@pytest.mark.skipif(os.environ.get("DSV41_TEST_HPU") != "1", reason="requires an explicit HPU lease")
@pytest.mark.parametrize("tokens", [1, 127, 8191, 8192])
def test_native_prompt_post_preserves_every_bf16_bit(tokens):
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    op = torch.ops.custom_op.custom_deepseek_v41_prefill_mhc_post_gaudi2
    for change in range(2):
        torch.manual_seed(5300 + change)
        x, r = torch.randn(tokens, 5120).bfloat16(), torch.randn(tokens, 4, 5120).bfloat16()
        post, comb = torch.rand(tokens, 4) * 2, torch.rand(tokens, 4, 4)
        if change:
            # Signed zero and cancellation expose lost product/accumulation
            # boundaries. Native floating operations follow the device rules.
            x[:, :4] = torch.tensor([0., -0., 1., -1.])
            r[:, :, :4] = x[:, None, :4]
            comb[:, 1] = -comb[:, 0]
        x, r, post, comb = (v.to("hpu") for v in (x, r, post, comb))
        mixed = (comb.unsqueeze(-1) * r.float().unsqueeze(2)).sum(1)
        expected = (x.float().unsqueeze(1) * post.unsqueeze(-1) + mixed).bfloat16()
        actual = op(x, r, post, comb)
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.skipif(os.environ.get("DSV41_TEST_HPU") != "1", reason="requires an explicit HPU lease")
@pytest.mark.parametrize("tokens", [1, 127, 8191, 8192])
def test_native_prompt_collapse_preserves_every_bf16_bit(tokens):
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    op = torch.ops.custom_op.custom_deepseek_v41_prefill_mhc_collapse_gaudi2
    for change in range(2):
        torch.manual_seed(6300 + change)
        residual = torch.randn(tokens, 4, 5120).bfloat16()
        previous_pre = torch.rand(tokens, 4)
        if change:
            residual[:, :, :4] = torch.tensor([0., -0., 1., -1.])
            previous_pre[:, 1] = -previous_pre[:, 0]
        residual, previous_pre = (value.to("hpu") for value in (residual, previous_pre))
        expected = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).bfloat16()
        actual = op(residual, previous_pre)
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
