# SPDX-License-Identifier: Apache-2.0
"""Prefill MLA's public and glue contracts must agree beyond decode shapes."""

import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

import torch  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("tokens,rows", [(1, 128), (6, 512), (16, 131072), (32, 2048), (64, 8192)])
def test_prefill_mla_tile_cache_and_changed_inputs(tokens, rows):
    torch.manual_seed(771 + tokens)
    op = torch.ops.custom_op.custom_deepseek_v41_prefill_mla_mme_gaudi2
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    cache = torch.randn(rows, 512).bfloat16()
    ids = ((torch.arange(tokens * 640).reshape(tokens, 640) * 17) % rows).int()
    ids[:, :3] = torch.tensor([-1, rows, 0], dtype=torch.int32)
    lengths = ((torch.arange(tokens) * 83) % 641).int()
    lengths[-1] = 640
    sink = torch.randn(32)
    scale = torch.tensor([512**-0.5])
    device = [x.to("hpu") for x in (cache, ids, sink, scale, lengths)]
    for _ in range(2):
        q = torch.randn(tokens, 32, 512).bfloat16()
        query = q.to("hpu")
        ordinary = op(query, *device).cpu()
        actual = compiled(query, *device).cpu()
        assert torch.equal(actual, ordinary), "Eager and compiled MLA differ"
        valid = (ids >= 0) & (ids < rows) & (torch.arange(640)[None, :] < lengths[:, None])
        selected = cache[ids.clamp(0, rows - 1)].float().masked_fill(~valid[:, :, None], 0)
        logits = torch.bmm(q.float(), selected.transpose(1, 2)) * scale
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        logits = torch.cat((logits, sink[None, :, None].expand(tokens, -1, -1)), -1)
        expected = torch.bmm(logits.softmax(-1)[..., :-1], selected).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.004)
