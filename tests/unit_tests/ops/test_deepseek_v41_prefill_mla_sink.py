# SPDX-License-Identifier: Apache-2.0
"""Sparse prefill preserves sink mass, invalid-row isolation and tail inputs."""

import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

import torch  # noqa: E402

from flashinfer_gaudi.mla import sparse_mla_prefill  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_prefill_mla import sparse_prefill_mla  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("tokens,heads,columns", [(17, 3, 17), (129, 32, 129), (513, 64, 640)])
def test_changing_selection_lengths_and_storage(tokens, heads, columns):
    torch.manual_seed(tokens)
    candidate = torch.compile(sparse_prefill_mla, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(sparse_mla_prefill, backend="hpu_backend", fullgraph=True, dynamic=False)
    with torch.inference_mode():
        query = torch.randn(tokens, heads, 512).bfloat16().to("hpu")
        cache = torch.randn(2051, 512).bfloat16().to("hpu")
        cache[0].fill_(float("nan"))
        ids = torch.randint(1, 2051, (tokens, columns), dtype=torch.int32).to("hpu")
        ids[:, :2] = -1
        ids[:, 2] = 2051
        sink = torch.randn(heads, dtype=torch.float32, device="hpu")
        lengths = (torch.arange(tokens, dtype=torch.int32, device="hpu") * 13).remainder(columns + 1)
        for change in range(2):
            if change:
                query = query.flip(0).contiguous() * .75
                ids = ids.flip(0).contiguous()
                sink.mul_(2)
            expected = reference(query, cache, ids, sink, lengths, query_tile=128).float().cpu()
            actual = candidate(query, cache, ids, sink, lengths, query_tile=128).float().cpu()
            assert bool(actual.isfinite().all())
            assert bool((actual[0] == 0).all()), "An empty selection must retain only zero sink contribution"
            relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
            assert relative < 3e-4


def test_fp32_sink_changes_denominator_without_a_value_row():
    with torch.inference_mode():
        query = torch.zeros(2, 2, 512, dtype=torch.bfloat16, device="hpu")
        cache = torch.ones(5, 512, dtype=torch.bfloat16, device="hpu")
        cache[0].fill_(float("nan"))
        ids = torch.tensor([[1, 2, 3, 4], [-1, -1, -1, -1]], dtype=torch.int32, device="hpu")
        sink = torch.tensor([0., 4.], dtype=torch.float32, device="hpu").log()
        sink[0] = 0
        lengths = torch.tensor([4, 0], dtype=torch.int32, device="hpu")
        actual = sparse_prefill_mla(query, cache, ids, sink, lengths).float().cpu()
        expected = torch.tensor([.8, .5]).bfloat16().float()
        torch.testing.assert_close(actual[0, :, 0], expected, rtol=0, atol=0)
        assert bool((actual[1] == 0).all()) and bool(actual.isfinite().all())


def test_native_bmm_rejects_invalid_reduction_extent():
    op = torch.ops.custom_op.custom_deepseek_v41_prefill_bmm_f32_gaudi2
    with pytest.raises(RuntimeError, match="matching"):
        op(torch.empty(2, 3, 512, dtype=torch.bfloat16, device="meta"),
           torch.empty(2, 17, 511, dtype=torch.bfloat16, device="meta"), True)
