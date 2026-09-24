# SPDX-License-Identifier: Apache-2.0
"""Native Sinkhorn keeps independent matrices isolated across vector tails."""

import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

import torch  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("tokens", [3, 17, 257])
def test_sinkhorn_vector_groups_and_tail_remain_independent(tokens):
    operation = torch.ops.custom_op.custom_deepseek_v4_sinkhorn4_gaudi2
    compiled = torch.compile(lambda value: operation(value), backend="hpu_backend", fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(239)
    value = (torch.randn(tokens, 4, 4, generator=generator) * 4).softmax(-1) + 1e-6
    oracle = value.double()
    for step in range(20):
        if step:
            oracle = oracle / (oracle.sum(-1, keepdim=True) + 1e-6)
        oracle = oracle / (oracle.sum(-2, keepdim=True) + 1e-6)
    actual = compiled(value.to("hpu")).cpu()
    torch.testing.assert_close(actual, oracle.float(), atol=3e-7, rtol=3e-6)

    # Rebinding and permuting tokens crosses SIMD groups and the partial tail.
    # No matrix may depend on a neighboring token or a previous invocation.
    order = torch.randperm(tokens, generator=generator)
    reordered = compiled(value[order].contiguous().to("hpu")).cpu()
    assert torch.equal(reordered, actual[order])
    zero = compiled(torch.zeros_like(value, device="hpu")).cpu()
    assert torch.equal(zero, torch.zeros_like(value))
