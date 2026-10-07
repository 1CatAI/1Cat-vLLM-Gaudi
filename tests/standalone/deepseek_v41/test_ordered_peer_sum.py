# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_peer_sum_reference


@pytest.mark.parametrize('ranks', [2, 4, 8])
def test_reference_keeps_rank_order_and_final_bf16_boundary(ranks):
    values = torch.randn((ranks, 5120), generator=torch.Generator().manual_seed(42)).bfloat16()
    # Exact cancellations expose reordered addition and intermediate BF16 rounding.
    values[0, :4] = torch.tensor([256., 1., -0., 512.])
    values[1, :4] = torch.tensor([1., -256., 0., -512.])
    if ranks > 2:
        values[2, :4] = torch.tensor([-256., 256., -0., 0.125])
    expected = values[0].float()
    for row in values[1:]:
        expected = expected + row.float()
    actual = ordered_peer_sum_reference(values)
    assert torch.equal(actual.view(torch.int16), expected.bfloat16().reshape(1, -1).view(torch.int16))
