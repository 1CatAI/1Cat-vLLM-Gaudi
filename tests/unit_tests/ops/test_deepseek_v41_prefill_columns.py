# SPDX-License-Identifier: Apache-2.0
"""The private column layout preserves projections and ordered BF16 reduction."""

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_grouped_prefill import ordered_reduce
from vllm_gaudi.ops.deepseek_v41_prefill_columns import permuted_ordered_reduce, restore_expert_columns


def physical_columns(width):
    return torch.tensor([
        column for tile in range(0, width, 256) for parity in (0, 1) for column in range(tile + parity, tile + 256, 2)
    ])


@pytest.mark.parametrize("width", [256, 2304, 5120])
def test_ordered_reduction_commutes_with_private_column_layout(width):
    generator = torch.Generator().manual_seed(width)
    value = torch.randn(3, 6, width, generator=generator).to(torch.bfloat16)
    # Cancellation and unequal magnitudes make changing the accumulation
    # order observable, including at the W13 gate/up half-tile boundary.
    value[:, 1] = -value[:, 0]
    value[:, 3].mul_(256)
    value[:, 4] = -value[:, 3]
    actual = permuted_ordered_reduce(value.index_select(-1, physical_columns(width)))
    expected = ordered_reduce(value)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_w13_gate_and_up_columns_survive_the_1152_column_boundary():
    generator = torch.Generator().manual_seed(20260924)
    inputs = torch.randint(-4, 5, (2, 3, 128), generator=generator).float()
    weights = torch.randint(-8, 9, (2, 128, 2304), generator=generator).float() * .5
    expected = torch.bmm(inputs, weights)
    actual = restore_expert_columns(torch.bmm(inputs, weights.index_select(-1, physical_columns(2304))))
    assert torch.equal(actual[..., :1152], expected[..., :1152])
    assert torch.equal(actual[..., 1152:], expected[..., 1152:])
