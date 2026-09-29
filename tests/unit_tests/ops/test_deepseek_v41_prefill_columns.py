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


@pytest.mark.parametrize("width", [256, 1280, 2304, 5120])
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


@pytest.mark.parametrize("intermediate", [640, 1152])
def test_w13_gate_and_up_columns_survive_the_half_tile_boundary(intermediate):
    generator = torch.Generator().manual_seed(20260924)
    inputs = torch.randint(-4, 5, (2, 3, 128), generator=generator).float()
    width = 2 * intermediate
    weights = torch.randint(-8, 9, (2, 128, width), generator=generator).float() * .5
    expected = torch.bmm(inputs, weights)
    actual = restore_expert_columns(torch.bmm(inputs, weights.index_select(-1, physical_columns(width))))
    assert torch.equal(actual[..., :intermediate], expected[..., :intermediate])
    assert torch.equal(actual[..., intermediate:], expected[..., intermediate:])


@pytest.mark.parametrize("width", [16, 24])
def test_fixed_pair_tail_preserves_every_live_route_and_nonzero_offset(monkeypatch, width):
    from vllm_gaudi.ops import deepseek_v41_prefill_columns as columns
    rows = 32
    for remainder in range(1, width):
        active = width + remainder
        slots = torch.arange(2 * width * rows).reshape(2 * width, rows)
        slots[active:] = -1

        def consume(*arguments, slots=slots):
            return slots.index_select(0, arguments[-1].long()).flatten()

        monkeypatch.setattr(columns, "permuted_body_single_fp8_write", consume)
        tail_indices = torch.arange(width, 2 * width, dtype=torch.int32)
        chunks = [
            columns.permuted_body_single_fp8_pair_tail(*([None] * 14), tail_indices,
                                                       torch.arange(start, start + 2, dtype=torch.int32))
            for start in range(0, remainder, 2)
        ]
        observed = torch.cat(chunks)
        assert torch.equal(observed[observed >= 0], torch.arange(width * rows, active * rows))
