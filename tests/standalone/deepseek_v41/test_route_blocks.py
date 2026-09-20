# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_route_blocks import device_route_blocks, route_block_capacity


@pytest.mark.parametrize("rows", [32, 64, 128])
@pytest.mark.parametrize("tokens,skew", [(1, False), (73, False), (8192, False), (8192, True)])
def test_fixed_descriptor_preserves_all_routes_and_expert_order(rows, tokens, skew):
    torch.manual_seed(41)
    ids = torch.randint(0, 384, (tokens, 6), dtype=torch.int32)
    if skew:
        ids.fill_(17)
    experts, slots, inverse, counts = device_route_blocks(ids, rows=rows)
    assert slots.shape == (route_block_capacity(ids.numel(), 384, rows), rows)
    assert int(counts.sum()) == ids.numel()
    valid = slots >= 0
    assert torch.equal(slots[valid].sort().values, torch.arange(ids.numel()))
    assert torch.equal(experts.T.expand_as(slots)[valid], ids.flatten()[slots[valid]])
    assert torch.equal(slots.flatten()[inverse], torch.arange(ids.numel()))
    for expert in ids.unique():
        actual = slots[(experts.flatten() == expert)]
        actual = actual[actual >= 0]
        expected = (ids.flatten() == expert).nonzero().flatten()
        assert torch.equal(actual, expected)


def test_occupancy_changes_do_not_change_descriptor_shapes():
    random = torch.randint(0, 384, (513, 6), dtype=torch.int32)
    concentrated = torch.full_like(random, 7)
    left, right = device_route_blocks(random), device_route_blocks(concentrated)
    assert [x.shape for x in left] == [x.shape for x in right]
