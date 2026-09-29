# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_route_blocks import (device_route_blocks, route_block_capacity)


@pytest.mark.parametrize("rows", [32, 64, 128, 512])
@pytest.mark.parametrize("tokens,skew", [(1, False), (73, False), (8192, False), (8192, True)])
def test_fixed_descriptor_preserves_all_routes_and_expert_order(rows, tokens, skew):
    torch.manual_seed(41)
    ids = torch.randint(0, 384, (tokens, 6), dtype=torch.int32)
    if skew:
        ids.fill_(17)
    experts, slots, inverse, counts = device_route_blocks(ids, rows=rows)
    assert slots.shape == (route_block_capacity(ids.numel(), 384, rows), rows)
    assert int(counts.sum()) == ids.numel()
    active_blocks = int(((counts + rows - 1) // rows).sum())
    assert 1 <= active_blocks <= slots.shape[0]
    assert bool((slots[active_blocks:] == -1).all())
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


@pytest.mark.parametrize("rows", [4, 8, 16])
@pytest.mark.parametrize("batch", [1, 4, 8, 32, 64])
def test_concurrent_blocks_preserve_routes_through_occupancy_changes(rows, batch):
    generator = torch.Generator().manual_seed(6401)
    routes = torch.randint(0, 384, (batch, 6), dtype=torch.int32, generator=generator)
    shape = None
    for ids in (routes, torch.zeros_like(routes), torch.arange(batch * 6).reshape(batch, 6).int() % 384):
        experts, slots, inverse, counts = device_route_blocks(ids, rows=rows)
        if shape is not None:
            assert [value.shape for value in (experts, slots, inverse, counts)] == shape
        shape = [value.shape for value in (experts, slots, inverse, counts)]
        assert counts.sum() == ids.numel()
        assert torch.equal(slots[slots >= 0].sort().values, torch.arange(ids.numel()))
        assert torch.equal(slots.flatten()[inverse], torch.arange(ids.numel()))
        assert torch.equal(experts.T.expand_as(slots)[slots >= 0], ids.flatten()[slots[slots >= 0]])
        for group in slots:
            real = group[group >= 0]
            assert torch.equal(real, real.sort().values)
        # Arbitrary padding values cannot enter the restored route order.
        data = torch.full_like(slots, -999)
        data[slots >= 0] = slots[slots >= 0] * 17 + 3
        assert torch.equal(data.flatten()[inverse], torch.arange(ids.numel()) * 17 + 3)
