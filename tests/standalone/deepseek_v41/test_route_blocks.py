# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_route_blocks import (device_hybrid_route_blocks, device_route_blocks,
                                                      route_block_capacity)


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


def test_full_32k_512_row_descriptors_preserve_order_and_tail():
    # This is the layer-major experiment's largest prompt bucket. It must
    # retain every one of the six route slots and keep all padding in a suffix.
    tokens, rows = 32768, 512
    ids = (torch.arange(tokens * 6, dtype=torch.int32).reshape(tokens, 6) * 13) % 384
    experts, slots, inverse, counts = device_route_blocks(ids, rows=rows)
    active_blocks = int(((counts + rows - 1) // rows).sum())
    assert active_blocks <= slots.shape[0]
    assert bool((slots[active_blocks:] == -1).all())
    valid = slots >= 0
    assert valid.sum() == tokens * 6
    assert torch.equal(slots[valid].sort().values, torch.arange(tokens * 6))
    assert torch.equal(experts.T.expand_as(slots)[valid], ids.flatten()[slots[valid]])
    assert torch.equal(slots.flatten()[inverse], torch.arange(tokens * 6))


@pytest.mark.parametrize("tokens", [1, 73, 512, 8192])
@pytest.mark.parametrize("distribution", ["random", "single", "heavy"])
def test_hybrid_descriptors_preserve_all_routes_and_expert_order(tokens, distribution):
    torch.manual_seed(tokens)
    ids = torch.randint(0, 384, (tokens, 6), dtype=torch.int32)
    if distribution == "single":
        ids.fill_(17)
    elif distribution == "heavy":
        ids.reshape(-1)[::8] = 17
    base_ids, base_slots, tail_ids, tail_slots, occupied = device_hybrid_route_blocks(ids)
    routes = ids.numel()
    assert base_slots.shape == (min(384, routes), 128)
    assert tail_slots.shape == ((routes + 63) // 64, 64)
    assert base_ids.shape == (1, base_slots.shape[0])
    assert tail_ids.shape == (1, tail_slots.shape[0])
    base_active, tail_active = map(int, occupied)
    assert base_active == ids.unique().numel()
    assert 0 <= tail_active <= tail_slots.shape[0]
    assert bool((base_slots[base_active:] == -1).all())
    assert bool((tail_slots[tail_active:] == -1).all())
    all_slots = torch.cat((base_slots.flatten(), tail_slots.flatten()))
    valid = all_slots >= 0
    assert torch.equal(all_slots[valid].sort().values, torch.arange(routes))
    for expert in ids.unique():
        first = base_slots[base_ids.flatten() == expert].flatten()
        overflow = tail_slots[tail_ids.flatten() == expert].flatten()
        actual = torch.cat((first[first >= 0], overflow[overflow >= 0]))
        expected = (ids.flatten() == expert).nonzero().flatten()
        assert torch.equal(actual, expected)


def test_hybrid_descriptor_shapes_do_not_depend_on_occupancy():
    random = torch.randint(0, 384, (509, 6), dtype=torch.int32)
    skewed = torch.full_like(random, 17)
    assert [value.shape for value in device_hybrid_route_blocks(random)
            ] == [value.shape for value in device_hybrid_route_blocks(skewed)]


def test_hybrid_descriptors_reject_unsupported_shapes_and_tile_sizes():
    ids = torch.zeros((2, 6), dtype=torch.int32)
    with pytest.raises(ValueError, match="integer"):
        device_hybrid_route_blocks(ids.float())
    with pytest.raises(ValueError, match=r"128\+64"):
        device_hybrid_route_blocks(ids, tail_rows=32)
