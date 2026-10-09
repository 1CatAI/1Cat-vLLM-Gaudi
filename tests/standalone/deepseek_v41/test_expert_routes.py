# SPDX-License-Identifier: Apache-2.0
"""Unique-expert descriptors retain exact route order under changing IDs."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_expert_routes import pack_expert_routes


@pytest.mark.parametrize("rows", [1, 2, 6])
@pytest.mark.parametrize("shared", [False, True])
def test_route_descriptor_roundtrip(rows, shared):
    ids = torch.arange(rows * 6).reshape(rows, 6)
    if shared:
        ids = ids % 6
    # Reverse both axes: order within an expert must remain original route order.
    ids = ids.flip((0, 1)).to(torch.int32)
    experts, counts, slots, inverse, inverse_row = pack_expert_routes(ids)
    assert counts.sum().item() == ids.numel()
    assert counts.max().item() <= rows
    assert torch.equal(experts.long().gather(0, inverse), ids.flatten().long())
    recovered = slots[inverse, inverse_row]
    assert torch.equal(recovered.long(), torch.arange(ids.numel()))
    assert torch.all(experts[counts == 0] == -1)
    assert torch.all(slots[counts == 0] == -1)
    for expert, count, row_slots in zip(experts.tolist(), counts.tolist(), slots.tolist(), strict=True):
        actual = [slot for slot, value in enumerate(ids.flatten().tolist()) if value == expert]
        assert row_slots[:count] == actual
        assert row_slots[count:] == [-1] * (rows - count)


def test_scatter_back_keeps_original_weighted_bf16_reduction_order():
    ids = torch.tensor([[3, 9, 2, 8, 7, 1], [9, 3, 8, 2, 1, 7]], dtype=torch.int32)
    _, _, slots, inverse, inverse_row = pack_expert_routes(ids)
    routes = torch.randn(12, 32).bfloat16()
    packed = routes[slots.clamp_min(0).long()]
    recovered = packed[inverse, inverse_row]
    assert torch.equal(recovered, routes)
    weights = torch.randn(2, 6, 1)
    reference = (routes.reshape(2, 6, 32).float() * weights).sum(1).bfloat16()
    actual = (recovered.reshape(2, 6, 32).float() * weights).sum(1).bfloat16()
    assert torch.equal(actual, reference)
