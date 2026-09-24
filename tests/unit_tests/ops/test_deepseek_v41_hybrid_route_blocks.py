# SPDX-License-Identifier: Apache-2.0
"""Route ownership at the hybrid 128/64 block boundaries."""

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_route_blocks import device_hybrid_route_blocks


@pytest.mark.parametrize("tokens,overflow_blocks", [(21, 0), (22, 1), (32, 1), (33, 2)])
def test_single_expert_overflow_preserves_each_route_once(tokens, overflow_blocks):
    ids = torch.full((tokens, 6), 17, dtype=torch.int32)
    base_ids, base_slots, tail_ids, tail_slots, occupied = device_hybrid_route_blocks(ids)
    assert occupied.tolist() == [1, overflow_blocks]
    assert base_ids[0, 0] == 17
    assert torch.equal(base_slots[0, :min(ids.numel(), 128)], torch.arange(min(ids.numel(), 128)))
    tail = tail_slots[:overflow_blocks].flatten()
    tail = tail[tail >= 0]
    assert torch.equal(tail, torch.arange(128, max(128, ids.numel())))
    assert bool((base_slots[1:] == -1).all())
    assert bool((tail_ids[0, overflow_blocks:] == -1).all())
