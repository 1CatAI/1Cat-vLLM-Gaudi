# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_grouped_decode import ordered_route_rows


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 16, 32, 64])
def test_unpadded_order_survives_changed_experts_and_duplicate_routes(batch):
    gen = torch.Generator().manual_seed(608 + batch)
    random = torch.randint(0, 384, (batch, 6), generator=gen, dtype=torch.int32)
    for ids in (random, torch.full_like(random, 383), random.remainder(7), random.flip(0)):
        experts, slots, inverse = ordered_route_rows(ids)
        expected = sorted(range(ids.numel()), key=lambda i: (int(ids.flatten()[i]), i))
        assert slots.shape == (ids.numel(), 1)
        assert slots[:, 0].tolist() == expected
        assert torch.equal(experts.flatten(), ids.flatten()[expected])
        assert torch.equal(slots.flatten()[inverse], torch.arange(ids.numel()))
        # Every top6 route retains its original position before reduction.
        payload = torch.arange(ids.numel()).reshape(batch, 6) * 17 + 3
        restored = payload.flatten()[slots.flatten()][inverse].reshape_as(payload)
        assert torch.equal(payload, restored)


def test_invalid_route_shape_fails_before_dispatch():
    for ids in (torch.zeros(1, 5, dtype=torch.int32), torch.zeros(65, 6, dtype=torch.int32), torch.zeros(1, 6)):
        with pytest.raises(ValueError):
            ordered_route_rows(ids)
