# SPDX-License-Identifier: Apache-2.0
"""Stable, bounded route ownership across expert occupancy buckets."""

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_prefill_buckets import device_bucketed_route_blocks, expert_bucket_width


def check_ownership(ids, experts=384):
    result = device_bucketed_route_blocks(ids, experts)
    flat = ids.flatten()
    live = []
    for index, rows in enumerate((64, 128, 192, 256)):
        expert_ids, slots = result[index * 2:index * 2 + 2]
        active = int(result[-1][index])
        width = expert_bucket_width(rows)
        assert slots.shape[1] == rows and slots.shape[0] % width == 0
        assert rows * width <= 4096
        assert int((expert_ids >= 0).sum()) == active
        assert bool((slots[active:] == -1).all())
        for block in range(active):
            valid = slots[block][slots[block] >= 0]
            assert valid.numel() > 0
            assert bool((flat[valid] == expert_ids[0, block]).all())
            assert bool((valid[1:] > valid[:-1]).all())
            live.append(valid)
    assert torch.equal(torch.cat(live).sort().values, torch.arange(ids.numel()))
    return result


@pytest.mark.parametrize("tokens", [1, 73, 128, 2048, 8192])
def test_every_route_is_owned_once_under_skew_and_changing_occupancy(tokens):
    generator = torch.Generator().manual_seed(tokens)
    random_ids = torch.randint(384, (tokens, 6), generator=generator, dtype=torch.int32)
    random_result = check_ownership(random_ids)
    skew_result = check_ownership(torch.full_like(random_ids, 17))
    assert [x.shape for x in random_result] == [x.shape for x in skew_result]


@pytest.mark.parametrize("count", [63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 512, 513, 1025])
def test_remainders_and_full_slabs_retain_token_topk_order(count):
    ids = torch.ones(((count + 6) // 6) * 6, dtype=torch.int64)
    ids[:count] = 0
    result = check_ownership(ids.reshape(-1, 6), experts=2)
    expert_zero_blocks = sum(int((result[i] == 0).sum()) for i in (0, 2, 4, 6))
    assert expert_zero_blocks == (count + 255) // 256


@pytest.mark.parametrize("ids", [
    torch.empty(0, 6, dtype=torch.int32),
    torch.zeros(2, 5, dtype=torch.int32),
    torch.zeros(2, 6, dtype=torch.float32)
])
def test_invalid_descriptor_contract_fails_before_dispatch(ids):
    with pytest.raises(ValueError):
        device_bucketed_route_blocks(ids)
