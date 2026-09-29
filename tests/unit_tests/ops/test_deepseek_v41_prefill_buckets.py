# SPDX-License-Identifier: Apache-2.0
"""Stable, bounded route ownership across expert occupancy buckets."""

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_prefill_buckets import device_bucketed_route_blocks, expert_bucket_width


def check_ownership(ids, experts=384, quantum=64):
    result = device_bucketed_route_blocks(ids, experts, quantum)
    flat = ids.flatten()
    live = []
    for index, rows in enumerate(range(quantum, 257, quantum)):
        expert_ids, slots = result[index * 2:index * 2 + 2]
        active = int(result[-1][index])
        width = expert_bucket_width(rows, quantum)
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


@pytest.mark.parametrize("quantum", [32, 64])
def test_tp4_full_chunk_and_halo_route_ownership(quantum):
    generator = torch.Generator().manual_seed(16400 + quantum)
    for tokens in (16384, 4096):
        ids = torch.randint(384, (tokens, 6), generator=generator, dtype=torch.int32)
        random_result = check_ownership(ids, quantum=quantum)
        skew_result = check_ownership(torch.full_like(ids, 383), quantum=quantum)
        assert [v.shape for v in random_result] == [v.shape for v in skew_result]


@pytest.mark.parametrize("ids", [
    torch.empty(0, 6, dtype=torch.int32),
    torch.zeros(2, 5, dtype=torch.int32),
    torch.zeros(2, 6, dtype=torch.float32)
])
def test_invalid_descriptor_contract_fails_before_dispatch(ids):
    with pytest.raises(ValueError):
        device_bucketed_route_blocks(ids)


def test_tp4_blocked_channel_layout_reuses_pair_plan_for_changing_occupancy(monkeypatch):
    from types import SimpleNamespace
    from vllm_gaudi.ops import deepseek_v41_grouped_prefill as grouped
    from vllm_gaudi.ops import deepseek_v41_prefill_buckets as buckets
    from vllm_gaudi.ops import deepseek_v41_prefill_columns as columns
    from vllm_gaudi.ops import deepseek_v41_prefill_plan as plans

    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE", "1")
    stream = SimpleNamespace(hpu_stream=0)
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.hpu, "default_stream", lambda: stream)
    monkeypatch.setattr(grouped, "compiled_single_prequant", lambda _: lambda value: (value, value[:, :1]))
    monkeypatch.setattr(columns, "compiled_permuted_reduce", lambda _: lambda value: value.sum(1))
    monkeypatch.setattr(columns, "compiled_permuted_single_fp8_write_body", lambda _: "full")
    monkeypatch.setattr(columns, "compiled_permuted_single_fp8_pair_tail", lambda _: "pair")
    occupancy = iter((25, 29, 48, 74))

    def descriptors(ids, experts, quantum):
        assert quantum == 64
        result = []
        for rows in range(quantum, 257, quantum):
            result.extend((torch.arange(96).reshape(1, -1), torch.zeros(96, rows, dtype=torch.int32)))
        return (*result, torch.tensor([next(occupancy), 0, 0, 0]))

    monkeypatch.setattr(buckets, "_compiled_routes", lambda _: descriptors)
    created, replayed = [], []

    class Plan:

        def __init__(self, body, arguments, groups, require_prefix=False):
            self.body, self.recipes = body, len(groups)
            created.append((body, [indices.tolist() for indices in groups]))

        def replay(self, arguments, count):
            replayed.append((self.body, count, arguments[-1].tolist() if self.body == "pair" else None))
            return count

    monkeypatch.setattr(plans, "PrefillExpertPlan", Plan)
    executor = buckets._BucketedPrefillPlans()
    value, ids = torch.zeros(128, 8, dtype=torch.bfloat16), torch.zeros(128, 6, dtype=torch.int32)
    q = torch.empty(384, 5, 1, dtype=torch.int16)
    # The real N256 loader retains channel scales in blocked N256 layout.
    channel = torch.empty(384, 5, 256, dtype=torch.bfloat16)
    for _ in range(4):
        executor(value, ids, torch.zeros(128, 6), q, q, q, q, torch.ones(16), True, channel)
    assert [body for body, _ in created] == ["full", "pair"]
    assert created[1][1] == [[i, i + 1] for i in range(0, 24, 2)]
    assert [(body, count) for body, count, _ in replayed] == [("full", 1), ("pair", 3), ("full", 2), ("full", 3),
                                                              ("pair", 1)]
    assert replayed[1][2] == list(range(24, 48))
    assert replayed[-1][2] == list(range(72, 96))
