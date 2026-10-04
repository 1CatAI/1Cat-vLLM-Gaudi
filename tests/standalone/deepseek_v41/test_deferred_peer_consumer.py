# SPDX-License-Identifier: Apache-2.0
"""Deferred communication retains rank-major rows and the ordinary reduction API."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives


@pytest.mark.parametrize('ranks', [4, 8])
def test_deferred_shards_are_not_reduced_or_reordered(monkeypatch, ranks):
    shards = torch.arange(ranks*5120).reshape(ranks, 5120).bfloat16()
    monkeypatch.setattr(torch.ops.vllm_gaudi, 'tp_peer_allgather', lambda value, size: shards,
                        raising=False)
    reduce, _ = stage_collectives(0, True, ranks, ordered_peer_sum=False)
    value = torch.zeros(1, 5120, dtype=torch.bfloat16)
    deferred = reduce(value, defer=True)
    assert deferred.shape == (ranks, 1, 5120)
    assert torch.equal(deferred[:, 0], shards)
    expected = shards[0].float()
    for row in shards[1:]:
        expected = expected + row.float()
    assert torch.equal(reduce(value), expected.bfloat16().reshape_as(value))


@pytest.mark.parametrize('rank', [0, 1])
def test_two_rank_deferred_rows_keep_global_rank_order(monkeypatch, rank):
    rows = torch.tensor([[1.]*128, [-2.]*128], dtype=torch.bfloat16)
    monkeypatch.setattr(torch.ops.vllm_gaudi, 'tp2_exchange_peer', lambda value: rows[1-rank].reshape(1, -1),
                        raising=False)
    reduce, _ = stage_collectives(rank, True, 2)
    assert torch.equal(reduce(rows[rank].reshape(1, -1), defer=True), rows.reshape(2, 1, 128))


def test_deferred_consumer_cannot_silently_use_a_different_collective():
    reduce, _ = stage_collectives(0, True, 4)
    with pytest.raises(ValueError, match='small native BF16'):
        reduce(torch.ones(1, 5120, dtype=torch.float32), defer=True)
    reduce, _ = stage_collectives(0, False, 4)
    with pytest.raises(ValueError, match='small native BF16'):
        reduce(torch.ones(1, 5120, dtype=torch.bfloat16), defer=True)
