# SPDX-License-Identifier: Apache-2.0
import torch
from vllm_gaudi.ops import deepseek_v41_indexer as target
from vllm_gaudi.ops import deepseek_v41_index_mirror as mirror


def test_wide_reindex_keeps_all_candidate_slots_and_uses_shared_scorer(monkeypatch):
    monkeypatch.setattr(target.gaudi_envs, 'VLLM_HPU_DSV41_INDEX_WIDE_REINDEX', True)
    monkeypatch.setattr(target.gaudi_envs, 'VLLM_HPU_DSV41_CANDIDATE_COORDINATES', False)
    query = torch.zeros((1, 32, 128), dtype=torch.bfloat16)
    weights = torch.ones((1, 32), dtype=torch.bfloat16)
    keys = torch.zeros((262144, 128), dtype=torch.bfloat16)
    positions = torch.tensor([16383], dtype=torch.int32)
    pool = torch.arange(2048, dtype=torch.int32).reshape(1, -1)
    pool[0, 4] = -1
    calls = []

    def score(q, w, k, p, rows, ratio, heads, **kwargs):
        calls.append(rows.clone())
        return torch.empty((1, rows.shape[-1]))

    monkeypatch.setattr(mirror, 'mirror_index_tile', score)
    out = target._bounded_mme_scores(query, weights, None, None, positions, pool, 2, 8, 16384, True, keys)
    assert out.shape == (1, 16384) and len(calls) == 1
    assert torch.equal(calls[0][0, :8], torch.arange(8, dtype=torch.int32))
    assert (calls[0][0, 32:40] == -1).all()
    monkeypatch.setattr(target.gaudi_envs, 'VLLM_HPU_DSV41_INDEX_WIDE_REINDEX', False)
    calls.clear()
    target._bounded_mme_scores(query, weights, None, None, positions, pool, 2, 8, 16384, True, keys)
    assert len(calls) == 8 and all(t.shape == (1, 2048) for t in calls)
