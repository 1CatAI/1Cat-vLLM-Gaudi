# SPDX-License-Identifier: Apache-2.0
"""Reindex threshold reuse preserves the causal score plane and row mapping."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('seed', range(3))
def test_common_selector_uses_all_query_planes_and_maps_long_context_rows(seed, monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'vllm_gaudi/ops/deepseek_v41_decode_selection.py'
    spec = importlib.util.spec_from_file_location('reindex_threshold_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = torch.Generator().manual_seed(seed)
    candidates = torch.stack([torch.randperm(4096, generator=rng)[:2048] for _ in range(6)]).int()
    rows = (candidates.unsqueeze(-1) * 8 + torch.arange(8, dtype=torch.int32)).flatten(1)
    positions = torch.arange(6, dtype=torch.int32) + 18000
    scores = torch.randn((6, 16384), generator=rng).bfloat16().float()
    scores.masked_fill_(rows > positions.reshape(-1, 1), -torch.inf)
    expected_offsets = scores.argsort(dim=-1, descending=True, stable=True)[:, :512].sort(dim=-1).values
    calls = []

    def accepted_selector(actual_scores, extent, pool, ratio, *, reindex=False, blocks=False):
        calls.append(actual_scores.clone())
        assert ratio == 1 and reindex and not blocks
        # Visibility belongs to the existing scoring producer; this extent
        # scans the fixed score columns, independently of mapped logical IDs.
        assert torch.equal(extent, torch.full_like(positions, 16383))
        assert torch.equal(pool, candidates)
        offsets = actual_scores.argsort(dim=-1, descending=True, stable=True)[:, :512].sort(dim=-1).values
        return pool.gather(1, offsets // 8) * 8 + offsets.remainder(8)

    monkeypatch.setitem(sys.modules, 'vllm_gaudi.ops.deepseek_v41_indexer',
                        SimpleNamespace(ordered_index_ids=accepted_selector))
    actual, block_scores, block_ids = module.threshold_decode_selection(
        list(scores.split(2048, dim=-1)), rows, positions, 1)
    assert len(calls) == 1 and torch.equal(calls[0], scores)
    assert torch.equal(actual, rows.gather(1, expected_offsets))
    assert bool((actual <= positions.reshape(-1, 1)).all())
    assert block_scores is None and block_ids is None
