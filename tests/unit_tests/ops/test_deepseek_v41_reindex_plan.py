# SPDX-License-Identifier: Apache-2.0
"""Explicit key-window bindings and transaction-owned replay destinations."""
from types import SimpleNamespace

import torch

from vllm_gaudi.ops import deepseek_v41_prefill_plan as plans
from vllm_gaudi.ops import deepseek_v41_prefill_reindex_plan as index
from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import _candidate_slot_topk


def test_replay_rebinds_keys_and_preserves_previous_transactions(monkeypatch):
    events = []
    active_roots = set()

    def scores(query, weights, keys, positions, rows, ratio):
        value = (query[:, 0, 0, None].float() + keys[None, :, 0].float() + rows[None]).remainder(19) - 9
        value = value * weights[:, :1].float()
        return value.masked_fill((rows[None] < 0) | (rows[None] >= (positions[:, None] + 1) // ratio), -torch.inf)

    def score_region(query, weights, keys, positions, rows, ratio):
        assert all(id(v) in active_roots for v in (query, weights, keys, positions))
        return index.source_scores(query, weights, keys, positions, rows, ratio)

    def select_region(positions, blocks, ratio, *parts):
        assert all(id(v) in active_roots for v in (positions, blocks))
        return index.select_scores(positions, blocks, ratio, *parts)

    def gather(common, blocks, positions, ratio):
        rows = blocks[..., None] * 8 + torch.arange(8, dtype=torch.int32)
        rows = torch.where(blocks[..., None] >= 0, rows, -1).flatten(1)
        value = common.gather(1, rows.clamp(0, common.shape[1] - 1).long())
        valid = (rows >= 0) & (rows < common.shape[1]) & (rows < (positions[:, None] + 1) // ratio)
        return value.masked_fill(~valid, -torch.inf), rows

    class Plan:

        def __init__(self, body, arguments, groups, require_prefix):
            assert groups is None and require_prefix
            self.body = body
            self.offsets = tuple(v.storage_offset() for v in arguments)
            self.plan = SimpleNamespace(invalidate=lambda: events.append("invalidate"))
            self.run(arguments)

        def run(self, arguments):
            active_roots.clear()
            active_roots.update(id(v) for v in arguments)
            self.body(*arguments)

        def replay(self, arguments, count):
            assert count == 1
            assert tuple(v.storage_offset() for v in arguments) == self.offsets
            self.run(arguments)
            events.append("replay")

    monkeypatch.setattr(plans, "PrefillExpertPlan", Plan)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: events.append("drain"))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: SimpleNamespace(hpu_stream=0))
    monkeypatch.setattr(torch.hpu, "default_stream", lambda: SimpleNamespace(hpu_stream=0))
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_prefill_index_scores_gaudi2", scores, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_candidate_gather_f32_gaudi2", gather, raising=False)
    monkeypatch.setattr(index, "compiled_scores", lambda signature: score_region)
    monkeypatch.setattr(index, "compiled_select", lambda signature: select_region)
    index._families.clear()
    torch.manual_seed(6711)
    retained = []
    shapes = ((257, 16392, 2), (257, 16392, 2), (129, 8200, 1), (257, 16392, 2), (17, 24, 1))
    for generation, (tokens, columns, ratio) in enumerate(shapes):
        query = torch.full((tokens, 32, 128), generation + 1, dtype=torch.bfloat16)
        weights = torch.full((tokens, 32), 0.25, dtype=torch.bfloat16)
        keys = torch.zeros(columns, 128, dtype=torch.bfloat16)
        keys[:, 0] = (torch.arange(columns) + generation).remainder(17).bfloat16()
        positions = torch.linspace(-1, columns * ratio - 1, tokens).int()
        blocks = torch.randint(-1, columns // 8, (tokens, 129), dtype=torch.int32)
        rows = torch.arange(columns, dtype=torch.int32)
        common = scores(query, weights, keys, positions, rows, ratio)
        expected = _candidate_slot_topk(common, positions, blocks, ratio)
        actual = index.prepared_reindex_selection(query, weights, keys, positions, blocks, ratio)
        assert torch.equal(actual, expected)
        for previous, snapshot in retained:
            assert torch.equal(previous, snapshot)
        retained.append((actual, actual.clone()))
        assert len(index._families) <= 2
        assert all(len(family) <= 2 for family in index._families.values())
    assert "replay" in events and "invalidate" in events
    for i, event in enumerate(events):
        if event == "invalidate":
            assert "drain" in events[:i]
    for family in index._families.values():
        index._close(family)
    index._families.clear()
