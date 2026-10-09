# SPDX-License-Identifier: Apache-2.0
"""Draft MLA ABI: ring capacity, tentative visibility and masked alignment."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_paged_attention as attention


@pytest.mark.parametrize("mode", ["mme", "codec"])
@pytest.mark.parametrize("start", [0, 255, 16384])
def test_packed_draft_keeps_tentative_rows_out_of_ring(monkeypatch, start, mode):
    ring = torch.arange(256 * 528, dtype=torch.int32).remainder(256).to(torch.uint8).reshape(256, 528)
    saved = ring.clone()
    tentative = torch.full((5, 528), 37, dtype=torch.uint8)
    positions = torch.arange(start, start + 5, dtype=torch.int32)
    seen = []

    def consume(query, packed, main, rows, indices, sink, scale, lengths):
        # A paged consumer decodes row_ids into a *new* cache. An index
        # must address this cache, including all five tentative rows.
        assert torch.equal(rows.flatten(), torch.arange(261, dtype=torch.int32))
        assert torch.equal(packed[:256], saved)
        assert torch.equal(packed[256:], tentative)
        assert indices.shape == (5, 192)
        assert torch.equal(indices[:, 128:133], torch.arange(256, 261).expand(5, -1))
        assert (indices[:, 133:] == -1).all()
        assert ((indices < 0) | (indices < rows.numel())).all()
        expected = torch.arange(start - 128, start, dtype=torch.int32)
        expected = torch.where(expected >= 0, expected.remainder(256), -1)
        assert torch.equal(indices[:, :128], expected.expand(5, -1))
        assert torch.equal(lengths, torch.full((5,), 133, dtype=torch.int32))
        seen.append(True)
        return query

    def decode(packed, main, rows, vector):
        assert torch.equal(rows.flatten(), torch.arange(261, dtype=torch.int32))
        assert torch.equal(packed[:256], saved)
        assert torch.equal(packed[256:], tentative)
        assert not vector
        return torch.zeros((261, 512), dtype=torch.bfloat16), rows

    def sparse(query, cache, indices, p):
        assert cache.shape == (261, 512)
        assert indices.shape == (5, 133)
        assert torch.equal(indices[:, 128:], torch.arange(256, 261).expand(5, -1))
        seen.append(True)
        return query

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_selected_kv_bf16_gaudi2", decode, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_paged_mla_mme_gaudi2", consume, raising=False)
    monkeypatch.setattr(attention, "rms_norm", lambda x, weight, eps: x)
    monkeypatch.setattr(attention, "pack_swa", lambda x: tentative)
    owner = SimpleNamespace(
        weights=SimpleNamespace(wq_a=0, q_norm=SimpleNamespace(weight=None), wq_b=1,
                                wkv=2, kv_norm=SimpleNamespace(weight=None), attn_sink=torch.zeros(1)),
        eps=1e-6, heads=1, linear=lambda x, weight: x, _rope=lambda x, p: x,
        swa=ring, window=128, window_offsets=torch.arange(128, dtype=torch.int32),
        draft_packed_mla=mode == "mme", draft_kv_decode=mode == "codec", _output=sparse,
        swa_only_main=torch.zeros((1, 288), dtype=torch.uint8),
        draft_mla_rows=torch.arange(261, dtype=torch.int32).reshape(1, -1),
        draft_mla_lengths=torch.full((5,), 133, dtype=torch.int32), scale=torch.ones(1),
        _finish_output=lambda x, p: x,
    )
    output = attention.PagedCSA2Attention.draft(owner, torch.ones((5, 512), dtype=torch.bfloat16), positions)
    assert output.shape == (5, 1, 512)
    assert seen == [True]
    assert torch.equal(ring, saved)
