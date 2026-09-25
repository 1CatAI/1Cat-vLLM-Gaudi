# SPDX-License-Identifier: Apache-2.0
"""Validate persistent Full/Reindex/Reuse state at the native op boundary."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_indexer as native
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@pytest.mark.parametrize("layer,ratio", [(2, 2), (20, 1), (24, 1)])
def test_runtime_selector_publishes_current_state_without_rebinding(monkeypatch, layer, ratio):
    attention = object.__new__(PagedCSA2Attention)
    torch.nn.Module.__init__(attention)
    attention.layer, attention.ratio, attention.length = layer, ratio, 1 << 20
    attention.search_length = attention.length
    attention.runtime_indexer, attention.owns_index, attention.candidate_source = True, True, 20
    pool = None if layer < 20 else torch.full((8192, 2048), -1, dtype=torch.int32)
    attention.shared = SimpleNamespace(candidate_pool=pool,
                                       block_table=torch.ones(8192, dtype=torch.int32),
                                       index_candidates_unused=torch.full((1, 2048), -1, dtype=torch.int32))
    attention.cache = SimpleNamespace(index=torch.empty(128, 68, dtype=torch.uint8))
    attention.selection = SimpleNamespace(indices=torch.full((8192, 512), -1, dtype=torch.int32))
    output_address = attention.selection.indices.data_ptr()
    query, weights = torch.zeros(1, 32, 128).bfloat16(), torch.zeros(1, 32).bfloat16()
    calls = []

    def select(q, w, cache, pages, positions, candidates, **options):
        assert q is query and w is weights
        assert cache is attention.cache.index and pages is attention.shared.block_table
        assert candidates.shape == (1, 2048)
        assert candidates.data_ptr() == (attention.shared.index_candidates_unused.data_ptr()
                                         if pool is None else pool.data_ptr())
        assert options == dict(ratio=ratio,
                               capacity=(1 << 20) // ratio,
                               reindex=layer > 20,
                               publish_candidates=layer == 20,
                               decoded_hot=None,
                               ordered_candidates=True)
        calls.append(positions.item())
        selected = torch.full((1, 512), positions.item() // ratio, dtype=torch.int32)
        blocks = torch.full((1, 2048), positions.item() // 8, dtype=torch.int32) if layer == 20 else None
        return selected, blocks

    monkeypatch.setattr(native, "runtime_index_select", select)
    for position in (511, 512, 1024, 0):
        output = attention._select(None, None, torch.tensor([position]), prepared=(query, weights))
        assert output.data_ptr() == output_address
        assert (output == position // ratio).all()
        if layer == 20:
            assert (pool[0] == position // 8).all()
        elif pool is not None:
            assert (pool == -1).all()
    assert calls == [511, 512, 1024, 0]

    attention.owns_index = False
    reused = attention._select(None, None, torch.tensor([1]))
    assert reused.data_ptr() == output_address
    assert (reused == 0).all()
    assert len(calls) == 4


@pytest.mark.parametrize("batch,paged", [(2, False), (1, True), (8, True)])
def test_single_request_hot_mirror_cannot_be_shared_across_request_pages(batch, paged):
    query = torch.zeros(batch, 32, 128, dtype=torch.bfloat16)
    weights = torch.zeros(batch, 32, dtype=torch.bfloat16)
    cache = torch.empty(128, 68, dtype=torch.uint8)
    pages = torch.ones((batch, 8) if paged else (8, ), dtype=torch.int32)
    positions = torch.full((batch, ), 2047, dtype=torch.int32)
    candidates = torch.full((batch, 2048), -1, dtype=torch.int32)
    hot = torch.zeros(native.INDEX_MME_HOT_TOKENS, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="belongs to one request"):
        native.runtime_index_select(query,
                                    weights,
                                    cache,
                                    pages,
                                    positions,
                                    candidates,
                                    ratio=1,
                                    capacity=1 << 20,
                                    decoded_hot=hot)


def test_external_reindex_pool_preserves_logical_output_order(monkeypatch):
    # A restored/external pool can be in selection order rather than logical
    # order. Its padding must stay last after mapping to original row IDs.
    raw = torch.tensor([[64, 7, -1, 31]], dtype=torch.int32)
    ops = SimpleNamespace(custom_deepseek_v41_index_scores_gaudi2=lambda *args: (torch.empty(1), torch.empty(1)),
                          custom_deepseek_v41_index_threshold_gaudi2=lambda *args: torch.empty(1),
                          custom_deepseek_v41_index_emit_gaudi2=lambda *args: raw)
    monkeypatch.setattr(native.torch.ops, "custom_op", ops)
    dummy = torch.empty(1)
    actual, published = native.runtime_index_select(dummy,
                                                    dummy,
                                                    dummy,
                                                    dummy,
                                                    dummy,
                                                    dummy,
                                                    ratio=1,
                                                    capacity=1 << 20,
                                                    reindex=True)
    assert actual.tolist() == [[7, 31, 64, -1]] and published is None
    assert raw.tolist() == [[64, 7, -1, 31]]
