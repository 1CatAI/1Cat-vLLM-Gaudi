# SPDX-License-Identifier: Apache-2.0
"""Preserve request/tile ownership, cutoff ties and candidate publication."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_decode_selection import batched_decode_selection, can_batch_full_r1
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("ratio,columns,publish,reindex,visible", [
    (2, 16384, False, False, 10240),
    (1, 32768, True, False, 20480),
    (1, 16384, False, True, None),
])
@pytest.mark.parametrize("distribution", ["bf16", "ties", "invalid"])
def test_identical_streamed_selections(tokens, ratio, columns, publish, reindex, visible, distribution):
    generator = torch.Generator().manual_seed(51810)
    positions = torch.arange(16384, 16384 + tokens, dtype=torch.int32)
    if reindex:
        blocks = torch.randperm(2048, generator=generator, dtype=torch.int32)
        rows = (blocks[:, None] * 8 + torch.arange(8, dtype=torch.int32)).reshape(1, -1).expand(tokens, -1)
    else:
        rows = torch.arange(columns, dtype=torch.int32)
    scores = torch.randn(tokens, columns, generator=generator).bfloat16().float()
    if distribution == "ties":
        scores = scores.round()
    elif distribution == "invalid":
        scores.fill_(-torch.inf)
    scores = scores.masked_fill(rows >= ((positions + 1) // ratio)[:, None], -torch.inf)
    tiles = list(scores.split(2048, -1))
    calls = iter(tiles)
    owner = SimpleNamespace(ratio=ratio, _merge_topk=PagedCSA2Attention._merge_topk,
                            _scores=lambda *unused: next(calls))
    expected = PagedCSA2Attention._stream_topk(owner, positions, rows,
                                              torch.empty(tokens, 32, 128), None,
                                              collect_blocks=publish, visible_rows=visible)
    actual = batched_decode_selection(tiles, rows, positions, ratio, collect_blocks=publish,
                                      selected_tiles=None if visible is None else visible // 2048)
    for got, want in zip(actual, expected, strict=True):
        if want is None:
            assert got is None
        else:
            torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_reject_incomplete_tile():
    with pytest.raises(ValueError, match="geometry"):
        batched_decode_selection([torch.zeros(1, 2047)], torch.arange(2047), torch.tensor([2046]), 1)


@pytest.mark.parametrize("field,value", [
    ("tensor_parallel_size", 2), ("ratio", 2), ("layer", 24),
    ("search_length", 65536), ("decode_batched_selection", False),
    ("tokens", 2), ("tokens", 6), ("device", "cpu"), ("prefix", None),
    ("native_scores", True), ("active_columns", 0), ("active_columns", 20096),
    ("columns", 16384), ("rows_ndim", 2),
])
def test_default_dispatch_is_limited_to_full_r1_c1(field, value):
    settings = dict(tensor_parallel_size=4, ratio=1, layer=20, candidate_source=20,
                    search_length=32768, decode_batched_selection=True,
                    tokens=1, device="hpu", prefix=object(), native_scores=False,
                    active_columns=20480, columns=32768, rows_ndim=1)
    def eligible():
        owner = SimpleNamespace(**settings)
        query = SimpleNamespace(device=SimpleNamespace(type=settings["device"]), shape=(settings["tokens"], 32, 128))
        rows = SimpleNamespace(ndim=settings["rows_ndim"], shape=(settings["columns"],))
        return can_batch_full_r1(owner, query, rows, settings["prefix"], settings["native_scores"],
                                 settings["active_columns"])
    assert eligible()
    settings[field] = value
    assert not eligible()
