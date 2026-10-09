# SPDX-License-Identifier: Apache-2.0
"""Shared Reindex scoring preserves page remapping and TP BF16 sums."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@pytest.mark.parametrize("tp_size", (2, 4))
@pytest.mark.parametrize("ratio", (1, 2))
@pytest.mark.parametrize("tokens", (2, 6))
def test_native_keys_keep_per_query_rows_causality_and_page_changes(monkeypatch, tp_size, ratio, tokens):
    from vllm_gaudi.ops import deepseek_v41_decode_index as decode_index
    torch.manual_seed(4186)
    width, columns = 128 // ratio, 193
    pages = torch.tensor([2, 1, 3, 0], dtype=torch.int32)
    packed = pack_fp4(torch.randn(4 * width, 128).bfloat16(), 32)
    rows = torch.randint(0, 3 * width, (tokens, columns), dtype=torch.int32)
    rows[:, ::11] = -1
    rows[:, 1] = rows[:, 2]
    positions = torch.arange(tokens, dtype=torch.int32) + ratio * width
    query = torch.randn(tokens, 32, 128).bfloat16()
    weights = (torch.randn(tokens, 32) * .02).bfloat16()

    def physical(ids, ratio):
        page_width = 128 // ratio
        return pages[(ids // page_width).long()] * page_width + ids % page_width

    attention = SimpleNamespace(shared=SimpleNamespace(physical_rows=physical, block_table=pages),
                                cache=SimpleNamespace(index=packed),
                                ratio=ratio,
                                index_heads=32 // tp_size,
                                tensor_parallel_size=tp_size)
    monkeypatch.setattr(decode_index, "supports_per_query_index_keys", lambda q, r: False)
    expected = PagedCSA2Attention._scores(attention, positions, rows, query, weights)
    calls = []

    def native_keys(cache, table, ids, ratio):
        calls.append((cache, table, ids, ratio))
        keys = unpack_fp4(cache[physical(ids.clamp_min(0), ratio).long()], 128, 32)
        return keys.masked_fill((ids < 0).unsqueeze(-1), 0)

    monkeypatch.setattr(decode_index, "supports_per_query_index_keys", lambda q, r: True)
    monkeypatch.setattr(decode_index, "per_query_index_keys", native_keys)
    actual = PagedCSA2Attention._scores(attention, positions, rows, query, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 1 and calls[0][2] is rows and calls[0][3] == ratio
    # A scheduler page rebind must be visible on the next invocation.
    pages[0], pages[1] = pages[1].clone(), pages[0].clone()
    rebound = PagedCSA2Attention._scores(attention, positions, rows, query, weights)
    monkeypatch.setattr(decode_index, "supports_per_query_index_keys", lambda q, r: False)
    rebound_expected = PagedCSA2Attention._scores(attention, positions, rows, query, weights)
    torch.testing.assert_close(rebound, rebound_expected, rtol=0, atol=0)
    assert not torch.equal(actual, rebound)


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("abi", ["query_rows", "shared_only", "wrong_dtype", "wrong_shape"])
def test_row_plane_abi_probe_keeps_old_libraries_on_tensor_path(monkeypatch, heads, abi):
    from vllm_gaudi.ops.deepseek_v41_decode_index import supports_query_row_reduction

    supports_query_row_reduction.cache_clear()

    def operation(dots, weights, positions, rows, ratio, local_heads):
        assert all(value.device.type == "meta" for value in (dots, weights, positions, rows))
        assert rows.shape == (2, 128) and local_heads == heads and ratio == 1
        if abi == "shared_only":
            raise RuntimeError("Legacy row-vector ABI")
        return torch.empty((2, 127 if abi == "wrong_shape" else 128), device="meta",
                           dtype=torch.bfloat16 if abi == "wrong_dtype" else torch.float32)

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_prefill_index_reduce_gaudi2",
                        operation, raising=False)
    try:
        assert supports_query_row_reduction(heads) == (abi == "query_rows")
        assert not supports_query_row_reduction(4)
    finally:
        supports_query_row_reduction.cache_clear()


@pytest.mark.parametrize("distribution", ["random", "ties", "invalid"])
def test_wide_scores_preserve_streamed_merge_order(distribution):
    torch.manual_seed(4190)
    tokens, columns = 6, 16384
    rows = torch.randperm(columns, dtype=torch.int32).expand(tokens, -1)
    positions = torch.arange(16384, 16390, dtype=torch.int32)
    bank = torch.randn(tokens, columns).bfloat16().float()
    if distribution == "ties":
        bank = bank.round()
    elif distribution == "invalid":
        bank.fill_(-torch.inf)
    calls = []

    def scores(pos, source, query, weights):
        calls.append(source.shape[-1])
        return bank.gather(1, source.long())

    query = SimpleNamespace(shape=(tokens, 32, 128), device=SimpleNamespace(type="hpu"))
    owner = SimpleNamespace(c6_index_reduce=True, c6_index_wide_scores=False,
                            _uses_index_mirror=lambda q: True, _scores=scores,
                            _merge_topk=PagedCSA2Attention._merge_topk)
    expected = PagedCSA2Attention._stream_topk(owner, positions, rows, query, None)
    assert calls == [2048] * 8
    calls.clear()
    owner.c6_index_wide_scores = True
    observed = PagedCSA2Attention._stream_topk(owner, positions, rows, query, None)
    assert calls == [16384]
    assert torch.equal(expected[0], observed[0])
    assert expected[1:] == observed[1:] == (None, None)
