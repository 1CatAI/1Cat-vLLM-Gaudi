# SPDX-License-Identifier: Apache-2.0
"""C6 mirrors consume canonical packed writes and keep per-query score order."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_index_mirror import per_query_mirror_keys, refresh_index_mirror_rows
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@pytest.mark.parametrize("ratio", (1, 2))
@pytest.mark.parametrize("tp", (2, 4))
def test_c6_mirror_scores_keep_masks_and_original_row_order(monkeypatch, ratio, tp):
    from vllm_gaudi.ops import deepseek_v41_decode_index as decoder
    torch.manual_seed(163846)
    width = 128 // ratio
    pages = torch.tensor([2, 1, 3, 0], dtype=torch.int32)
    packed = pack_fp4(torch.randn(4 * width, 128).bfloat16(), 32)
    logical = torch.arange(4 * width)
    physical = pages[(logical // width).long()] * width + logical % width
    mirror = unpack_fp4(packed.index_select(0, physical.long()), 128, 32)
    rows = torch.randint(0, 3 * width, (6, 193), dtype=torch.int32)
    rows[:, ::11] = -1
    query = torch.randn(6, 32, 128).bfloat16()
    weights = torch.randn(6, 32).bfloat16() * .02
    positions = torch.arange(6, dtype=torch.int32) + width * ratio
    shared = SimpleNamespace(block_table=pages, index_mirror_valid=True, index_mirror_tokens=4 * width * ratio,
                             physical_rows=lambda ids, ratio: pages[(ids // width).long()] * width + ids % width)
    owner = SimpleNamespace(shared=shared, cache=SimpleNamespace(index=packed, index_mirror=mirror), ratio=ratio,
                            tensor_parallel_size=tp, index_heads=32 // tp, search_length=4 * width * ratio,
                            index_mirror_scores=False)
    owner._uses_index_mirror = MethodType(PagedCSA2Attention._uses_index_mirror, owner)
    monkeypatch.setattr(decoder, "supports_per_query_index_keys", lambda q, r: False)
    expected = PagedCSA2Attention._scores(owner, positions, rows, query, weights)
    owner.index_mirror_scores = True
    actual = PagedCSA2Attention._scores(owner, positions, rows, query, weights)
    assert torch.equal(actual, expected)
    assert torch.equal(per_query_mirror_keys(mirror, rows)[rows < 0], torch.zeros((rows < 0).sum(), 128).bfloat16())


@pytest.mark.parametrize("ratio", (1, 2))
def test_refresh_reads_after_canonical_write_with_duplicate_compressed_rows(monkeypatch, ratio):
    width = 128 // ratio
    pages = torch.tensor([2, 1, 3, 0], dtype=torch.int32)
    packed = pack_fp4(torch.randn(4 * width, 128).bfloat16(), 32)
    positions = torch.arange(6) + 128
    rows = (positions // ratio).int()
    mirror = torch.full((4 * width, 128), 77, dtype=torch.bfloat16)

    def canonical(cache, table, ids, r):
        physical = table[(ids // width).long()] * width + ids % width
        return unpack_fp4(cache[physical.long()], 128, 32)

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_index_keys_gaudi2", canonical, raising=False)
    expected = canonical(packed, pages, rows.reshape(1, -1), ratio).reshape(-1, 128)
    refresh_index_mirror_rows(packed, pages, rows, mirror, ratio)
    assert torch.equal(mirror.index_select(0, rows.long()), expected)
    untouched = torch.ones(mirror.shape[0], dtype=torch.bool)
    untouched[rows.long()] = False
    assert torch.all(mirror[untouched] == 77)
