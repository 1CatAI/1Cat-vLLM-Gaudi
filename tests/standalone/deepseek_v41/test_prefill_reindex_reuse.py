# SPDX-License-Identifier: Apache-2.0
"""Candidate-slot semantics and invocation-local paged key ownership."""

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import (
    decode_shared_index_keys,
    decoded_reindex,
    index_scores,
)


def reference(query, weights, packed, table, positions, blocks, ratio):
    rows = blocks[..., None] * 8 + torch.arange(8, dtype=torch.int32)
    rows = torch.where(blocks[..., None] >= 0, rows, -1).flatten(1)
    best_values = best_rows = None
    for start in range(0, rows.shape[-1], 2048):
        current = rows[:, start:start + 2048]
        scores = index_scores(query, weights, packed, table, positions, current, ratio, 16, False, False)
        values, offsets = scores.topk(min(512, scores.shape[-1]), dim=-1, sorted=False)
        selected = current.gather(1, offsets)
        if best_values is not None:
            values = torch.cat((best_values, values), -1)
            selected = torch.cat((best_rows, selected), -1)
            values, offsets = values.topk(min(512, values.shape[-1]), dim=-1, sorted=False)
            selected = selected.gather(1, offsets)
        best_values, best_rows = values, selected
    limit = packed.shape[0]
    valid = (best_rows >= 0) & (best_rows < ((positions + 1) // ratio)[:, None])
    selected = torch.where(valid, best_rows, limit).sort(-1).values
    return torch.where(selected < limit, selected, -1).int()


@pytest.mark.parametrize("ratio,capacity", [(1, 2048), (2, 4096), (1, 32768)])
@pytest.mark.parametrize("tied", [False, True])
def test_candidate_slots_causality_and_ties(ratio, capacity, tied):
    torch.manual_seed(793)
    tokens = 3
    packed = torch.randint(0, 256, (capacity, 68), dtype=torch.uint8)
    packed[:, 64:] = 125
    table = torch.randperm(capacity // (128 // ratio)).int()
    query = torch.randint(-2, 3, (tokens, 32, 128)).bfloat16()
    weights = (torch.zeros(tokens, 32) if tied else torch.randn(tokens, 32)).bfloat16()
    positions = torch.tensor([0, 713 * ratio, capacity * ratio - 1], dtype=torch.int32)
    blocks = torch.stack([torch.randperm(capacity // 8)[:2048] for _ in range(tokens)]).int()
    blocks[:, 3:9] = -1
    # Keep duplicate slots: deduplication would change the top-k contract.
    blocks[:, 11] = blocks[:, 12]
    keys = decode_shared_index_keys(packed, table, ratio, capacity)
    expected = reference(query, weights, packed, table, positions, blocks, ratio)
    actual = decoded_reindex(query, weights, keys, positions, blocks, ratio, 16)
    assert torch.equal(actual, expected)


def test_shared_keys_refresh_after_source_write_and_page_rebinding():
    packed = torch.zeros(4224, 68, dtype=torch.uint8)
    packed[:, 64:] = 127
    table = torch.arange(33, dtype=torch.int32)
    before = decode_shared_index_keys(packed, table, 1, 4097)
    packed[128:256, :64] = 0x77
    table[0], table[1] = table[1].clone(), table[0].clone()
    after = decode_shared_index_keys(packed, table, 1, 4097)
    assert (before == 0).all() and (after[:128] == 6).all()
    rows = torch.arange(4097)
    physical = table[rows // 128] * 128 + rows % 128
    assert torch.equal(after, unpack_fp4(packed[physical], 128, 32))


@pytest.mark.parametrize("rows,ratio", [(0, 1), (32769, 1), (512, 4)])
def test_workspace_rejects_unbounded_or_unsupported_requests(rows, ratio):
    with pytest.raises(ValueError):
        decode_shared_index_keys(torch.empty(128, 68, dtype=torch.uint8), torch.zeros(1).int(), ratio, rows)
