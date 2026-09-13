# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.gqa_compact import compact_gqa_matmul


@pytest.mark.parametrize("blocks,tokens", [(1, 17), (4, 128), (4, 896)])
def test_shared_kv_and_output_rows_are_exact(blocks, tokens):
    generator = torch.Generator().manual_seed(72)
    query = torch.randint(-8, 9, (blocks, 2, 6, 1, 256), generator=generator).to(torch.bfloat16) / 16
    storage = torch.randint(-8, 9, (blocks, tokens, 2, 256), generator=generator).to(torch.bfloat16) / 16
    value = storage.transpose(1, 2).unsqueeze(2)
    key = value.transpose(-1, -2)
    expected = query @ key
    actual = compact_gqa_matmul(query, key)
    assert torch.equal(actual, expected)
    assert torch.equal(compact_gqa_matmul(actual, value), expected @ value)
    assert key.untyped_storage()._cdata == storage.untyped_storage()._cdata


def test_invalid_query_rows_are_rejected_before_matmul():
    left = torch.zeros((1, 2, 6, 2, 8), dtype=torch.bfloat16)
    right = torch.zeros((1, 2, 1, 8, 17), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="one query row"):
        compact_gqa_matmul(left, right, lambda *_: pytest.fail("invalid shape reached matmul"))
