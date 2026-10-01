# SPDX-License-Identifier: Apache-2.0
"""Shared coordinates must follow each call's positions and retain ring writes."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_decode_metadata import prepare_decode_metadata


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_changing_positions_and_ring_wrap_match_original_writes(dtype, tokens):
    positions = torch.zeros(tokens, dtype=dtype)
    cache = torch.zeros(256, 2)
    parent = cache.clone()
    for start in (-2, 0, 127, 255, 256, 32767, 1048575):
        positions.copy_(torch.arange(start, start + tokens, dtype=dtype))
        rows, lengths = prepare_decode_metadata(positions)
        values = torch.arange(tokens * 2).reshape(tokens, 2).float() + start
        parent.index_copy_(0, positions.remainder(256).long(), values)
        cache.index_copy_(0, rows, values)
        assert torch.equal(cache, parent)
        assert rows.dtype == torch.int64 and lengths.dtype == torch.int32
        assert lengths.shape == (tokens,) and torch.equal(lengths, torch.full((tokens,), 640, dtype=torch.int32))


def test_compiled_metadata_reads_new_tensor_contents_each_call(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    compiled = torch.compile(prepare_decode_metadata, backend='eager', fullgraph=True, dynamic=False)
    positions = torch.tensor([255, 256], dtype=torch.int32)
    for offset in (0, 2, 32768, 1000000):
        positions.add_(offset)
        rows, lengths = compiled(positions)
        assert torch.equal(rows, positions.remainder(256).long())
        assert torch.equal(lengths, torch.full((2,), 640, dtype=torch.int32))
