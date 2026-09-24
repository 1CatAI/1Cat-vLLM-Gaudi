# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA mask/sink semantics, independent of Gaudi device availability."""
import pytest
import torch
import torch.nn.functional as F

from flashinfer_gaudi.mla import _cache_with_zero_row, _final_tile_inputs, _tile_inputs, sparse_mla_prefill
from flashinfer_gaudi._dispatch import BackendUnavailableError


@pytest.mark.parametrize("tokens,heads,columns", [(1, 1, 1), (3, 4, 7), (17, 3, 64), (19, 3, 128), (33, 3, 640)])
def test_selected_rows_sink_and_lengths(tokens, heads, columns):
    generator = torch.Generator().manual_seed(914)
    q = torch.randn(tokens, heads, 512, generator=generator).bfloat16()
    cache = torch.randn(29, 512, generator=generator).bfloat16()
    ids = torch.randint(-3, 34, (tokens, columns), generator=generator, dtype=torch.int32)
    lengths = torch.arange(tokens, dtype=torch.int32).mul(5).remainder(columns + 1)
    sink = torch.randn(heads, generator=generator).float()
    queries, kv, mask = _tile_inputs(q, cache, ids, sink, lengths)
    actual = F.scaled_dot_product_attention(queries.float(), kv.float(), kv.float(), mask).squeeze(1)
    expected = torch.zeros_like(actual)
    for token in range(tokens):
        selected = ids[token, :lengths[token]]
        selected = selected[(selected >= 0) & (selected < len(cache))].long()
        values = cache[selected].float()
        scores = q[token].float() @ values.T * 512**-.5
        probability = torch.cat((scores, sink[:, None]), -1).softmax(-1)[:, :-1]
        expected[token] = probability @ values
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert queries.shape == (tokens, 1, heads, 512)
    assert kv.dtype == torch.bfloat16
    assert torch.equal(kv[:, :, -1], torch.zeros_like(kv[:, :, -1]))


def test_zero_row_reused_across_tiles_without_mutating_cache():
    cache = torch.arange(5 * 512).reshape(5, 512).bfloat16()
    cache[0] = float("nan")
    original = cache.clone()
    padded = _cache_with_zero_row(cache)
    q = torch.ones(3, 2, 512, dtype=torch.bfloat16)
    ids = torch.tensor([[-1, 5, 1], [1, 2, 3], [4, -1, 2]], dtype=torch.int32)
    lengths = torch.tensor([3, 0, 1], dtype=torch.int32)
    for start, stop in ((0, 2), (2, 3)):
        _, kv, mask = _final_tile_inputs(q[start:stop], padded, ids[start:stop], torch.zeros(2), lengths[start:stop])
        assert kv.is_contiguous() and kv.isfinite().all()
        assert torch.equal(kv[:, :, -1], torch.zeros_like(kv[:, :, -1]))
        assert torch.equal(mask[..., -1], torch.zeros_like(mask[..., -1]))
    _, kv, _ = _final_tile_inputs(q, padded, ids, torch.zeros(2), lengths)
    assert torch.equal(kv[0, 0, :2], torch.zeros_like(kv[0, 0, :2]))
    assert torch.equal(kv[0, 0, 2], cache[1])
    assert torch.equal(kv[1], torch.zeros_like(kv[1]))
    assert torch.equal(kv[2, 0, 0], cache[4])
    assert torch.equal(kv[2, 0, 1:], torch.zeros_like(kv[2, 0, 1:]))
    torch.testing.assert_close(cache, original, rtol=0, atol=0, equal_nan=True)


def test_invalid_index_does_not_read_nan_into_pv():
    q = torch.ones(2, 3, 512, dtype=torch.bfloat16)
    cache = torch.ones(5, 512, dtype=torch.bfloat16)
    cache[0] = float("nan")
    cache[-1] = float("nan")
    ids = torch.tensor([[-1, 8], [-1, 8]], dtype=torch.int32)
    lengths = torch.tensor([2, 0], dtype=torch.int32)
    queries, kv, mask = _tile_inputs(q, cache, ids, torch.zeros(3), lengths)
    assert kv.isfinite().all()
    output = F.scaled_dot_product_attention(queries.float(), kv.float(), kv.float(), mask)
    assert torch.equal(output, torch.zeros_like(output))


def test_native_entry_rejects_cpu_instead_of_silent_reference():
    with pytest.raises(BackendUnavailableError, match="no reference fallback"):
        sparse_mla_prefill(torch.zeros(1, 2, 512, dtype=torch.bfloat16), torch.zeros(2, 512, dtype=torch.bfloat16),
                           torch.zeros(1, 64, dtype=torch.int32), torch.zeros(2), torch.ones(1, dtype=torch.int32))


def test_native_entry_rejects_shape_before_device_launch():
    with pytest.raises(ValueError, match="cache, indices"):
        sparse_mla_prefill(torch.zeros(2, 2, 512, dtype=torch.bfloat16), torch.zeros(2, 512, dtype=torch.bfloat16),
                           torch.zeros(1, 64, dtype=torch.int32), torch.zeros(2), torch.ones(2, dtype=torch.int32))
