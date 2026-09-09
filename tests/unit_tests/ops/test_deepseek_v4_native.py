# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v4_native import NativeDecodeMetadata, canonical_storage, embedding_partial


def test_metadata_reuse_requires_a_new_completed_consumer_generation():
    old = object()
    entry = {"completion": [old, None], "generations": [4, 0]}
    metadata = NativeDecodeMetadata({}, entry, 0)
    with pytest.raises(RuntimeError, match="no new decoder completion"):
        metadata.consumed()
    ticket = object()
    metadata.native_completion = ticket
    metadata.consumed()
    assert entry["completion"] == [ticket, None]
    assert entry["generations"] == [5, 0]
    with pytest.raises(RuntimeError, match="no new decoder completion"):
        metadata.consumed()


def test_mutated_cache_aliases_have_one_canonical_binding_without_copying():
    pool = torch.arange(128, dtype=torch.uint8)
    alias = torch.empty(0, dtype=torch.uint8).set_(pool.untyped_storage(), 0, (128,), (1,))
    registry = {}
    first = canonical_storage(pool.view(-1), registry)
    second = canonical_storage(alias.view(-1), registry)
    assert first is second and first._base is None
    assert first.untyped_storage()._cdata == pool.untyped_storage()._cdata
    first[32:64].fill_(7)
    assert torch.equal(alias[32:64], torch.full((32,), 7, dtype=torch.uint8))
    assert torch.equal(pool[:32], torch.arange(32, dtype=torch.uint8))


def test_partial_cache_alias_cannot_be_used_as_a_whole_pool():
    pool = torch.zeros(128, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="complete contiguous"):
        canonical_storage(pool[1:], {})


def test_embedding_preserves_shard_padding_added_vocabulary_and_mask():
    weight = torch.arange(12 * 8).reshape(12, 8).bfloat16()
    tokens = torch.tensor([-1, 0, 7, 8, 10, 13, 14, 19, 20, 21, 22, 31])
    # Original rows 8..13 occupy local rows 0..5, two padding rows follow,
    # and added vocabulary rows 20..21 occupy local rows 8..9.
    output = embedding_partial(tokens, weight, 8, 14, 2, 20, 22)
    expected = torch.zeros(len(tokens), 8, dtype=torch.bfloat16)
    for row, token in enumerate(tokens.tolist()):
        if 8 <= token < 14:
            expected[row] = weight[token - 8]
        elif 20 <= token < 22:
            expected[row] = weight[token - 20 + 8]
    assert torch.equal(output, expected)
