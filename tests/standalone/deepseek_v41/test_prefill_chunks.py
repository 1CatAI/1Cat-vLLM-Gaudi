# SPDX-License-Identifier: Apache-2.0
from vllm_gaudi.v1.worker.deepseek_v41_runner import (
    PREFILL_BLOCK_TOKENS,
    target_chunks,
    target_search_length,
)
from vllm_gaudi.ops.deepseek_v41_math import NATIVE_KV_CODEC_TOKENS


def test_scheduler_transaction_is_one_c8192_prefill_block():
    tokens = list(range(8192))
    chunks = list(target_chunks(tokens))
    assert PREFILL_BLOCK_TOKENS == 8192
    assert len(chunks) == 1
    assert all(len(chunk) == PREFILL_BLOCK_TOKENS for _, chunk in chunks)
    assert [token for _, chunk in chunks for token in chunk] == tokens


def test_native_kv_codec_covers_the_complete_prefill_transaction():
    assert NATIVE_KV_CODEC_TOKENS == PREFILL_BLOCK_TOKENS == 8192


def test_scheduler_transaction_uses_one_search_bucket_for_all_internal_tiles():
    assert target_search_length(0, 8192, 1 << 20) == 8192
    assert target_search_length(8192, 823, 1 << 20) == 16384
    assert target_search_length((1 << 20) - 64, 64, 1 << 20) == 1 << 20


def test_prefill_tail_is_exact_and_never_splits_into_dspark_c6():
    for count in (1, 3, 6, 7, 127, 129, 8191):
        tokens = list(range(count))
        chunks = list(target_chunks(tokens))
        assert [token for _, chunk in chunks for token in chunk] == tokens
        assert all(1 <= len(chunk) <= PREFILL_BLOCK_TOKENS for _, chunk in chunks)
        assert all(len(chunk) == PREFILL_BLOCK_TOKENS for _, chunk in chunks[:-1])
        if count != 6:
            assert not (len(chunks) > 1 and any(len(chunk) == 6 for _, chunk in chunks))
        assert all(offset == sum(len(previous) for _, previous in chunks[:index])
                   for index, (offset, _) in enumerate(chunks))
