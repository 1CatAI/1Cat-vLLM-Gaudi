# SPDX-License-Identifier: Apache-2.0
from vllm_gaudi.v1.worker.deepseek_v41_runner import PREFILL_BLOCK_TOKENS, target_chunks


def test_scheduler_transaction_uses_c128_prefill_blocks():
    tokens = list(range(8192))
    chunks = list(target_chunks(tokens))
    assert len(chunks) == 64
    assert all(len(chunk) == PREFILL_BLOCK_TOKENS for _, chunk in chunks)
    assert [token for _, chunk in chunks for token in chunk] == tokens


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
