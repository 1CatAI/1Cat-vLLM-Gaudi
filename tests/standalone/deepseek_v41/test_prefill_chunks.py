# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm_gaudi.v1.worker.deepseek_v41_runner import (
    PREFILL_BLOCK_TOKENS,
    decode_search_warmups,
    target_chunks,
    target_search_length,
)
from vllm_gaudi.ops.deepseek_v41_math import NATIVE_KV_CODEC_TOKENS


def test_decode_warmup_covers_all_context_buckets_without_scanning_tokens():
    for maximum in (128, 512, 513, 8192, 10000, 1 << 20):
        warmups = list(decode_search_warmups(maximum))
        assert warmups[0] == (0, min(512, maximum))
        assert warmups[-1][1] == maximum
        assert len(warmups) <= 12
        for index, (position, search) in enumerate(warmups):
            assert target_search_length(position, 1, maximum) == search
            if index:
                assert position == warmups[index - 1][1]
                assert search > position


def test_native_readiness_requires_every_bucket_and_clears_warmup_state(monkeypatch):
    from vllm_gaudi.v1.worker import deepseek_v41_runner as module

    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "0")
    monkeypatch.setenv("VLLM_HPU_DSV41_VERIFY_TIMING", "0")
    calls, validated = [], []

    class State:

        def clear(self):
            calls.append("clear")

    monkeypatch.setattr(module, "PagedStageState", State)
    runner = object.__new__(module.V41ModelRunner)
    runner.use_dspark = False
    runner.state = State()
    runner.graphed_buckets = set()
    runner.active_request = "warmup"
    owner = SimpleNamespace(require_ready=lambda count, search=512: validated.append((count, search)))
    runner.model = SimpleNamespace(native=True, pp_rank=0, program=SimpleNamespace(length=10000, replay_owner=owner))
    runner.pp = SimpleNamespace(group=SimpleNamespace(barrier=lambda: calls.append("barrier")))
    runner._dummy_run = lambda count, native, start_position=0: calls.append((count, native, start_position))

    runner.warmup_model()

    assert validated == [(1, length) for length in (512, 1024, 2048, 4096, 8192, 10000)]
    for position, _ in decode_search_warmups(10000):
        assert calls.count((1, True, position)) == 4
    assert calls[-2:] == ["barrier", "clear"]
    assert runner.active_request is None


def test_scheduler_transaction_is_one_c8192_prefill_block():
    tokens = list(range(8192))
    chunks = list(target_chunks(tokens))
    assert PREFILL_BLOCK_TOKENS == 8192
    assert len(chunks) == 1
    assert all(len(chunk) == PREFILL_BLOCK_TOKENS for _, chunk in chunks)
    assert [token for _, chunk in chunks for token in chunk] == tokens


def test_runtime_indexer_warms_one_metadata_driven_decode_graph(monkeypatch):
    from vllm_gaudi.v1.worker import deepseek_v41_runner as module

    monkeypatch.setenv("VLLM_HPU_DSV41_VERIFY_TIMING", "0")
    calls, validated = [], []

    class State:

        def clear(self):
            calls.append("clear")

    monkeypatch.setattr(module, "PagedStageState", State)
    runner = object.__new__(module.V41ModelRunner)
    runner.use_dspark = False
    runner.state, runner.graphed_buckets, runner.active_request = State(), set(), "warmup"
    owner = SimpleNamespace(require_ready=lambda count, search: validated.append((count, search)))
    runner.model = SimpleNamespace(native=True, pp_rank=0,
                                   program=SimpleNamespace(length=1 << 20, runtime_indexer=True, replay_owner=owner))
    runner.pp = SimpleNamespace(group=SimpleNamespace(barrier=lambda: calls.append("barrier")))
    runner._dummy_run = lambda count, native, start_position=0: calls.append((count, native, start_position))

    runner.warmup_model()

    assert list(decode_search_warmups(1 << 20, runtime_indexer=True)) == [(0, 1 << 20)]
    assert validated == [(1, 1 << 20)]
    assert calls == [(1, True, 0)] * 4 + ["barrier", "clear"]
    assert runner.active_request is None


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
