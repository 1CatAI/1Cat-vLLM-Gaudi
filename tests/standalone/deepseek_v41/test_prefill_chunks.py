# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_pp import PackedC1Buffers
from vllm_gaudi.v1.worker.deepseek_v41_runner import (
    PREFILL_BLOCK_TOKENS,
    PREFILL_COMPUTE_BUCKETS,
    V41ModelRunner,
    decode_search_warmups,
    runtime_search_length,
    target_chunks,
    target_search_length,
)
from vllm_gaudi.ops.deepseek_v41_math import NATIVE_KV_CODEC_TOKENS


def test_scheduler_transaction_uses_finite_exact_compute_buckets():
    tokens = list(range(8192))
    chunks = list(target_chunks(tokens))
    assert PREFILL_BLOCK_TOKENS == 8192
    assert [len(chunk) for _, chunk in chunks] == [2048] * 4
    assert [token for _, chunk in chunks for token in chunk] == tokens


def test_normal_2k_chat_shape_reuses_c2048_and_c1_replay():
    chunks = list(target_chunks(list(range(2052))))
    assert [len(chunk) for _, chunk in chunks] == [2048, 1, 1, 1, 1]


@pytest.mark.parametrize("count", [7, 129, 2052])
def test_prefill_retires_completed_tail_packets_before_reuse(monkeypatch, count):
    packet = PackedC1Buffers("cpu")
    events = []

    def forward(_req_id, chunk, _start, **_kwargs):
        if len(chunk) == 1:
            packet.acquire()
        events.append("forward")
        return None

    def complete_packet():
        assert events[-2:] == ["drain", "synchronize"]
        packet.complete()
        events.append("retire")

    monkeypatch.setattr(torch.hpu, "synchronize", lambda: events.append("synchronize"))
    request = SimpleNamespace(num_computed_tokens=0, tokens=list(range(count)), prompt=list(range(count)))
    runner = SimpleNamespace(
        round_timing_enabled=False,
        requests={"request": request},
        _bind_request=lambda _request: None,
        use_dspark=False,
        model_config=SimpleNamespace(max_model_len=1 << 20),
        verify_timing=None,
        model=SimpleNamespace(last_aux=None, complete_step=lambda _count: None),
        _forward=forward,
        _insert=lambda _aux, _positions: None,
        positions=list(range(count)),
        pp=SimpleNamespace(group=SimpleNamespace(is_last_rank=False),
                           drain=lambda: events.append("drain"),
                           complete_packet=complete_packet),
    )
    V41ModelRunner._execute_request(runner, SimpleNamespace(scheduled_spec_decode_tokens={}), "request", count)
    chunks = list(target_chunks(request.tokens))
    assert events.count("retire") == len(chunks) - 1
    assert packet.generation == sum(len(chunk) == 1 for _, chunk in chunks)
    # The last consumer is still in flight; sampling must retire this packet.
    assert packet.active is not None
    assert runner.pending[2] == count
    with pytest.raises(RuntimeError, match="consumer has not completed"):
        packet.acquire()
    packet.complete()
    packet.acquire()


def test_every_scheduler_length_uses_only_prepared_shapes():
    prepared = set(PREFILL_COMPUTE_BUCKETS) | {1}
    for count in range(1, PREFILL_BLOCK_TOKENS + 1):
        chunks = list(target_chunks(range(count)))
        assert all(len(chunk) in prepared for _, chunk in chunks)
        assert sum(len(chunk) for _, chunk in chunks) == count


def test_native_kv_codec_covers_the_complete_prefill_transaction():
    assert NATIVE_KV_CODEC_TOKENS == PREFILL_BLOCK_TOKENS == 8192


def test_scheduler_transaction_uses_one_search_bucket_for_all_internal_tiles():
    assert target_search_length(0, 8192, 1 << 20) == 8192
    assert target_search_length(8192, 823, 1 << 20) == 16384
    assert target_search_length((1 << 20) - 64, 64, 1 << 20) == 1 << 20


def test_runtime_indexer_prewarms_and_reuses_one_bounded_2k_bucket():
    capacity = 1 << 20
    assert list(decode_search_warmups(capacity, runtime_indexer=True)) == [(0, 2560), (2560, capacity)]
    for start, count in ((0, 2052), (2051, 1), (2306, 254)):
        assert runtime_search_length(start, count, capacity) == 2560
    assert runtime_search_length(2560, 1, capacity) == capacity


def test_prefill_tail_is_exact_and_never_splits_into_dspark_c6():
    for count in (1, 3, 6, 7, 127, 129, 8191):
        tokens = list(range(count))
        chunks = list(target_chunks(tokens))
        assert [token for _, chunk in chunks for token in chunk] == tokens
        assert all(len(chunk) in set(PREFILL_COMPUTE_BUCKETS) | {1} for _, chunk in chunks)
        if count != 6:
            assert not (len(chunks) > 1 and any(len(chunk) == 6 for _, chunk in chunks))
        assert all(offset == sum(len(previous) for _, previous in chunks[:index])
                   for index, (offset, _) in enumerate(chunks))
