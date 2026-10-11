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
    prefill_search_length,
    prefill_search_warmups,
    runtime_search_length,
    prefill_compute_buckets,
    target_chunks,
    target_search_length,
)
from vllm_gaudi.ops.deepseek_v41_math import NATIVE_KV_CODEC_TOKENS


def test_serving_prefill_warmup_bounds_pool_pages_to_request_capacity(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
    from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState

    slots = RequestSlots(1)
    slot = slots.acquire("warmup")
    retired = []
    bank = SimpleNamespace(slots=slots, pages=torch.zeros(1, 2048, dtype=torch.int32),
                           page_host=torch.zeros(1, 2048, dtype=torch.int32), page_versions={}, pending=None,
                           _invalidate_index_mirror=lambda: None, acquire=lambda _: slot,
                           warmup_single_handoff=lambda _: None, bind_prefill=lambda _: None,
                           restore_single_bindings=lambda: retired.append("restore"),
                           release=lambda _: retired.append("release"))
    bank.publish_pages = lambda *args: BatchStageState.publish_pages(bank, *args)
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner.request_batches = None
    runner.prefix_checkpoints = object()
    runner.model = SimpleNamespace(batch_state=bank, pp_rank=0)
    runner.state = SimpleNamespace(blocks=2056)
    runner.model_config = SimpleNamespace(max_model_len=262144)
    runner.prefill_capacity = 16384

    def first_prefill(count, *, start_position, prefill):
        assert prefill
        assert torch.equal(bank.pages[0], torch.arange(1, 2049, dtype=torch.int32))
        raise RuntimeError("checked first production prefill contract")

    runner._dummy_run = first_prefill
    with pytest.raises(RuntimeError, match="checked first production"):
        runner.warmup_model()
    assert retired == ["restore", "release"]


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


@pytest.mark.parametrize("count", (1, 6))
def test_native_warmup_covers_interior_visible_prefix_keys(count):
    from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound

    maximum = 262144
    warmups = list(decode_search_warmups(maximum, native_visible_prefixes=True))
    covered = {(search, decode_source_prefix_bound(start + count, search, 4))
               for start, search in warmups if search <= 32768}
    for start in range(32768 - count + 1):
        search = target_search_length(start, count, maximum)
        assert (search, decode_source_prefix_bound(start + count, search, 4)) in covered
    old = list(decode_search_warmups(maximum))
    assert set(warmups) - set(old) == {(12288, 16384), (20480, 32768), (24576, 32768), (28672, 32768)}
    assert [(start, search) for start, search in warmups if search > 32768] == [(start, search) for start, search in old
                                                                                if search > 32768]


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
    runner.request_batches = None
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


def test_scheduler_transaction_uses_finite_exact_compute_buckets(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", raising=False)
    tokens = list(range(8192))
    chunks = list(target_chunks(tokens))
    assert PREFILL_BLOCK_TOKENS == 8192
    assert [len(chunk) for _, chunk in chunks] == [8192]
    assert [token for _, chunk in chunks for token in chunk] == tokens


def test_larger_compute_tile_preserves_scheduler_budget_and_exact_tail(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", "8192")
    for count in (8191, 8192, 8193, 32768):
        tokens = list(range(count))
        chunks = list(target_chunks(tokens))
        assert [token for _, chunk in chunks for token in chunk] == tokens
        assert all(len(chunk) in set(prefill_compute_buckets()) | {1} for _, chunk in chunks)
        assert all(len(chunk) <= PREFILL_BLOCK_TOKENS for _, chunk in chunks)
    assert [len(chunk) for _, chunk in target_chunks(range(32768))] == [8192] * 4
    assert [len(chunk) for _, chunk in target_chunks(range(8193))] == [8192, 1]


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
    request = SimpleNamespace(
        num_computed_tokens=0, tokens=list(range(count)), prompt=list(range(count)), decode_start=count
    )
    request.output = []
    request.token_slice = lambda start, stop: request.tokens[start:stop]
    runner = SimpleNamespace(
        round_timing_enabled=False,
        prefill_capacity=8192,
        requests={"request": request},
        _bind_request=lambda _request: None,
        _prepare_device_sampling_request=lambda _request: None,
        use_dspark=False,
        model_config=SimpleNamespace(max_model_len=1 << 20),
        verify_timing=None,
        model=SimpleNamespace(last_aux=None, pp_rank=0, tp_rank=0, complete_step=lambda _count: None),
        _forward=forward,
        _insert=lambda _aux, _positions: None,
        positions=list(range(count)),
        pp=SimpleNamespace(
            generation=0,
            group=SimpleNamespace(is_last_rank=False),
            drain=lambda: events.append("drain"),
            complete_packet=complete_packet,
        ),
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
    assert NATIVE_KV_CODEC_TOKENS == 16384
    assert PREFILL_BLOCK_TOKENS == 8192


def test_scheduler_transaction_uses_one_search_bucket_for_all_internal_tiles():
    assert target_search_length(0, 8192, 1 << 20) == 8192
    assert target_search_length(8192, 823, 1 << 20) == 16384
    assert target_search_length((1 << 20) - 64, 64, 1 << 20) == 1 << 20


def test_runtime_indexer_prewarms_and_reuses_one_bounded_2k_bucket():
    capacity = 1 << 20
    assert list(decode_search_warmups(capacity, runtime_indexer=True)) == [
        (0, 512),
        (512, 2560),
        *[(start, 32768) for start in range(2560, 32768, 4096)],
        *[(start, search) for lower, search in ((32768, 65536), (65536, 131072), (131072, 262144),
                                              (262144, 524288), (524288, 1048576))
          for start in range(lower, search, min(32768, max(4096, search // 8)))],
    ]
    for start, count in ((0, 2052), (2051, 1), (2306, 254)):
        assert runtime_search_length(start, count, capacity) == 2560
    assert runtime_search_length(2560, 1, capacity) == 32768
    assert runtime_search_length(32768, 1, capacity) == 65536


def test_shared_prefill_geometry_does_not_change_decode_or_long_context_capacity():
    capacity = 1 << 20
    for start, count in ((2560, 128), (8192, 8192), (16384, 1), (24576, 8192)):
        expected = 16384 if start + count <= 16384 else 32768
        assert prefill_search_length(start, count, capacity, reuse_index_keys=True) == expected
        assert runtime_search_length(start, count, capacity) == 32768
        assert prefill_search_length(start, count, capacity) == capacity
    assert prefill_search_length(32768, 128, capacity, reuse_index_keys=True) == 65536
    assert prefill_search_length(512, 128, capacity, reuse_index_keys=True) == 2560
    assert prefill_search_length(0, 128, capacity, reuse_index_keys=True) == 512
    assert prefill_search_length(8192, 8192, 16384, reuse_index_keys=True) == 16384
    with pytest.raises(ValueError):
        prefill_search_length(capacity - 128, 256, capacity, reuse_index_keys=True)


@pytest.mark.parametrize("capacity", [65536, 100000, 524288, 1048576])
def test_long_runtime_geometry_covers_crossings_and_is_prepared_before_readiness(capacity):
    warmups = list(decode_search_warmups(capacity, runtime_indexer=True))
    long_buckets = {search for _, search in warmups if search > 32768}
    for end in [32768, 32769, 41983, 62463, 65536, 65537, capacity]:
        if end > capacity:
            continue
        search = runtime_search_length(end - 1, 1, capacity)
        assert end <= search <= capacity
        if end > 32768:
            assert search < 2 * end
            assert search in long_buckets
            assert prefill_search_length(end - 128, 128, capacity, reuse_index_keys=True) == search
    assert len(long_buckets) <= 5


@pytest.mark.parametrize("capacity", [65536, 100000, 524288, 1048576])
def test_every_long_decode_prefix_geometry_is_warmed(capacity):
    from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound
    prepared = {(search, decode_source_prefix_bound(start + 1, search, 4))
                for start, search in decode_search_warmups(capacity, runtime_indexer=True)}
    for end in range(32769, capacity + 1):
        search = runtime_search_length(end - 1, 1, capacity)
        assert (search, decode_source_prefix_bound(end, search, 4)) in prepared


def test_prefill_warmup_covers_residual_tiles_in_each_reachable_search_bucket():
    capacity = 524288
    warmed = {(count, prefill_search_length(start, count, capacity, reuse_index_keys=True))
              for start, count in prefill_search_warmups(capacity, 16384)}
    assert len(warmed) <= 8 * 8
    for prompt in [1024, 16384, 21504, 41983, 62463, 65537, 131073, 262145, capacity]:
        for start in range(0, prompt, 16384):
            count = min(16384, prompt - start)
            search = prefill_search_length(start, count, capacity, reuse_index_keys=True)
            for _, chunk in target_chunks(range(count), 16384):
                if len(chunk) > 1:
                    assert (len(chunk), search) in warmed


def test_complete_prompt_search_retains_decoder_halo_mla_admission():
    from vllm_gaudi.ops.deepseek_v41_prefill_sequence import can_partition_prefill_mla

    for capacity in (16384, 524288, 1048576):
        search = prefill_search_length(0, 16384, capacity, reuse_index_keys=True)
        assert can_partition_prefill_mla((16384, 16, 512), 640, 4, search)
        assert can_partition_prefill_mla((4096, 16, 512), 640, 4, search)
    assert can_partition_prefill_mla((4096, 16, 512), 640, 4, 32768)
    assert can_partition_prefill_mla((4096, 16, 512), 640, 4, 65536)
    for search in (131072, 262144, 524288, 1048576):
        assert can_partition_prefill_mla((4096, 16, 512), 640, 4, search)


def test_prefill_tail_is_exact_and_never_splits_into_dspark_c6():
    for count in (1, 3, 6, 7, 127, 129, 8191):
        tokens = list(range(count))
        chunks = list(target_chunks(tokens))
        assert [token for _, chunk in chunks for token in chunk] == tokens
        assert all(len(chunk) in set(PREFILL_COMPUTE_BUCKETS) | {1} for _, chunk in chunks)
        if count != 6:
            assert not (len(chunks) > 1 and any(len(chunk) == 6 for _, chunk in chunks))
        assert all(
            offset == sum(len(previous) for _, previous in chunks[:index]) for index, (offset, _) in enumerate(chunks)
        )


def test_tp4_16k_chunk_and_exact_tail_follow_scheduler_capacity(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_prefill_capacity import prefill_capacity

    monkeypatch.delenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", raising=False)
    assert prefill_capacity(16384, 4) == 16384
    assert prefill_capacity(8192, 4) == 8192
    assert prefill_capacity(16384, 2) == 8192
    for count, expected in ((16384, [16384]), (16385, [16384, 1]), (32768, [16384, 16384])):
        chunks = list(target_chunks(range(count), prefill_capacity(16384, 4)))
        assert [len(chunk) for _, chunk in chunks] == expected
        assert [token for _, chunk in chunks for token in chunk] == list(range(count))
    assert [len(c) for _, c in target_chunks(range(16384), 8192)] == [8192, 8192]
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", "8192")
    assert [len(c) for _, c in target_chunks(range(16384), 16384)] == [8192, 8192]
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS", "16384")
    assert max(prefill_compute_buckets(8192)) == 8192
    assert [len(c) for _, c in target_chunks(range(16384), 16384)] == [16384]
    with pytest.raises(ValueError):
        list(target_chunks(range(32768), 32768))
