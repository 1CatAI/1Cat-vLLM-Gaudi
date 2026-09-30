# SPDX-License-Identifier: Apache-2.0
"""Packet retirement preserves consumer ownership without a frontend drain."""
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops.deepseek_v41_host import _C1Packet


def packet():
    value = _C1Packet.__new__(_C1Packet)
    value.generation = value.pending_generation = 0
    value.inflight = False
    value.native_completion = None
    return value


def test_native_packet_waits_at_reuse_and_retains_each_completion():
    value = packet()
    events, retired = [], []

    def record():
        generation = value.pending_generation
        events.append(('record', generation))
        ticket = SimpleNamespace(synchronize=lambda: events.append(('wait', generation)))
        retired.append(ticket)
        return ticket

    value.prepare_native_completion(record)
    for generation in (1, 2, 3):
        value.pending_generation = generation
        with pytest.raises(RuntimeError, match='pending'):
            value.reuse()
        with pytest.raises(RuntimeError, match='Stale'):
            value.complete(generation - 1, SimpleNamespace(stream_id=0))
        value.complete(generation, SimpleNamespace(stream_id=0))
        assert events[-1] == ('record', generation)
        assert value.consumer_done is retired[-1]
        assert value.inflight and value.pending_generation == 0
        value.reuse()
        assert events[-1] == ('wait', generation)
        assert not value.inflight and value.generation == generation
    assert events == [(kind, generation) for generation in (1, 2, 3) for kind in ('record', 'wait')]


def test_native_packet_rejects_wrong_stream_without_retiring_generation():
    value = packet()
    value.prepare_native_completion(lambda: pytest.fail('Unexpected default-stream event'))
    value.pending_generation = 1
    with pytest.raises(RuntimeError, match='default stream'):
        value.complete(1, SimpleNamespace(stream_id=9))
    assert value.pending_generation == 1 and value.generation == 0 and not value.inflight


def test_packet_completion_mode_cannot_change_after_first_consumer():
    value = packet()
    value.generation = 1
    with pytest.raises(RuntimeError, match='first transaction'):
        value.prepare_native_completion(lambda: None)


def test_ordinary_packet_preserves_actual_consumer_stream():
    value = packet()
    records = []
    value.consumer_done = SimpleNamespace(record=lambda stream: records.append(stream), synchronize=lambda: None)
    stream = SimpleNamespace(stream_id=5)
    value.pending_generation = 1
    value.complete(1, stream)
    assert records == [stream]
    value.reuse()
    assert value.generation == 1 and not value.inflight
