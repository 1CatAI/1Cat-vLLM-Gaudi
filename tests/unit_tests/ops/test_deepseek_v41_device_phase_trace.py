# SPDX-License-Identifier: Apache-2.0
"""Device phase observers never add a formal request wait or a false interval."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_gaudi.ops import deepseek_v41_native_trace as trace


def fixture(monkeypatch, *, active):
    calls = []
    events = []

    class Event:
        def __init__(self, **kwargs):
            events.append(self)

        def record(self):
            calls.append('record')

        def synchronize(self):
            raise AssertionError('Observer must not wait in the round')

    factory = Mock(side_effect=Event)
    owner = SimpleNamespace(device_events=[])
    monkeypatch.setattr(trace, '_active', active)
    monkeypatch.setattr(trace, '_device_recorder', owner)
    monkeypatch.setattr(trace, 'scope', lambda label: nullcontext())
    monkeypatch.setattr(trace.torch.hpu, 'Event', factory)
    return calls, events, factory, owner


def test_unprofiled_stage_has_no_device_observer(monkeypatch):
    calls, _, factory, owner = fixture(monkeypatch, active=False)
    with trace.device_scope('target'):
        calls.append('body')
    assert calls == ['body']
    factory.assert_not_called()
    assert not owner.device_events


def test_profiled_stage_records_order_without_wait(monkeypatch):
    calls, events, _, owner = fixture(monkeypatch, active=True)
    with trace.device_scope('target'):
        calls.append('body')
    assert calls == ['record', 'body', 'record']
    label, start, end, begin, done = owner.device_events[0]
    assert label == 'target' and end >= start
    assert begin is events[0] and done is events[1]


def test_failed_stage_is_not_published_as_completed(monkeypatch):
    calls, _, _, owner = fixture(monkeypatch, active=True)
    with pytest.raises(ValueError, match='stage failed'), trace.device_scope('target'):
        raise ValueError('stage failed')
    assert calls == ['record']
    assert not owner.device_events
