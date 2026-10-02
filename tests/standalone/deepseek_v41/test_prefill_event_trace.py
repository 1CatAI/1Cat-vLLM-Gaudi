# SPDX-License-Identifier: Apache-2.0
"""Unselected requests must not create events or consume diagnostic quota."""
import json
from types import SimpleNamespace

from vllm_gaudi.ops import deepseek_v41_prefill_event_trace as trace


def test_request_and_label_selection_preserves_quota(monkeypatch, tmp_path):
    events = []

    class Event:

        def __init__(self, **kwargs):
            events.append(self)

        def record(self):
            self.stamp = len(events)

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return float(other.stamp - self.stamp)

    monkeypatch.setattr(trace.torch.hpu, "Event", Event)
    monkeypatch.setattr(trace.torch.hpu, "memory_allocated", lambda: 1024)
    monkeypatch.setattr(trace.torch.hpu, "max_memory_allocated", lambda: 2048)
    monkeypatch.setattr(trace, "_active", None)
    monkeypatch.setattr(trace, "_completed", 0)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE", str(tmp_path))
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_LIMIT", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_REQUEST_PREFIX", " chatcmpl-phase- , cmpl-phase- ")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_LABELS", " layer, attention , moe ")
    assert not trace.begin("chatcmpl-formal-1", 1, 16384, 0, 0)
    with trace.span("layer", layer=20, rows=16384):
        pass
    assert not events and trace._completed == 0
    assert trace.begin("cmpl-phase-0", 1, 16384, 0, 0)
    trace.abort()
    events.clear()
    assert trace._completed == 0
    assert trace.begin("chatcmpl-phase-1", 2, 16384, 0, 0)
    with trace.span("layer", layer=20, rows=16384):
        with trace.span("router", layer=20, rows=16384):
            pass
        with trace.span("moe", layer=20, rows=16384):
            pass
    trace.finish()
    result = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert [span["name"] for span in result["spans"]] == ["layer", "moe"]
    assert result["request_id"] == "chatcmpl-phase-1"
    assert result["memory"]["allocated_bytes"] == 1024
    assert result["memory"]["peak_allocated_bytes"] == 2048
    assert len(events) == 6 and trace._completed == 1
    assert not trace.begin("chatcmpl-phase-2", 3, 16384, 0, 0)
    assert len(events) == 6


def test_zero_disables_trace_even_for_a_matching_request(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE", "0")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_REQUEST_PREFIX", "chatcmpl-phase-")
    monkeypatch.setattr(trace, "_active", None)
    monkeypatch.setattr(trace, "_completed", 0)

    def unexpected_event(**kwargs):
        raise AssertionError("Disabled tracing must not allocate an HPU event")

    monkeypatch.setattr(trace.torch.hpu, "Event", unexpected_event)
    assert not trace.begin("chatcmpl-phase-disabled", 1, 16384, 0, 0)
    assert trace._completed == 0 and trace._active is None


def test_serving_trace_includes_sampling_checkpoint_and_rank_ack(monkeypatch, tmp_path):
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    class Event:
        stamp = 0
        def __init__(self, **kwargs):
            Event.stamp += 1
            self.stamp = Event.stamp
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return float(other.stamp - self.stamp)

    monkeypatch.setattr(trace.torch.hpu, 'Event', Event)
    monkeypatch.setattr(trace.torch.hpu, 'memory_allocated', lambda: 1024)
    monkeypatch.setattr(trace.torch.hpu, 'max_memory_allocated', lambda: 2048)
    monkeypatch.setattr(trace, '_active', None)
    monkeypatch.setattr(trace, '_completed', 0)
    monkeypatch.setenv('VLLM_HPU_DSV41_PREFILL_EVENT_TRACE', str(tmp_path))
    runner = object.__new__(V41ModelRunner)
    runner.pending = None
    runner.request_batches = None
    runner.requests = {'a': SimpleNamespace(num_computed_tokens=0)}
    runner.prefix_checkpoints = SimpleNamespace(begin=lambda _: None, capture_at=lambda *_: None,
                                               finish=lambda: [])
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True))
    runner.draft_token_ids = None
    runner._update = lambda _: None
    def execute(*_):
        runner._prefill_event_trace_pending = trace.begin('a', 1, 16384, 0, 0)
    runner._execute_request = execute
    runner._finish_request = lambda: ModelRunnerOutput(req_ids=['a'], req_id_to_index={'a': 0},
                                                      sampled_token_ids=[[17]])
    runner.execute_model(SimpleNamespace(num_scheduled_tokens={'a': 16384}))
    result = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert [row['name'] for row in result['spans']] == [
        'sample_completion', 'prefix_checkpoint', 'prefix_acknowledgment']
    assert trace._active is None and not runner._prefill_event_trace_pending
