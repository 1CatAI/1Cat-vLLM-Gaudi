# SPDX-License-Identifier: Apache-2.0
"""Deferred recurrence, group output and event/lifetime contracts."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from flashinfer_gaudi._reference import qwen38_fused_decode_step_direct
from vllm_gaudi.models.qwen3_next import HpuQwen3DecoderLayerGroup, HpuQwen3AsyncStateGroup
from vllm_gaudi.ops.gdn_async_state import GDNStateDMAPipeline, GDNQueuedStateDMAPipeline


class FakeStream:

    def __init__(self, api, name):
        self.api, self.name = api, name
        self.hpu_stream = 0 if name == "compute" else 2

    def record_event(self):
        event = (self.name, len(self.api.operations))
        self.api.operations.append(("record", event))
        return event

    def wait_event(self, event):
        self.api.operations.append(("wait", self.name, event))

    def synchronize(self):
        self.api.operations.append(("synchronize", self.name))


class FakeHPU:

    def __init__(self):
        self.operations = []
        self.compute = FakeStream(self, "compute")
        self.current = self.compute

    def Stream(self):
        return FakeStream(self, "dma")

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, selected):
        previous, self.current = self.current, selected
        yield
        self.current = previous

    def record_stream(self, tensor, stream):
        self.operations.append(("retain", id(tensor), stream.name))

    def synchronize(self):
        self.operations.append(("synchronize", "device"))


def test_next_group_does_not_wait_for_unrelated_write_and_sources_are_retained():
    api = FakeHPU()
    pipeline = GDNStateDMAPipeline(api=api)
    first, second = torch.zeros(2), torch.ones(2)
    pipeline.submit(0, ((first, second), ))
    pending = pipeline._pending[0]
    assert pending.tensors[0][1] is second
    assert ("retain", id(second), "dma") in api.operations
    assert ("retain", id(first), "dma") in api.operations
    pipeline.before_read(1)
    assert not any(op[:2] == ("wait", "compute") for op in api.operations)
    pipeline.before_read(0)
    assert ("wait", "compute", pending.done) in api.operations
    assert not any(op[0] == "synchronize" for op in api.operations)
    assert pipeline.consumer_waits == 1 and not pipeline._pending


def test_reset_waits_for_all_groups_and_duplicate_submit_fails_before_mutation():
    api = FakeHPU()
    pipeline = GDNStateDMAPipeline(api=api)
    destination = torch.zeros(1)
    pipeline.submit(0, ((destination, torch.ones(1)), ))
    with pytest.raises(RuntimeError, match="consume"):
        pipeline.submit(0, ((destination, torch.full((1, ), 9.0)), ))
    assert destination.item() == 1
    pipeline.submit(1, ((torch.zeros(1), torch.ones(1)), ))
    pipeline.wait_all()
    assert pipeline.flush_waits == 2 and not pipeline._pending
    assert not any(op[0] == "synchronize" for op in api.operations)
    pipeline.submit(0, ((destination, torch.ones(1)), ))
    pipeline.synchronize()
    assert pipeline.host_synchronizations == 1 and not pipeline._pending


def test_native_batch_is_submitted_on_copy_stream_between_events():
    api = FakeHPU()
    captured = []

    def native(updates):
        assert api.current.name == "dma"
        assert api.operations[-1][0:2] == ("wait", "dma")
        captured.append(updates)
        api.operations.append(("native", ))

    pipeline = GDNStateDMAPipeline(api=api, copy_function=native)
    updates = ((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    pipeline.submit(0, updates)
    assert captured == [updates]
    assert pipeline.submitted_bytes == 20
    # Native completion owns lifetime; Python must not add an eager copy or
    # record_stream invocation that would reintroduce per-tensor submission.
    assert all(torch.count_nonzero(destination) == 0 for destination, _ in updates)
    assert api.operations[-2] == ("native", ) and api.operations[-1][0] == "record"
    assert not any(op[0] == "retain" for op in api.operations)
    pipeline.before_read(0)
    assert api.operations[-1][0:2] == ("wait", "compute")


def test_queued_backend_preserves_only_required_waits_without_python_stream_operations():
    api = FakeHPU()
    submissions, waits = [], []

    def submit(backend, sources, destinations, stream):
        assert backend is sentinel
        assert api.current is api.compute
        ticket = object()
        submissions.append((ticket, sources, destinations, stream))
        return ticket

    sentinel = object()
    runtime = (SimpleNamespace(queue_gdn_state_copies=submit, queue_gdn_state_waits=waits.append), sentinel)
    pipeline = GDNQueuedStateDMAPipeline(api=api, runtime=runtime)
    pipeline.submit(0, ((torch.zeros(2), torch.ones(2)), ))
    pipeline.before_read(1)
    assert not waits
    pipeline.submit(1, ((torch.zeros(3), torch.ones(3)), ))
    pipeline.before_read(0)
    assert waits == [[submissions[0][0]]]
    pipeline.wait_all()
    assert waits == [[submissions[0][0]], [submissions[1][0]]]
    assert [item[3] for item in submissions] == [2, 2]
    assert pipeline.submitted_bytes == 20
    assert not api.operations
    # Teardown must drain even when wait_all has already cleared the tickets.
    pipeline.synchronize()
    assert api.operations == [("synchronize", "device")]
    assert pipeline.host_synchronizations == 1


def test_queued_groups_reject_overlapping_views_before_native_submission():
    calls = []
    runtime = (SimpleNamespace(queue_gdn_state_copies=lambda *args: calls.append(args),
                               queue_gdn_state_waits=lambda tickets: None), None)
    pipeline = GDNQueuedStateDMAPipeline(api=FakeHPU(), runtime=runtime)
    pool = torch.zeros(4)
    pipeline.submit(0, ((pool[:2], torch.ones(2)), ))
    with pytest.raises(ValueError, match="disjoint"):
        pipeline.submit(1, ((pool[1:3], torch.ones(2)), ))
    assert len(calls) == 1
    pipeline.submit(1, ((pool[2:], torch.ones(2)), ))
    assert len(calls) == 2
    pipeline.clear_bindings()
    assert not pipeline._bound_groups and not pipeline._pending


def test_precise_events_keep_reset_dependency_on_all_consumers():
    calls, waits = [], []

    def submit(*args):
        ticket = object()
        calls.append((ticket, args))
        return ticket

    runtime = (SimpleNamespace(queue_gdn_state_copies=lambda *args: pytest.fail("legacy submit"),
                               queue_gdn_state_copies_precise=submit,
                               queue_gdn_state_waits=lambda tickets, **kw: waits.append((tickets, kw))), None)
    pipeline = GDNQueuedStateDMAPipeline(api=FakeHPU(), runtime=runtime, precise_events=True)
    pipeline.submit(0, ((torch.zeros(2), torch.ones(2)), ))
    pipeline.submit(1, ((torch.zeros(3), torch.ones(3)), ))
    pipeline.before_read(0)
    assert waits == [([calls[0][0]], {})]
    pipeline.wait_all()
    assert waits[-1] == ([calls[1][0]], {"all_consumers": True})


@pytest.mark.parametrize("tp", [1, 2])
def test_functional_ssm_is_exact_and_does_not_modify_cache_before_commit(tp):
    rng = torch.Generator().manual_seed(734 + tp)

    def rand(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=rng) * 0.03).to(dtype)

    heads, width = 48 // tp, 10240 // tp
    conv = rand(1, 3, width)
    conv_deferred = conv.clone()
    state = rand(1, heads, 128, 128, dtype=torch.float32)
    state_deferred = state.clone()
    weight = rand(width, 4)
    a_log, dt_bias = rand(heads, dtype=torch.float32) - 2, rand(heads)
    for _ in range(5):
        packed, a, b = rand(1, width), rand(1, heads), rand(1, heads)
        original = state_deferred.clone()
        expected, _, _ = qwen38_fused_decode_step_direct(packed, a, b, a_log, dt_bias, conv, weight, None, state,
                                                         128**-0.5)
        output, _, update = qwen38_fused_decode_step_direct(packed,
                                                            a,
                                                            b,
                                                            a_log,
                                                            dt_bias,
                                                            conv_deferred,
                                                            weight,
                                                            None,
                                                            state_deferred,
                                                            128**-0.5,
                                                            inplace_state=False)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv_deferred, conv, rtol=0, atol=0)
        torch.testing.assert_close(state_deferred, original, rtol=0, atol=0)
        torch.testing.assert_close(update, state, rtol=0, atol=0)
        assert update.untyped_storage().data_ptr() != state_deferred.untyped_storage().data_ptr()
        state_deferred.copy_(update)


class StateAttention(torch.nn.Module):

    def __init__(self, state):
        super().__init__()
        self._hpu_active_ssm_state = state
        self._hpu_defer_ssm_writeback = False
        self._hpu_pending_ssm_state = None

    def prepare_decode_state_view(self, *args):
        pass

    def forward(self, x):
        update = self._hpu_active_ssm_state * 0.5 + x
        if self._hpu_defer_ssm_writeback:
            self._hpu_pending_ssm_state = update
        else:
            self._hpu_active_ssm_state.copy_(update)
        return x + update.sum()


class Layer(torch.nn.Module):

    def __init__(self, state):
        super().__init__()
        self.linear_attn = StateAttention(state)

    def forward(self, *, positions, hidden_states, residual):
        return self.linear_attn(hidden_states), residual + hidden_states


def test_compiled_group_returns_fresh_states_for_normal_stream_wrapper(monkeypatch):
    # This AOT/CPU contract must not acquire a shared HPU during Dynamo setup.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    import vllm_gaudi.models.qwen3_next as model

    metadata = SimpleNamespace(is_prompt=False, direct_gdn_state=True)
    monkeypatch.setattr(model, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    pool = torch.zeros(98, 2)
    layers = (Layer(pool[1:2]), Layer(pool[33:34]))
    group = HpuQwen3DecoderLayerGroup(layers)
    compiled = torch.compile(group, backend="aot_eager", fullgraph=True, dynamic=False)
    pipeline = GDNStateDMAPipeline(api=FakeHPU())
    wrapper = HpuQwen3AsyncStateGroup(compiled, group, 0, pipeline)
    expected = pool.clone()
    for step in range(4):
        x = torch.tensor([[0.01 * (step + 1)]])
        expected_x = x
        for start in (1, 33):
            update = expected[start:start + 1] * 0.5 + expected_x
            expected_x = expected_x + update.sum()
            expected[start:start + 1].copy_(update)
        out, _ = wrapper(positions=torch.zeros(1), hidden_states=x, residual=torch.zeros_like(x))
        torch.testing.assert_close(out, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(pool, expected, rtol=0, atol=0)
        assert all(not layer.linear_attn._hpu_defer_ssm_writeback for layer in layers)
        assert all(layer.linear_attn._hpu_pending_ssm_state is None for layer in layers)
    assert wrapper.async_calls == 4 and pipeline.submitted_groups == 4
    assert pipeline.consumer_waits == 3 and pipeline.host_synchronizations == 0
