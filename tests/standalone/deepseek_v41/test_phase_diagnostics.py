# SPDX-License-Identifier: Apache-2.0
"""Host diagnostics must preserve calls and never invent native event owners."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_diagnostics import phase_fields, phase_name, trace_phase


def test_disabled_instrumentation_returns_original_function(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_DSV41_PHASE_TRACE", raising=False)

    def operation(self, value):
        return value

    assert trace_phase(operation) is operation


def test_enabled_ranges_keep_generation_and_propagate_failures(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PHASE_TRACE", "1")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record)
    worker = SimpleNamespace(pp=SimpleNamespace(generation=3, packed=None))
    scheduled = SimpleNamespace(num_scheduled_tokens={"test-request": 1})

    @trace_phase
    def execute_model(self, scheduled):
        return len(scheduled.num_scheduled_tokens)

    @trace_phase
    def sample(self):
        raise RuntimeError("state already changed")

    assert execute_model(worker, scheduled) == 1
    with pytest.raises(RuntimeError, match="already changed"):
        sample(worker)
    fields = [phase_fields(name) for name in names]
    assert all(row["generation"] == 4 and row["rank"] == 2 and row["stage"] == 1 for row in fields)
    assert all(row["segment"] is None and row["completion_event"] is None for row in fields)
    assert all("test-request" not in name for name in names)


def test_name_survives_exporters_without_json_string_escaping():
    fields = dict(phase='quoted";\\scope', rank=2, generation=13, completion_event=None)
    name = phase_name(fields)
    assert json.loads('{"name":"' + name + '"}')["name"] == name
    assert phase_fields(name) == fields


def test_raw_diagnostics_reach_explicit_scopes_without_torch_profiler(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_native_trace

    monkeypatch.setenv("VLLM_HPU_DSV41_PHASE_TRACE", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_RAW_TRACE", "1")
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(deepseek_v41_native_trace, "scope", record)
    monkeypatch.setattr(torch.profiler, "record_function",
                        lambda name: pytest.fail("Raw scopes must not require Torch CPU operator profiling"))

    @trace_phase
    def entry(self, value):
        return value + 1

    assert entry(None, 4) == 5
    assert len(names) == 1 and phase_fields(names[0])["phase"].endswith("entry")


def test_stage_identity_uses_shared_tp_topology(monkeypatch):
    monkeypatch.setenv('VLLM_HPU_DSV41_PHASE_TRACE', '1')
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 3)
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, 'record_function', record)
    worker = SimpleNamespace(pp=SimpleNamespace(generation=0, packed=None),
                             model=SimpleNamespace(tensor_parallel_size=4, pp_rank=0))

    @trace_phase
    def execute_model(self, scheduled):
        return 17

    assert execute_model(worker, SimpleNamespace(num_scheduled_tokens={'r': 1})) == 17
    assert phase_fields(names[0])['stage'] == 0
