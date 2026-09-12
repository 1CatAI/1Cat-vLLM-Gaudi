# SPDX-License-Identifier: Apache-2.0
"""Host diagnostics must preserve calls and never invent native event owners."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase


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
    fields = [json.loads(name.removeprefix("dsv41_phase:")) for name in names]
    assert all(row["generation"] == 4 and row["rank"] == 2 and row["stage"] == 1 for row in fields)
    assert all(row["segment"] is None and row["completion_event"] is None for row in fields)
    assert all("test-request" not in name for name in names)
