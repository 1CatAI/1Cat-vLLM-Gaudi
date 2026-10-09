# SPDX-License-Identifier: Apache-2.0
"""Saved frontend guards rebind current buffers and preserve state writes."""
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.compilation.deepseek_v41_frontend_cache import GuardedFrontendEntry, cached_group_entry


@pytest.fixture(autouse=True)
def cpu_frontend_only(monkeypatch):
    # Plugin discovery can register HPU as the default accelerator even for
    # CPU tensors. These contract checks must never acquire a device.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


class StateGroup(torch.nn.Module):

    def __init__(self, value):
        super().__init__()
        self.register_buffer("weight", torch.full((8, 8), value))
        self.register_buffer("state", torch.zeros((6, 8)))

    def forward(self, inputs):
        value = inputs @ self.weight
        self.state.copy_(value)
        return value, self.state


def backend(graph, inputs):
    return graph.forward


def cached(owner, directory, **options):
    return GuardedFrontendEntry(StateGroup.forward, owner, backend, directory, identity="1" * 64, **options)


def test_restored_frontend_uses_current_owner_and_retains_aliases(tmp_path):
    previous = StateGroup(1.)
    inputs = torch.ones((6, 8))
    original = cached(previous, tmp_path)
    original(inputs)
    fresh = StateGroup(2.)
    restored = cached(fresh, tmp_path)
    for value in (1., 2., 3.):
        inputs.fill_(value)
        result, state = restored(inputs)
        assert torch.equal(result, inputs @ fresh.weight)
        assert state.data_ptr() == fresh.state.data_ptr()
        assert torch.equal(state, result)
    assert torch.equal(previous.state, torch.full_like(previous.state, 8.))
    assert restored.stats["restores"] == 1
    assert restored.stats["captures"] == 0
    assert restored.stats["hits"] == 3


def test_incompatible_shape_fails_before_state_execution(tmp_path):
    group = StateGroup(1.)
    entry = cached(group, tmp_path, max_variants=1)
    entry(torch.ones((6, 8)))
    before = group.state.clone()
    with pytest.raises(RuntimeError, match="capacity"):
        entry(torch.ones((5, 8)))
    assert torch.equal(group.state, before)


def test_changed_artifact_or_dependencies_fail_closed(tmp_path):
    group = StateGroup(1.)
    cached(group, tmp_path)(torch.ones((6, 8)))
    with pytest.raises(ValueError, match="fingerprint"):
        GuardedFrontendEntry(StateGroup.forward, StateGroup(1.), backend, tmp_path, identity="2" * 64)
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        cached(StateGroup(1.), tmp_path)


def test_shared_group_factory_rebinds_a_named_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.setenv("DSV41_SERVING_COMPILE_IDENTITY", "1" * 64)
    group = StateGroup(1.)
    group.layers = [SimpleNamespace(layer=20)]
    entry = cached_group_entry(StateGroup.forward,
                               group,
                               "eager",
                               tmp_path,
                               native=True,
                               shared_coordinates=False,
                               memory_ready=False)
    inputs = torch.ones((6, 8))
    assert torch.equal(entry(inputs)[0], inputs @ group.weight)
    fresh = StateGroup(2.)
    fresh.layers = [SimpleNamespace(layer=20)]
    restored = cached_group_entry(StateGroup.forward,
                                  fresh,
                                  "eager",
                                  tmp_path,
                                  native=True,
                                  shared_coordinates=False,
                                  memory_ready=False)
    assert torch.equal(restored(inputs)[0], inputs @ fresh.weight)
    assert restored.stats["captures"] == 0
