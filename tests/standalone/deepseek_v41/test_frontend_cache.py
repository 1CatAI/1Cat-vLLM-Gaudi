# SPDX-License-Identifier: Apache-2.0
"""Saved frontend guards rebind current buffers and preserve state writes."""
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.compilation.deepseek_v41_frontend_cache import (GuardedFrontendEntry, cached_group_entry,
                                                                cached_tensor_entry)


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


def defaults_transform(value, multiplier=2., *, shift=.5):
    return value * multiplier + shift


class DefaultedStateGroup(StateGroup):

    def forward(self, inputs):
        value = defaults_transform(inputs @ self.weight)
        self.state.copy_(value)
        return value, self.state


def nested_transform():

    def transform(value, *, multiplier=2., shift=.5):
        return value * multiplier + shift

    return transform


class NestedDefaultGroup(StateGroup):

    def __init__(self, value):
        super().__init__(value)
        self.transform = nested_transform()

    def forward(self, inputs):
        value = self.transform(inputs @ self.weight)
        self.state.copy_(value)
        return value, self.state


class QualifiedTensorGroup(StateGroup):

    def __init__(self, value, active=8):
        super().__init__(value)
        self.weight.dsv41_active_k = active
        self.weight.dsv41_sat_eligible = True

    def forward(self, inputs):
        assert self.weight.dsv41_sat_eligible
        value = inputs @ self.weight[:, :self.weight.dsv41_active_k]
        self.state.copy_(value)
        return value, self.state


def test_restored_tensor_qualification_keeps_current_weights_and_guards(tmp_path):
    inputs = torch.ones((6, 8))
    old = QualifiedTensorGroup(1.)
    GuardedFrontendEntry(QualifiedTensorGroup.forward, old, backend, tmp_path, identity="1" * 64,
                         max_variants=1)(inputs)
    fresh = QualifiedTensorGroup(2.)
    entry = GuardedFrontendEntry(QualifiedTensorGroup.forward,
                                 fresh,
                                 backend,
                                 tmp_path,
                                 identity="1" * 64,
                                 max_variants=1)
    assert torch.equal(entry(inputs)[0], torch.full((6, 8), 16.))
    assert entry.stats["captures"] == 0
    assert entry.stats["restores"] == 1
    fresh.weight.dsv41_active_k = 7
    before = fresh.state.clone()
    with pytest.raises(RuntimeError, match="capacity"):
        entry(inputs)
    assert torch.equal(fresh.state, before)


class BucketGroup(torch.nn.Module):

    def __init__(self, value, search_length=8):
        super().__init__()
        self.register_buffer("weight", torch.full((8, 8), value))
        self.search_length = search_length

    def forward(self, inputs):
        return (inputs @ self.weight) * self.search_length


def test_only_matching_input_bucket_is_restored(tmp_path):
    previous = BucketGroup(1.)
    entry = GuardedFrontendEntry(BucketGroup.forward, previous, backend, tmp_path, identity="1" * 64)
    for rows in (1, 2, 6):
        entry(torch.ones((rows, 8)))
    fresh = BucketGroup(2.)
    restored = GuardedFrontendEntry(BucketGroup.forward, fresh, backend, tmp_path, identity="1" * 64)
    assert restored.stats["restores"] == 0
    assert torch.equal(restored(torch.ones((6, 8))), torch.full((6, 8), 128.))
    assert restored.stats["restores"] == 1
    assert restored.stats["captures"] == 0
    assert len(restored.pending) == 2


def test_owner_search_bucket_is_indexed_before_restore(tmp_path):
    inputs = torch.ones((6, 8))
    for search in (8, 16, 32):
        owner = BucketGroup(1., search)
        entry = GuardedFrontendEntry(BucketGroup.forward, owner, backend, tmp_path, identity="1" * 64)
        entry(inputs)
    fresh = BucketGroup(2., 32)
    entry = GuardedFrontendEntry(BucketGroup.forward, fresh, backend, tmp_path, identity="1" * 64)
    assert torch.equal(entry(inputs), torch.full((6, 8), 512.))
    assert entry.stats["restores"] == 1
    assert entry.stats["captures"] == 0
    assert len(entry.pending) == 2


def test_restored_nested_keyword_defaults_keep_live_callable_unchanged(tmp_path):
    inputs = torch.ones((6, 8))
    previous, fresh = NestedDefaultGroup(1.), NestedDefaultGroup(2.)
    original = GuardedFrontendEntry(NestedDefaultGroup.forward, previous, backend, tmp_path, identity="1" * 64)
    original(inputs)
    restored = GuardedFrontendEntry(NestedDefaultGroup.forward, fresh, backend, tmp_path, identity="1" * 64)
    for value in (1., 2., 3.):
        inputs.fill_(value)
        assert torch.equal(restored(inputs)[0], (inputs @ fresh.weight) * 2. + .5)
    assert fresh.transform.__kwdefaults__ == {"multiplier": 2., "shift": .5}
    assert restored.stats["restores"] == 1
    assert restored.stats["captures"] == 0


def test_restored_default_arguments_retain_current_weight_binding(tmp_path):
    inputs = torch.ones((6, 8))

    def make(value):
        return GuardedFrontendEntry(DefaultedStateGroup.forward,
                                    DefaultedStateGroup(value),
                                    backend,
                                    tmp_path,
                                    identity="1" * 64,
                                    max_variants=1)

    make(1.)(inputs)
    restored = make(2.)
    assert torch.equal(restored(inputs)[0], torch.full((6, 8), 32.5))
    assert restored.stats["restores"] == 1
    assert restored.stats["captures"] == 0


def test_cached_default_sources_restore_positional_and_keyword_names():
    import pickle
    from torch._dynamo.source import DefaultsSource, GlobalSource
    from vllm_gaudi.compilation.deepseek_v41_frontend_cache import _source_dumps, _source_loads

    sources = (DefaultsSource(GlobalSource("defaults_transform"),
                              0), DefaultsSource(GlobalSource("defaults_transform"), "shift", True))
    for data in (pickle.dumps(sources), _source_dumps(sources)):
        restored = _source_loads(data)
        assert restored == sources
        assert [source.name for source in restored] == [source.name for source in sources]


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


def test_changed_artifact_or_dependencies_rebuild_without_using_old_graph(tmp_path):
    group = StateGroup(1.)
    cached(group, tmp_path)(torch.ones((6, 8)))
    changed = GuardedFrontendEntry(StateGroup.forward, StateGroup(2.), backend, tmp_path, identity="2" * 64)
    assert torch.equal(changed(torch.ones((6, 8)))[0], torch.full((6, 8), 16.))
    assert changed.stats["captures"] == 1
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    restored = GuardedFrontendEntry(StateGroup.forward, StateGroup(3.), backend, tmp_path, identity="2" * 64)
    assert torch.equal(restored(torch.ones((6, 8)))[0], torch.full((6, 8), 24.))
    assert restored.stats["captures"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_corruption_rebuilds_only_affected_shape(tmp_path):
    previous = cached(StateGroup(1.), tmp_path)
    for rows in (1, 6):
        previous(torch.ones((rows, 8)))
    for manifest in tmp_path.glob("*.json"):
        if json.loads(manifest.read_text())["inputs"][1][0][1] == [1, 8]:
            manifest.with_suffix(".bin").write_bytes(b"invalid")
    fresh = cached(StateGroup(2.), tmp_path)
    assert torch.equal(fresh(torch.ones((1, 8)))[0], torch.full((1, 8), 16.))
    assert torch.equal(fresh(torch.ones((6, 8)))[0], torch.full((6, 8), 16.))
    assert fresh.stats["captures"] == 1
    assert fresh.stats["restores"] == 1


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


def test_cpu_relocation_keeps_frontend_but_precision_change_invalidates_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.setenv("DSV41_SERVING_COMPILE_IDENTITY", "1" * 64)
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_CPUS", "10,15,38,43")
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_HELPER_CPUS", "11,12;16,17;39,40;44,45")

    def make(value):
        group = StateGroup(value)
        group.layers = [SimpleNamespace(layer=20)]
        return cached_group_entry(StateGroup.forward,
                                  group,
                                  "eager",
                                  tmp_path,
                                  native=True,
                                  shared_coordinates=False,
                                  memory_ready=False)

    inputs = torch.ones((6, 8))
    make(1.)(inputs)
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_CPUS", "10,1,38,43")
    monkeypatch.setenv("VLLM_HPU_DSV4_WORKER_HELPER_CPUS", "11,12;2,3;39,40;28,29")
    restored = make(2.)
    assert torch.equal(restored(inputs)[0], torch.full((6, 8), 16.))
    assert restored.stats["restores"] == 1
    assert restored.stats["captures"] == 0
    monkeypatch.setenv("VLLM_HPU_DSV41_ATTN_DENSE_FP8", "1")
    changed = make(2.)
    assert changed.stats["restores"] == 0


def test_serving_workers_use_distributed_rank_without_torchrun_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.setenv("DSV41_SERVING_COMPILE_IDENTITY", "1" * 64)
    for rank in range(4):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        group = StateGroup(float(rank + 1))
        group.layers = [SimpleNamespace(layer=20)]
        entry = cached_group_entry(StateGroup.forward,
                                   group,
                                   "eager",
                                   tmp_path,
                                   native=True,
                                   shared_coordinates=False,
                                   memory_ready=False)
        for _ in range(4):
            assert torch.equal(entry(torch.ones((6, 8)))[0], torch.full((6, 8), float((rank + 1) * 8)))
        assert entry.stats["captures"] == 1
        assert entry.directory.name == f"rank{rank}"
        assert len(list(entry.directory.glob("*.json"))) == 1
    for rank in range(4):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        group = StateGroup(2.)
        group.layers = [SimpleNamespace(layer=20)]
        entry = cached_group_entry(StateGroup.forward,
                                   group,
                                   "eager",
                                   tmp_path,
                                   native=True,
                                   shared_coordinates=False,
                                   memory_ready=False)
        assert torch.equal(entry(torch.ones((6, 8)))[0], torch.full((6, 8), 16.))
        assert entry.stats["captures"] == 0
        assert entry.stats["restores"] == 1


def test_tensor_region_method_restores_current_owner_and_keyword_arguments(tmp_path, monkeypatch):
    from torch._dynamo.backends import registry

    monkeypatch.setenv("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.setenv("DSV41_SERVING_COMPILE_IDENTITY", "1" * 64)
    monkeypatch.setattr(registry, "lookup_backend", lambda name: backend)
    inputs = torch.ones((6, 8))
    old = StateGroup(1.)
    first = cached_tensor_entry(StateGroup.forward, (inputs, ), {}, owner=old)
    first(inputs)
    fresh = StateGroup(2.)
    restored = cached_tensor_entry(StateGroup.forward, (inputs, ), {}, owner=fresh)
    for value in (1., 2., 3.):
        inputs.fill_(value)
        output, state = restored(inputs)
        assert torch.equal(output, inputs @ fresh.weight)
        assert state.data_ptr() == fresh.state.data_ptr()
    pure = cached_tensor_entry(defaults_transform, (inputs, ), {"shift": 1.5})
    assert torch.equal(pure(inputs, shift=1.5), inputs * 2 + 1.5)
    again = cached_tensor_entry(defaults_transform, (inputs, ), {"shift": 1.5})
    assert torch.equal(again(inputs, shift=1.5), inputs * 2 + 1.5)
