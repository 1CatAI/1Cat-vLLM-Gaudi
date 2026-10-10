# SPDX-License-Identifier: Apache-2.0
"""Lowered-cache validity, fresh inputs and bounded metadata ownership."""
import pytest
import torch

from vllm_gaudi.compilation.deepseek_v41_backend_cache import restore_or_compile, serialize_callable


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


def test_saved_callable_uses_current_inputs_without_compiling(tmp_path):

    def cold(graph, inputs):

        def compiled(x, weight):
            return x @ weight

        return compiled

    restore_or_compile(cold, None, [], tmp_path, "1" * 64)

    def must_not_compile(*args):
        raise AssertionError("A cache hit must bypass the backend")

    restored = restore_or_compile(must_not_compile, None, [], tmp_path, "1" * 64)
    x = torch.ones((6, 8))
    for value in (1., 2., 3.):
        assert torch.equal(restored(x, torch.full((8, 8), value)), torch.full((6, 8), value * 8))


def test_concrete_tensor_closure_cannot_be_persisted():
    state = torch.zeros(8)

    def compiled(x):
        state.copy_(x)
        return state

    with pytest.raises(ValueError, match="concrete"):
        serialize_callable(compiled)


def test_corrupt_artifact_rebuilds_only_that_entry(tmp_path):
    calls = []

    def backend(graph, inputs):
        calls.append(True)
        return lambda x: x * 2

    restore_or_compile(backend, None, [], tmp_path, "1" * 64)
    (tmp_path / ("1" * 64 + ".bin")).write_bytes(b"truncated")
    result = restore_or_compile(backend, None, [], tmp_path, "1" * 64)
    assert result(3) == 6
    assert len(calls) == 2
    assert (tmp_path / ("1" * 64 + ".json")).exists()


def test_changed_identity_does_not_select_other_recipes(tmp_path):
    restore_or_compile(lambda g, i: lambda x: x * 2, None, [], tmp_path, "1" * 64)
    changed = restore_or_compile(lambda g, i: lambda x: x * 3, None, [], tmp_path, "2" * 64)
    assert changed(5) == 15
