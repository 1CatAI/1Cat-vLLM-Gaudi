# SPDX-License-Identifier: Apache-2.0
"""Lowered-cache validity, fresh inputs and bounded metadata ownership."""
import pytest
import torch
import cloudpickle

from vllm_gaudi.compilation.deepseek_v41_backend_cache import (portable_jit, restore_jit, restore_or_compile,
                                                               serialize_callable)


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


def test_hot_lowered_restore_does_not_reconstruct_unused_frontend(tmp_path):
    calls = []

    def graph_factory():
        calls.append("graph")
        return torch.fx.symbolic_trace(lambda x: x * 2)

    def backend(graph, inputs):
        return graph.forward

    restore_or_compile(backend, None, [], tmp_path, "1" * 64, graph_factory=graph_factory)
    assert calls == ["graph"]

    def must_not_restore():
        raise AssertionError("A complete lowered hit must not deserialize the unused frontend")

    result = restore_or_compile(backend, None, [], tmp_path, "1" * 64, graph_factory=must_not_restore)
    for value in (1., 2., 3.):
        assert torch.equal(result(torch.full((6, 8), value)), torch.full((6, 8), value * 2))
    (tmp_path / ("1" * 64 + ".bin")).write_bytes(b"corrupt")
    repaired = restore_or_compile(backend, None, [], tmp_path, "1" * 64, graph_factory=graph_factory)
    assert repaired(3) == 6
    assert calls == ["graph", "graph"]


def test_only_certified_literal_buffers_are_restored_in_fresh_allocations():
    root = torch.nn.Module()
    literal = torch.tensor([1., 0., 0., 0.])
    root.register_buffer("_tensor_constant0", literal)
    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    constant = graph.get_attr("_tensor_constant0")
    graph.output(graph.call_function(torch.add, (inputs, constant)))
    module = torch.fx.GraphModule(root, graph)
    with pytest.raises(ValueError, match="constant"):
        serialize_callable(module.forward)
    with pytest.raises(ValueError, match="constant"):
        serialize_callable(module.forward, literals=[torch.zeros(4)])
    data = serialize_callable(module.forward, literals=[literal.clone()])
    restored = cloudpickle.loads(data)
    assert torch.equal(restored(torch.ones(4)), torch.tensor([2., 1., 1., 1.]))
    assert module._tensor_constant0 is literal
    assert restored.__self__._tensor_constant0.data_ptr() != literal.data_ptr()
    assert torch.equal(restored.__self__._tensor_constant0, literal)


def test_compatible_transport_key_avoids_backend_reprocessing(tmp_path):
    restore_or_compile(lambda g, i: lambda x: x * 2, None, [], tmp_path, "1" * 64)

    def must_not_compile(*args):
        raise AssertionError("Compatible metadata must preserve the processed graph")

    restored = restore_or_compile(must_not_compile, None, [], tmp_path, "2" * 64, compatible=("1" * 64, ))
    assert restored(3) == 6


def test_jit_tensor_literal_transport_preserves_scalar_nodes_and_values():
    from habana_frameworks.torch._torch_jit_C import jit

    literal = torch.tensor([1., 0., 0., 0.])
    original = torch.jit.trace(lambda x: x * 2 + torch.tensor([1., 0., 0., 0.]), (torch.ones(4), ))
    text, records = portable_jit(jit.createFromUpstreamGraph(original.graph), [literal, torch.tensor(2)])
    restored = torch._C._create_function_from_graph("restored", restore_jit(text, records))
    for value in (1., 2., 3.):
        assert torch.equal(restored(torch.full((4, ), value)), original(torch.full((4, ), value)))
