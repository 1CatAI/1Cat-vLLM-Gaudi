# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.fx import Graph, GraphModule

from vllm_gaudi.compilation.deepseek_v41_integer_constants import (
    propagate_with_resident_buffers, retain_integer_constants, retain_static_factories)


@pytest.mark.parametrize('count', [1, 2, 6])
def test_static_factory_keeps_dynamic_request_positions(count):
    graph = Graph()
    x = graph.placeholder('positions')
    constant = graph.call_function(torch.ops.aten.full.default, ([count], 640),
                                   dict(dtype=torch.int32, device=torch.device('cpu')))
    constant.meta['val'] = torch.empty(count, dtype=torch.int32)
    result = graph.call_function(torch.ops.aten.add.Tensor, (x, constant))
    graph.output(result)
    module = GraphModule(torch.nn.Module(), graph)
    inputs = [torch.arange(count, dtype=torch.int32) + p for p in (-1, 255, 16384, 524287)]
    expected = [module(x) for x in inputs]
    assert retain_static_factories(module, device_type='cpu')['replaced_factories'] == 1
    for value, reference in zip(inputs, expected):
        torch.testing.assert_close(module(value), reference, rtol=0, atol=0)
    assert retain_static_factories(module, device_type='cpu')['replaced_factories'] == 0


@pytest.mark.parametrize('mutated_view', [False, True])
def test_factory_with_mutated_result_or_view_is_not_retained(mutated_view):
    graph = Graph()
    constant = graph.call_function(torch.ops.aten.zeros.default, ([6],), dict(dtype=torch.int32))
    constant.meta['val'] = torch.empty(6, dtype=torch.int32)
    destination = (graph.call_function(torch.ops.aten.view.default, (constant, [2, 3]))
                   if mutated_view else constant)
    mutated = graph.call_function(torch.ops.aten.add_.Scalar, (destination, 1))
    graph.output(mutated)
    module = GraphModule(torch.nn.Module(), graph)
    assert retain_static_factories(module, device_type='cpu')['replaced_factories'] == 0
    torch.testing.assert_close(module(), torch.ones((2, 3) if mutated_view else (6,), dtype=torch.int32))


def test_static_factory_rejects_dynamic_fill_uninitialized_and_large_values():
    graph = Graph()
    fill = graph.placeholder('fill')
    dynamic = graph.call_function(torch.ops.aten.full.default, ([2], fill), dict(dtype=torch.int32))
    dynamic.meta['val'] = torch.empty(2, dtype=torch.int32)
    empty = graph.call_function(torch.ops.aten.empty.memory_format, ([2],), dict(dtype=torch.int32))
    empty.meta['val'] = torch.empty(2, dtype=torch.int32)
    large = graph.call_function(torch.ops.aten.ones.default, ([8192],), dict(dtype=torch.int32))
    large.meta['val'] = torch.empty(8192, dtype=torch.int32)
    graph.output((dynamic, empty, large))
    module = GraphModule(torch.nn.Module(), graph)
    assert retain_static_factories(module, device_type='cpu')['replaced_factories'] == 0


def test_resident_factory_preserves_fake_tensor_propagation():
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    mode = FakeTensorMode()
    graph = Graph()
    x = graph.placeholder('positions')
    fake = mode.from_tensor(torch.empty(6, dtype=torch.int32))
    x.meta['val'] = fake
    constant = graph.call_function(torch.ops.aten.arange.default, (6,), dict(dtype=torch.int32))
    constant.meta['val'] = fake
    result = graph.call_function(torch.ops.aten.add.Tensor, (x, constant))
    graph.output(result)
    module = GraphModule(torch.nn.Module(), graph)
    assert retain_static_factories(module, device_type='cpu')['replaced_factories'] == 1
    output = propagate_with_resident_buffers(
        module, [fake], lambda: FakeTensorProp(module, mode).propagate_dont_convert_inputs(fake))
    assert isinstance(output, FakeTensor) and not mode.allow_non_fake_inputs
    assert all(not isinstance(value, FakeTensor) for value in module.buffers())


def make_graph(dtype, count):
    graph = Graph()
    x = graph.placeholder('positions')
    x.meta['val'] = torch.empty(count, dtype=dtype)
    mask = graph.call_function(torch.ops.aten.bitwise_and.Scalar, (x, 255))
    mask.meta['val'] = torch.empty(count, dtype=dtype)
    shift = graph.call_function(torch.ops.aten.bitwise_right_shift.Tensor_Scalar, (mask, 3))
    shift.meta['val'] = torch.empty(count, dtype=dtype)
    test = graph.call_function(torch.ops.aten.eq.Scalar, (mask, 255))
    graph.output((mask, shift, test))
    return GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize('count', [1, 2, 6])
def test_changing_positions_and_reordered_request_rows(count):
    module = make_graph(torch.int32, count)
    values = torch.tensor([-1, 0, 127, 128, 255, 1048575], dtype=torch.int32)
    references = [module(values.roll(offset)[:count]) for offset in (0, 1, 3)]
    audit = retain_integer_constants(module, device_type='cpu')
    assert audit['replaced_operands'] == 3 and audit['resident_constants'] == 2
    assert all(n.meta['placement'] == 'eager' for n in module.graph.nodes if n.op == 'get_attr')
    for offset, expected in zip((0, 1, 3), references):
        actual = module(values.roll(offset)[:count])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert retain_integer_constants(module, device_type='cpu')['replaced_operands'] == 0


def test_int64_positions_are_never_narrowed():
    module = make_graph(torch.int64, 2)
    value = torch.tensor([1 << 40, (1 << 40)+255])
    expected = module(value)
    assert retain_integer_constants(module, device_type='cpu')['replaced_operands'] == 0
    for a, b in zip(module(value), expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_dynamic_overflow_and_mutation_operands_are_not_retained():
    graph = Graph()
    x = graph.placeholder('positions')
    x.meta['val'] = torch.empty(2, dtype=torch.int32)
    other = graph.placeholder('other')
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, other))
    wide = graph.call_function(torch.ops.aten.eq.Scalar, (x, 1 << 40))
    mutation = graph.call_function(torch.ops.aten.add_.Scalar, (x, 1))
    graph.output((add, wide, mutation))
    module = GraphModule(torch.nn.Module(), graph)
    assert retain_integer_constants(module, device_type='cpu')['replaced_operands'] == 0


def test_fake_propagation_converts_resident_buffers_and_restores_mode():
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    mode = FakeTensorMode()
    module = make_graph(torch.int32, 6)
    fake = mode.from_tensor(torch.empty(6, dtype=torch.int32))
    for node in module.graph.nodes:
        if 'val' in node.meta:
            node.meta['val'] = fake
    retain_integer_constants(module, device_type='cpu')
    result = propagate_with_resident_buffers(
        module, [fake], lambda: FakeTensorProp(module, mode).propagate_dont_convert_inputs(fake))
    assert all(isinstance(value, FakeTensor) for value in result)
    assert not mode.allow_non_fake_inputs
    assert all(not isinstance(value, FakeTensor) for value in module.buffers())
    def fail():
        raise RuntimeError('metadata failed')
    with pytest.raises(RuntimeError, match='metadata failed'):
        propagate_with_resident_buffers(module, [fake], fail)
    assert not mode.allow_non_fake_inputs
