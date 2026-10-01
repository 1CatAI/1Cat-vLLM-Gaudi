# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.fx import Graph, GraphModule

from vllm_gaudi.compilation.deepseek_v41_integer_constants import (
    propagate_with_resident_buffers, retain_integer_constants)


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
