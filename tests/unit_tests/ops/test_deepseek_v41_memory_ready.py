# SPDX-License-Identifier: Apache-2.0
"""Cold annotations must reject a payload access before device acquisition."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / "vllm_gaudi/compilation/deepseek_v41_memory_ready.py"
_spec = importlib.util.spec_from_file_location("dsv41_memory_ready_contract", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
annotation = _module.memory_ready_annotation


class Target:
    def __init__(self, name):
        self.name = name
        self.__name__ = name.replace(".", "_")

    def __str__(self):
        return self.name

    def __call__(self, *args):
        raise AssertionError("Cold graph inspection must not execute an operator")


def graph(line=3, *, premature=False, reset=False):
    g = torch.fx.Graph()
    root = g.placeholder("root")
    peer = g.placeholder("peer")
    residual = g.placeholder("residual")
    gates = g.placeholder("gates")
    norm = g.placeholder("norm")
    enabled = g.placeholder("enabled")
    if reset:
        clear = g.call_function(Target("custom_op.private_memory_flags_zero.default"), (root,))
        g.output((peer, clear))
    else:
        if premature:
            g.call_function(torch.ops.aten.add.Tensor, (peer, peer))
        viewed = g.call_function(torch.ops.aten.view.default, (peer, [4, 1, 5120]))
        reader = g.call_function(Target("custom_op.private_memory_ready_post.default"),
                                 (viewed, residual, gates, norm, root, 1e-20, line, enabled))
        g.output(reader)
    return torch.fx.GraphModule({}, g)


@pytest.mark.parametrize("line", [0, 1, 3, 14, 39])
def test_cold_reader_has_no_communication_ordinal(line):
    result = annotation(graph(line))
    assert (result.kind, result.root, result.peer, result.line) == ("reader", 0, 1, line)
    assert result.enabled == 5


def test_reset_keeps_the_real_output_slot():
    result = annotation(graph(reset=True))
    assert (result.kind, result.root) == ("reset", 1)


def test_early_payload_read_rejected():
    with pytest.raises(ValueError, match="before"):
        annotation(graph(premature=True))


@pytest.mark.parametrize("line", [-1, True, 2.0])
def test_dynamic_or_invalid_line_rejected(line):
    with pytest.raises(ValueError, match="constant"):
        annotation(graph(line))


def test_unmarked_graph_unchanged():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    g.output(x)
    assert annotation(torch.fx.GraphModule({}, g)) is None


def test_reset_is_hoisted_before_peer_and_reader_keeps_root_input():
    import operator
    child = graph(reset=True)
    clear = next(n for n in child.graph.nodes if str(n.target) == "custom_op.private_memory_flags_zero.default")
    clear.meta['val'] = torch.zeros(40, 32, dtype=torch.int32)
    placeholders = [n for n in child.graph.nodes if n.op == 'placeholder']
    for node in placeholders:
        node.meta['val'] = torch.zeros(40, 32, dtype=torch.int32)
    # Include ordinary payload work in the same original compiled partition.
    output = next(n for n in child.graph.nodes if n.op == 'output')
    with child.graph.inserting_before(output):
        value = child.graph.call_function(torch.ops.aten.neg.default, (placeholders[1],))
        value.meta['val'] = placeholders[1].meta['val']
    output.args = ((value, clear),)
    child.recompile()
    g = torch.fx.Graph()
    inputs = [g.placeholder(f'x{i}') for i in range(6)]
    peer = g.call_function(torch.ops.aten.clone.default, (inputs[1],))
    call = g.call_module('chunk', (inputs[0], peer, *inputs[2:]))
    values = [g.call_function(operator.getitem, (call, i)) for i in range(2)]
    g.output(tuple(values))
    module = torch.fx.GraphModule({'chunk': child}, g)
    assert _module.isolate_memory_resets(module) == 1
    nodes = list(module.graph.nodes)
    resets = [n for n in nodes if n.op == 'call_module' and
              any(str(x.target) in _module._RESETS for x in module.get_submodule(n.target).graph.nodes)]
    assert len(resets) == 1 and nodes.index(resets[0]) < nodes.index(peer)
    root_entry = annotation(module.get_submodule(resets[0].target))
    assert root_entry.kind == 'reset'
    module.graph.lint()
