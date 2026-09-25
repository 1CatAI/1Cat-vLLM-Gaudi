# SPDX-License-Identifier: Apache-2.0
"""Optional scoring cannot swallow state mutation or a downstream consumer."""
import copy

import pytest
import torch
from torch.fx.passes.shape_prop import ShapeProp

from vllm_gaudi.compilation.deepseek_v41_reindex import optional_tile_bound, split_reindex_tiles

_library = torch.library.Library("custom_op", "FRAGMENT")
_tiles = []
for _name in ("custom_deepseek_v41_reindex_tile_gaudi2", "custom_deepseek_v41_reindex_tiled_gaudi2"):
    if not hasattr(torch.ops.custom_op, _name):
        _library.define(f"{_name}(Tensor q, Tensor w, Tensor cache, "
                        "Tensor pages, Tensor rows, int ratio, int ordinal) -> Tensor")
    _library.impl(_name, lambda q, w, cache, pages, rows, ratio, ordinal: q * (ordinal + 1), "CPU")
    _tiles.append(getattr(torch.ops.custom_op, _name).default)


def graph_with_state(tile):
    graph = torch.fx.Graph()
    x, state = graph.placeholder("x"), graph.placeholder("state")
    initialized = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
    first = graph.call_function(tile, (initialized, x, x, x, x, 1, 0))
    write = graph.call_function(torch.ops.aten.copy_.default, (state, initialized))
    second = graph.call_function(tile, (first, x, x, x, x, 1, 1))
    consumed = graph.call_function(torch.ops.aten.add.Tensor, (second, write))
    graph.output((consumed, state))
    child = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Populate exact output contracts as the Bridge partitioner does.
    class Values(torch.fx.Interpreter):

        def run_node(self, node):
            value = super().run_node(node)
            node.meta["val"] = value
            return value

    Values(child).run(torch.randn(2, 3), torch.zeros(2, 3))
    ShapeProp(child).propagate(torch.randn(2, 3), torch.zeros(2, 3))
    parent = torch.fx.Graph()
    a, b = parent.placeholder("a"), parent.placeholder("b")
    call = parent.call_module("child", (a, b))
    import operator
    result = parent.call_function(operator.getitem, (call, 0))
    saved = parent.call_function(operator.getitem, (call, 1))
    parent.output((result, saved))
    root = torch.nn.Module()
    root.add_module("child", child)
    return torch.fx.GraphModule(root, parent)


@pytest.mark.parametrize("tile", _tiles)
def test_partition_preserves_mutation_and_consumption(tile):
    parent = graph_with_state(tile)
    original = copy.deepcopy(parent)
    assert split_reindex_tiles(parent) == 2
    bounds = []
    for node in parent.graph.nodes:
        if node.op == "call_module":
            child = parent.get_submodule(node.target)
            bound = optional_tile_bound(child)
            if bound:
                bounds.append(bound)
                assert not any(
                    getattr(n.target, "_schema", None) and n.target._schema.is_mutable for n in child.graph.nodes)
    assert bounds == [1, 2]
    for _ in range(3):
        value = torch.randn(2, 3)
        left, right = torch.zeros_like(value), torch.zeros_like(value)
        expected, actual = original(value, left), parent(value, right)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
        assert torch.equal(left, right)


@pytest.mark.parametrize("tile", _tiles)
def test_omittable_recipe_rejects_mixed_work(tile):
    with pytest.raises(RuntimeError, match="mandatory work"):
        optional_tile_bound(graph_with_state(tile).child)


@pytest.mark.parametrize("tile", _tiles)
@pytest.mark.parametrize("ordinal", [0, 3, 8])
def test_pure_tile_preserves_short_and_whole_plan_bounds(tile, ordinal):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    result = graph.call_function(tile, (x, x, x, x, x, 1, ordinal))
    graph.output(result)
    assert optional_tile_bound(torch.fx.GraphModule(torch.nn.Module(), graph)) == ordinal + 1
