# SPDX-License-Identifier: Apache-2.0
"""The native command pass must retain collective and storage contracts."""
import operator
import weakref

import pytest
import torch

from vllm_gaudi.compilation.deepseek_v41_native_groups import eligible_collectives, _lower_collectives
from vllm_gaudi.distributed import tp2_fused_ar_norm  # noqa: F401
from vllm_gaudi.compilation.deepseek_v41_native_groups import _DirectGroupEntry, _storage
from vllm_gaudi.ops import tp2_prepared_plan as prepared


class Recipe(torch.nn.Module):
    _recipe_id = 1
    _in_to_out_dups = {}

    def forward(self, value):
        return value.clone(),


def graph(*, count=1, alias=False, reused=False, consumer=True, gather=False):
    root = torch.nn.Module()
    root.producer = Recipe()
    root.consumer = Recipe()
    if alias:
        root.producer._in_to_out_dups = {0: 0}
    ir = torch.fx.Graph()
    argument = ir.placeholder("argument")
    produced = ir.call_module("producer", (argument,))
    value = ir.call_function(operator.getitem, (produced, 0))
    value.meta["val"] = torch.empty(count, 8, 128, dtype=torch.bfloat16)
    if gather:
        collective = ir.call_function(torch.ops._c10d_functional.all_gather_into_tensor.default, (value, 4, "tp4"))
        collective.meta["val"] = torch.empty(count * 4, 8, 128, dtype=torch.bfloat16)
    else:
        collective = ir.call_function(torch.ops._c10d_functional.all_reduce_.default, (value, "sum", "tp4"))
        collective.meta["val"] = value.meta["val"]
    ready = ir.call_function(torch.ops._c10d_functional.wait_tensor.default, (collective,))
    result = ir.call_module("consumer", (ready,)) if consumer else ready
    ir.output((result, value) if reused else result)
    return torch.fx.GraphModule(root, ir)


@pytest.mark.parametrize("gather", [False, True])
def test_complete_c1_chain_keeps_shape_and_consumer_after_lowering(gather):
    gm = graph(gather=gather)
    selected = eligible_collectives(gm)
    assert len(selected) == 1
    _lower_collectives(gm, selected)
    nodes = list(gm.graph.nodes)
    consumer = next(node for node in nodes if node.op == "call_module" and node.target == "consumer")
    assert tuple(consumer.args[0].meta["val"].shape) == (4 if gather else 1, 8, 128)
    exchanged = consumer.args[0].args[0]
    assert exchanged.target == (torch.ops.vllm_gaudi.tp4_allgather_plain.default if gather
                                else torch.ops.vllm_gaudi.tp4_allreduce_plain.default)
    assert not any(node.target == torch.ops._c10d_functional.wait_tensor.default for node in nodes)
    gm.graph.lint()


@pytest.mark.parametrize("options", [dict(count=2), dict(count=6), dict(alias=True),
                                    dict(reused=True), dict(consumer=False)])
def test_unsupported_bucket_or_observable_alias_stays_ordinary(options):
    gm = graph(**options)
    original = gm.code
    assert not eligible_collectives(gm)
    assert gm.code == original


@pytest.mark.parametrize("used_by_graph", [False, True])
def test_direct_entry_keeps_mutated_staging_output_and_unused_passthrough(monkeypatch, used_by_graph):
    owner = torch.nn.Module()
    owner.generation = 1
    captured = torch.zeros(2)

    class Graph:
        def replay_fixed(self):
            if used_by_graph:
                captured.add_(1)

    class Bindings:
        def updates(self, roots):
            return roots["metadata"]["0"]

        def apply(self, source, graph):
            if used_by_graph:
                captured.copy_(source)

    graph = Graph()
    entry = _DirectGroupEntry.__new__(_DirectGroupEntry)
    entry.owner, entry.group = weakref.ref(owner), lambda: None
    entry.graph, entry.key, entry.state_fields = graph, "test", []
    entry.bindings = Bindings()
    entry.allowed_outputs = {_storage(captured)} if used_by_graph else set()
    assert entry.finish((captured,), (captured,)) is entry
    monkeypatch.setattr(prepared, "_native_graphs", {entry.key: graph})
    monkeypatch.setattr(prepared, "_tp4_direct_group_replays", 0)
    for value in (2., 5.):
        incoming = torch.full((2,), value)
        replayed, result = entry.replay((incoming,))
        assert replayed
        torch.testing.assert_close(result[0], incoming + int(used_by_graph), rtol=0, atol=0)
        if not used_by_graph:
            assert result[0] is incoming
