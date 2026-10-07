# SPDX-License-Identifier: Apache-2.0
"""Pure graph scheduling must preserve complete mHC math and peer dependencies."""
import copy
import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_gaudi.compilation.deepseek_v41_overlap import (
    deduplicate_float_casts,
    independent_mhc_nodes,
    split_mhc_consumers,
    require_candidate_operators,
)


def test_candidate_activation_is_checked_on_the_graph():
    graph = make_fx(lambda x: torch.sigmoid(x))(torch.ones(1))
    assert require_candidate_operators(graph, ['aten.sigmoid']) == {'aten.sigmoid': 1}
    with pytest.raises(RuntimeError, match='did not activate'):
        require_candidate_operators(graph, ['custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_slots'])


@torch.library.custom_op("dsv41_overlap_test::deepseek_v41_control_gemv", mutates_args=())
def control(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(value, weight)


@control.register_fake
def _(value, weight):
    return value.new_empty(value.shape[0], weight.shape[0])


@torch.library.custom_op("dsv41_overlap_test::deepseek_v41_control_batch4_f32", mutates_args=())
def control_batch4(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(value, weight)


@control_batch4.register_fake
def _(value, weight):
    return value.new_empty(value.shape[0], weight.shape[0])


@torch.library.custom_op("dsv41_overlap_test::deepseek_v41_control_prefetch_f32", mutates_args=())
def control_prefetch(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(value, weight)


@control_prefetch.register_fake
def _(value, weight):
    return value.new_empty(value.shape[0], weight.shape[0])


@torch.library.custom_op("dsv41_overlap_test::exchange", mutates_args=())
def exchange(value: torch.Tensor) -> torch.Tensor:
    return value * 2


@exchange.register_fake
def _(value):
    return torch.empty_like(value)


def _consumer(residual, weight, partial, peer):
    flat = residual.flatten(1).float()
    projected = control(flat, weight)
    scaled = projected * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
    post = torch.sigmoid(scaled[:, :4]) * 2
    mixed = residual.float() * post.unsqueeze(-1)
    value = (partial + peer).float()
    return (value.unsqueeze(1) * post.unsqueeze(-1) + mixed).to(torch.bfloat16), post


def _example():
    torch.manual_seed(3)
    return torch.randn(1, 4, 8).bfloat16(), torch.randn(24, 32), torch.randn(1, 8).bfloat16()


def _graph():
    residual, weight, partial = _example()
    child = make_fx(_consumer)(residual, weight, partial, partial.clone())
    root = torch.nn.Module()
    root.add_module("consumer0", child)
    root.add_module("consumer1", copy.deepcopy(child))
    graph = torch.fx.Graph()
    r, w, p = (graph.placeholder(name) for name in ("residual", "weight", "partial"))
    for index in range(2):
        peer = graph.call_function(torch.ops.dsv41_overlap_test.exchange.default, (p, ))
        result = graph.call_module(f"consumer{index}", (r, w, p, peer))
        r = graph.call_function(operator.getitem, (result, 0))
    graph.output(r)
    return torch.fx.GraphModule(root, graph)


def test_mhc_split_preserves_outputs_with_changing_inputs_and_multiple_exchanges():
    source = _graph()
    candidate = copy.deepcopy(source)
    audit = split_mhc_consumers(candidate, torch.ops.dsv41_overlap_test.exchange.default)
    assert len(audit) == 2
    assert all(any("rsqrt" in op for op in item["operators"]) for item in audit)
    for seed in range(8):
        torch.manual_seed(seed)
        r, w, p = _example()
        r = r + seed
        assert torch.equal(source(r, w, p).view(torch.int16), candidate(r, w, p).view(torch.int16))
    calls = [node for node in candidate.graph.nodes if node.op == "call_module"]
    assert len(calls) == 4
    for call in calls[::2]:
        assert not any(arg.target == torch.ops.dsv41_overlap_test.exchange.default for arg in call.all_input_nodes)


def test_dependent_control_cannot_be_hoisted():
    r, w, p = _example()
    child = make_fx(_consumer)(r, w, p, p)
    # The residual itself is now the communication result (e.g. next sublayer).
    assert not independent_mhc_nodes(child, [0, 3])


@torch.library.custom_op("dsv41_overlap_test::deepseek_v41_mhc_gates_f32", mutates_args=())
def gates_only(projection: torch.Tensor, scale: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(projection * scale + base)


@gates_only.register_fake
def _(projection, scale, base):
    return torch.empty_like(projection)


def _gate_consumer(projection, scale, base, peer):
    gates = gates_only(projection, scale, base)
    return (peer.float() + gates[:, :peer.shape[1]]).bfloat16(), gates


@pytest.mark.parametrize("tokens", [1, 2, 6])
def test_gate_only_recipe_runs_before_its_peer_consumer(tokens):
    projection = torch.randn(tokens, 24)
    scale, base = torch.ones(24), torch.randn(24)
    partial = torch.randn(tokens, 8).bfloat16()
    child = make_fx(_gate_consumer)(projection, scale, base, partial)
    root = torch.nn.Module()
    root.add_module("consumer", child)
    graph = torch.fx.Graph()
    p, s, b, v = (graph.placeholder(name) for name in ("projection", "scale", "base", "partial"))
    peer = graph.call_function(torch.ops.dsv41_overlap_test.exchange.default, (v,))
    result = graph.call_module("consumer", (p, s, b, peer))
    first = graph.call_function(operator.getitem, (result, 0))
    second = graph.call_function(operator.getitem, (result, 1))
    graph.output((first, second))
    source = torch.fx.GraphModule(root, graph)
    candidate = copy.deepcopy(source)
    audit = split_mhc_consumers(candidate, torch.ops.dsv41_overlap_test.exchange.default)
    assert len(audit) == 1
    assert any("mhc_gates_f32" in name for name in audit[0]["operators"])
    calls = [node for node in candidate.graph.nodes if node.op == "call_module"]
    assert len(calls) == 2
    assert not any(arg.target == torch.ops.dsv41_overlap_test.exchange.default for arg in calls[0].all_input_nodes)
    for seed in range(5):
        torch.manual_seed(seed)
        values = torch.randn_like(projection), scale, base, torch.randn_like(partial)
        assert all(torch.equal(a, b) for a, b in zip(source(*values), candidate(*values)))
    # A gate using peer-derived projection cannot bypass the collective wait.
    assert not independent_mhc_nodes(child, [0, 3])


@pytest.mark.parametrize("name", ["control_batch4", "control_prefetch"])
def test_batch_control_keeps_independent_work_before_peer_consumer(name):
    source = _graph()
    for child in (source.consumer0, source.consumer1):
        for node in child.graph.nodes:
            if node.target == torch.ops.dsv41_overlap_test.deepseek_v41_control_gemv.default:
                node.target = getattr(torch.ops.dsv41_overlap_test, f"deepseek_v41_{name}_f32").default
        child.recompile()
    candidate = copy.deepcopy(source)
    audit = split_mhc_consumers(candidate, torch.ops.dsv41_overlap_test.exchange.default)
    assert len(audit) == 2
    assert all(any(name in op for op in item["operators"]) for item in audit)
    assert torch.equal(source(*_example()), candidate(*_example()))
    assert not independent_mhc_nodes(source.consumer0, [0, 3])


def test_mutating_partition_is_not_reordered():
    candidate = _graph()
    child = candidate.consumer0
    output = next(node for node in child.graph.nodes if node.op == "output")
    residual = next(iter(child.graph.nodes))
    with child.graph.inserting_before(output):
        child.graph.call_function(torch.ops.aten.add_.Tensor, (residual, 1))
    child.recompile()
    audit = split_mhc_consumers(candidate, torch.ops.dsv41_overlap_test.exchange.default)
    assert len(audit) == 1 and audit[0]["partition"] == "consumer1"


def test_private_reinplaced_adds_move_but_peer_sum_stays_after_wait():
    source = _graph()
    for name in ("consumer0", "consumer1"):
        child = source.get_submodule(name)
        projected = next(n for n in child.graph.nodes if "control_gemv" in str(n.target))
        with child.graph.inserting_after(projected):
            adjusted = child.graph.call_function(torch.ops.aten.add_.Tensor, (projected, 0.125))
        projected.replace_all_uses_with(adjusted)
        adjusted.args = (projected, 0.125)
        partial = [n for n in child.graph.nodes if n.op == "placeholder"][2]
        peer_sum = next(n for n in child.graph.nodes
                        if n.target == torch.ops.aten.add.Tensor and partial in n.all_input_nodes)
        peer_sum.target = torch.ops.aten.add_.Tensor
        child.recompile()
        r, w, p = _example()
        source.add_module(name, make_fx(child)(r, w, p, p.clone()))
    candidate = copy.deepcopy(source)
    audit = split_mhc_consumers(candidate, torch.ops.dsv41_overlap_test.exchange.default)
    assert len(audit) == 2
    assert all(item["private_adds_restored"] == 1 for item in audit)
    r, w, p = _example()
    assert torch.equal(source(r.clone(), w, p.clone()), candidate(r.clone(), w, p.clone()))


def test_residual_and_flat_control_share_exact_float_conversion():

    def casts(residual):
        mixed = residual.float()
        flat = residual.view(1, 32).float()
        return mixed.square().sum(1), flat.square().mean(-1)

    r, _, _ = _example()
    original = make_fx(casts)(r)
    candidate = copy.deepcopy(original)
    assert deduplicate_float_casts(candidate, set(candidate.graph.nodes)) == 1
    assert sum(n.target == torch.ops.aten._to_copy.default for n in candidate.graph.nodes) == 1
    for value in (r, r * 0, r * 16):
        assert all(torch.equal(a, b) for a, b in zip(original(value), candidate(value)))


def _register_controller_variant(name):
    @torch.library.custom_op(f"dsv41_overlap_test::{name}", mutates_args=())
    def project(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, weight)

    @project.register_fake
    def _(value, weight):
        return value.new_empty(value.shape[0], weight.shape[0])

    return project


_controller_variants = [_register_controller_variant(name) for name in (
    'deepseek_v41_control_rrms_unpack', 'deepseek_v41_control_rrms_parallel',
    'deepseek_v41_control_rrms_swizzled',
    'deepseek_v41_control_mme_f32')]


@pytest.mark.parametrize('project', _controller_variants)
def test_controller_variants_keep_residual_only_overlap(project):
    value, weight, peer = torch.randn(1, 32), torch.randn(24, 32), torch.randn(1, 24)
    graph = make_fx(lambda x, w, p: project(x, w) + p)(value, weight, peer)
    selected = independent_mhc_nodes(graph, [2])
    assert any('control_' in str(node.target) for node in selected)
    assert all('aten.add' not in str(node.target) for node in selected)
    dependent = make_fx(lambda x, w, p: project(x + p, w))(value, weight, value)
    assert independent_mhc_nodes(dependent, [2]) == set()
