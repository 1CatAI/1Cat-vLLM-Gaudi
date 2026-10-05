# SPDX-License-Identifier: Apache-2.0
"""Scheduling preserves arithmetic, producer users, and mutable-state ordering."""
import copy
import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_gaudi.compilation.deepseek_v41_mhc_producer_fusion import fuse_mhc_producers
from vllm_gaudi.compilation.deepseek_v41_overlap import split_mhc_consumers


@torch.library.custom_op("dsv41_producer_test::deepseek_v41_control_rrms_parallel", mutates_args=())
def control(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(value, weight)


@control.register_fake
def _(value, weight):
    return value.new_empty(value.shape[0], weight.shape[0])


@torch.library.custom_op("dsv41_producer_test::peer", mutates_args=())
def peer(value: torch.Tensor) -> torch.Tensor:
    return value * 2


@peer.register_fake
def _(value):
    return torch.empty_like(value)


def _fixture():
    generator = torch.Generator().manual_seed(29)
    return torch.randn(1, 8, generator=generator), torch.randn(8, 8, generator=generator)


def _graph(*, tuple_producer=True, tuple_control=True, internal_residual=False, mutate=False):
    x, weight = _fixture()

    def produce(value):
        result = value * 3
        if mutate:
            value.add_(1)
        return (result, value.clone()) if tuple_producer else result

    producer = make_fx(produce)(x.clone())

    def project(value, w):
        result = control(value, w)
        return (result, value * .25) if tuple_control else result

    controller = make_fx(project)(x, weight)
    root, graph = torch.nn.Module(), torch.fx.Graph()
    root.add_module("producer", producer)
    root.add_module("consumer_mhc_submod_0", controller)
    value, w = graph.placeholder("value"), graph.placeholder("weight")
    value.meta["val"] = x
    w.meta["val"] = weight
    p = graph.call_module("producer", (value, ))
    partial = graph.call_function(operator.getitem, (p, 0)) if tuple_producer else p
    residual = graph.call_function(operator.getitem, (p, 1)) if internal_residual else value
    exchanged = graph.call_function(torch.ops.dsv41_producer_test.peer.default, (partial, ))
    c = graph.call_module("consumer_mhc_submod_0", (residual, w))
    projected = graph.call_function(operator.getitem, (c, 0)) if tuple_control else c
    result = graph.call_function(torch.ops.aten.add.Tensor, (exchanged, projected))
    graph.output((result, partial, p))
    # Producer tuple must be projected explicitly. Keep both values observable.
    if tuple_producer:
        output = next(node for node in graph.nodes if node.op == "output")
        with graph.inserting_before(output):
            extra = graph.call_function(operator.getitem, (p, 1))
        output.args = ((result, partial, extra), )
    return torch.fx.GraphModule(root, graph)


@pytest.mark.parametrize("tuple_producer,tuple_control,internal_residual", [(True, True, False), (True, True, True),
                                                                            (True, False, False), (False, True, False),
                                                                            (False, False, False)])
def test_fusion_preserves_five_inputs_and_earlier_producer_users(tuple_producer, tuple_control, internal_residual):
    source = _graph(tuple_producer=tuple_producer, tuple_control=tuple_control, internal_residual=internal_residual)
    candidate = copy.deepcopy(source)
    audit = fuse_mhc_producers(candidate, torch.ops.dsv41_producer_test.peer.default)
    assert len(audit) == 1 and audit[0]["fused"]
    calls = [node for node in candidate.graph.nodes if node.op == "call_module"]
    assert len(calls) == 1
    assert len(calls[0].args) == 2  # Weight now belongs to the producer.
    for index in range(5):
        value, weight = _fixture()
        args = (value + index * .3, weight - index * .1)
        expected, observed = source(*args), candidate(*args)
        assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(expected, observed))
    assert fuse_mhc_producers(candidate, torch.ops.dsv41_producer_test.peer.default) == []


@pytest.mark.parametrize("reason", ["late", "no_peer", "parent_mutation", "child_mutation", "escape", "alias"])
def test_unsafe_motion_stays_unchanged(reason):
    source = _graph(mutate=reason == "alias")
    nodes = list(source.graph.nodes)
    c = next(node for node in nodes if node.op == "call_module" and "_mhc_" in node.target)
    p = next(node for node in nodes if node.op == "call_module" and node.target == "producer")
    exchanged = next(node for node in nodes if node.target == torch.ops.dsv41_producer_test.peer.default)
    if reason == "late":
        c.args = (exchanged, c.args[1])
    elif reason == "no_peer":
        exchanged.target = torch.ops.aten.neg.default
    elif reason == "parent_mutation":
        with source.graph.inserting_before(c):
            source.graph.call_function(torch.ops.aten.add_.Tensor, (c.args[0], 1))
    elif reason == "child_mutation":
        child = source.consumer_mhc_submod_0
        output = next(node for node in child.graph.nodes if node.op == "output")
        with child.graph.inserting_before(output):
            child.graph.call_function(torch.ops.aten.add_.Tensor, (next(iter(child.graph.nodes)), 1))
        child.recompile()
    elif reason == "escape":
        output = next(node for node in source.graph.nodes if node.op == "output")
        output.args = ((p, c), )
    source.recompile()
    before = source.code
    audit = fuse_mhc_producers(source, torch.ops.dsv41_producer_test.peer.default)
    assert len(audit) == 1 and not audit[0]["fused"], audit
    assert source.code == before


def test_other_controllers_are_unchanged():
    source = _graph()
    for node in source.consumer_mhc_submod_0.graph.nodes:
        if "control_rrms_parallel" in str(node.target):
            node.target = torch.ops.aten.mm.default
    source.consumer_mhc_submod_0.recompile()
    assert fuse_mhc_producers(source, torch.ops.dsv41_producer_test.peer.default) == []


def test_output_metadata_is_preserved_without_post_partition_fake_propagation():
    source = _graph()
    old_output = next(node for node in source.producer.graph.nodes if node.op == "output").args[0][0]
    old_output.meta["output_offset"] = [7]
    audit = fuse_mhc_producers(source, torch.ops.dsv41_producer_test.peer.default)
    assert audit[0]["fused"]
    call = next(node for node in source.graph.nodes if node.op == "call_module")
    assert call.meta["output_offset"][0] == 7
    assert len(call.meta["_mhc_result_meta"]) == 4


def test_follows_real_split_across_multiple_peer_boundaries():
    x, weight = _fixture()
    root, graph = torch.nn.Module(), torch.fx.Graph()
    value, w = graph.placeholder("value"), graph.placeholder("weight")
    for index in range(3):
        root.add_module(f"producer{index}", make_fx(lambda r: (r * 3, ))(x))
        root.add_module(f"consumer{index}", make_fx(lambda r, w, p: (control(r, w) + p, ))(x, weight, x.clone()))
        p = graph.call_module(f"producer{index}", (value, ))
        partial = graph.call_function(operator.getitem, (p, 0))
        exchanged = graph.call_function(torch.ops.dsv41_producer_test.peer.default, (partial, ))
        c = graph.call_module(f"consumer{index}", (value, w, exchanged))
        value = graph.call_function(operator.getitem, (c, 0))
    graph.output(value)
    source = torch.fx.GraphModule(root, graph)
    candidate = copy.deepcopy(source)
    assert len(split_mhc_consumers(candidate, torch.ops.dsv41_producer_test.peer.default)) == 3
    audit = fuse_mhc_producers(candidate, torch.ops.dsv41_producer_test.peer.default)
    assert len(audit) == 3 and all(item["fused"] for item in audit), audit
    assert sum(node.op == "call_module" for node in candidate.graph.nodes) == 6
    assert sum(node.target == torch.ops.dsv41_producer_test.peer.default for node in candidate.graph.nodes) == 3
    for index in range(5):
        assert torch.equal(source(x + index, weight), candidate(x + index, weight))


def test_resident_constants_and_readonly_views_between_peer_and_control():
    source = _graph()
    source.register_buffer("resident_weight", _fixture()[1])
    call = next(node for node in source.graph.nodes if node.op == "call_module" and "_mhc_" in node.target)
    with source.graph.inserting_before(call):
        weight = source.graph.get_attr("resident_weight")
        weight.meta["val"] = source.resident_weight
        source.graph.call_function(torch.ops.aten.view.default, (call.args[0], [8]))
    call.args = (call.args[0], weight)
    source.recompile()
    candidate = copy.deepcopy(source)
    audit = fuse_mhc_producers(candidate, torch.ops.dsv41_producer_test.peer.default)
    assert len(audit) == 1 and audit[0]["fused"], audit
    for index in range(5):
        value, weight = _fixture()
        assert all(torch.equal(a, b) for a, b in zip(source(value + index, weight), candidate(value + index, weight)))


def test_bridge_interleaved_placeholders_keep_parent_argument_order():
    source = _graph()
    child = source.producer
    first_compute = next(node for node in child.graph.nodes if node.op == "call_function")
    with child.graph.inserting_after(first_compute):
        gain = child.graph.placeholder("gain")
    with child.graph.inserting_after(gain):
        scaled = child.graph.call_function(torch.ops.aten.mul.Tensor, (first_compute, gain))
    for user in list(first_compute.users):
        if user is not scaled:
            user.replace_input_with(first_compute, scaled)
    child.recompile()
    source.register_buffer("gain", torch.tensor([1.5]))
    producer = next(node for node in source.graph.nodes if node.op == "call_module" and node.target == "producer")
    with source.graph.inserting_before(producer):
        resident_gain = source.graph.get_attr("gain")
        resident_gain.meta["val"] = source.gain
    producer.args = (*producer.args, resident_gain)
    source.recompile()
    candidate = copy.deepcopy(source)
    assert fuse_mhc_producers(candidate, torch.ops.dsv41_producer_test.peer.default)[0]["fused"]
    placeholder_names = [node.target for node in candidate.producer.graph.nodes if node.op == "placeholder"]
    assert placeholder_names[1] == "gain"
    assert placeholder_names[-1].startswith("mhc_")
    for index in range(5):
        value, weight = _fixture()
        assert all(torch.equal(a, b) for a, b in zip(source(value + index, weight), candidate(value + index, weight)))
