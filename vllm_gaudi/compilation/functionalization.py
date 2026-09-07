# SPDX-License-Identifier: Apache-2.0
"""Lower selected mutable HPU custom-op boundaries before partitioning."""

import operator
from copy import copy

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2_dense
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.pattern_matcher import CallFunctionVarArgs, is_match
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree as pytree


def copy_graph_module(source):
    graph = torch.fx.Graph()
    mapping = {}
    for node in source.graph.nodes:
        mapping[node] = graph.node_copy(node, lambda old: mapping[old])
    graph.set_codegen(copy(source.graph._codegen))
    return torch.fx.GraphModule(source, graph)


def lower_functionalized(graph_module, allowed_ops, *, reinplace=False):
    """Return a separate FX module and an audit; never mutate the caller's graph.

    Reuse Inductor's base-liveness analysis and dense decomposition. Keep all
    clones unless reinplacing is explicitly requested and metadata is complete.
    No model tensors are copied or executed during this transformation.
    """
    hop = torch.ops.higher_order.auto_functionalized_v2
    candidate = copy_graph_module(graph_module)
    selected = [
        node for node in candidate.graph.nodes
        if node.op == "call_function" and node.target is hop and node.args[0] in allowed_ops
    ]
    audit = {"nodes": [], "reinplace_requested": reinplace}
    if not selected:
        return candidate, audit

    required_nodes = set(candidate.graph.nodes) if reinplace else {
        parent for node in selected for parent in node.all_input_nodes
    }
    for node in required_nodes:
        if node.op in ("placeholder", "call_function", "call_method", "call_module", "get_attr"):
            # make_fx omits val for a None tuple element; derive it without
            # executing the op or inventing alias metadata for a tensor.
            if "val" not in node.meta and node.target is operator.getitem:
                parent = node.args[0].meta.get("val")
                if isinstance(parent, (tuple, list)) and parent[node.args[1]] is None:
                    node.meta["val"] = None
            if "val" not in node.meta:
                raise ValueError(f"Missing fake/alias metadata for {node.name}")
            for value in pytree.tree_leaves(node.meta["val"]):
                if isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor):
                    raise ValueError(f"Expected FakeTensor metadata for {node.name}")
            if reinplace and node.op == "placeholder" and isinstance(node.meta["val"], (tuple, list, dict)):
                raise ValueError("Reinplacing requires unboxed inputs with complete storage alias information")

    decisions = {}
    if reinplace:
        # Other Inductor rewrites stay on the analysis copy. Only selected HOP
        # clone decisions are transferred, never collective or arithmetic edits.
        analysis = copy_graph_module(candidate)
        reinplace_inplaceable_ops_core(analysis.graph)
        decisions = {
            node.name: tuple(node.meta["only_clone_these_tensors"])
            for node in analysis.graph.nodes
            if node.target is hop and "only_clone_these_tensors" in node.meta
        }

    pattern = CallFunctionVarArgs(hop)
    for node in selected:
        clones = decisions.get(node.name, tuple(range(len(node.kwargs["_all_bases"]))))
        audit["nodes"].append({
            "name": node.name,
            "op": str(node.args[0]),
            "bases": len(node.kwargs["_all_bases"]),
            "cloned_bases": list(clones),
        })
        match = pattern.match(node)
        if not is_match(match):
            raise ValueError(f"Could not match {node.name}")
        flat_args, spec = pytree.tree_flatten((node.args, dict(node.kwargs)))

        def decompose(*flat, _spec=spec, _clones=clones):
            args, kwargs = pytree.tree_unflatten(flat, _spec)
            return auto_functionalized_v2_dense(args[0], _clones, **kwargs)

        # The upstream decomposition disables functional-only DCE: the lowered
        # custom op returns None, but its output mutation must remain observable.
        match.replace_by_example(decompose, flat_args, run_functional_passes=False)

    candidate.graph.lint()
    candidate.recompile()
    return candidate, audit
