# SPDX-License-Identifier: Apache-2.0
"""Represent mutable output dependencies explicitly before HPU partitioning."""

from copy import copy

import torch


_attention_library = globals().get("_attention_library")


def attention_out_op():
    global _attention_library
    if _attention_library is None:
        library = torch.library.Library("dsv4_ordered", "FRAGMENT")
        library.define(
            "attention_out(Tensor hidden_states, Tensor positions, Tensor(a!) out, str layer_name) -> Tensor(a!)")

        def implementation(hidden_states, positions, out, layer_name):
            torch.ops.vllm.deepseek_v4_attention.default(hidden_states, positions, out, layer_name)
            return out

        def fake(hidden_states, positions, out, layer_name):
            return out

        library.impl("attention_out", implementation, "CompositeExplicitAutograd")
        torch.library.register_fake("dsv4_ordered::attention_out", fake, lib=library)
        _attention_library = library
    return torch.ops.dsv4_ordered.attention_out.default


def make_output_dependency(graph_module, op, ordered_op, out_index):
    """Give post-write consumers a producer edge instead of only shared storage.

    This pass runs after AOT functionalization and before HPU partitioning. The
    adapter keeps the mutation and returns the exact same output tensor. It is
    not an independently functional custom op for Dynamo/AOT input graphs.
    """
    graph = graph_module.graph
    positions = {node: index for index, node in enumerate(graph.nodes)}
    replaced = 0
    for node in list(graph.nodes):
        if node.target is not op:
            continue
        if node.users:
            raise ValueError("Expected a mutable op returning None without live return users")
        out = node.args[out_index]
        if not isinstance(out, torch.fx.Node) or not isinstance(out.meta.get("val"), torch.Tensor):
            raise ValueError("Expected an output tensor with fake metadata")
        storage = out.meta["val"].untyped_storage()._cdata
        for alias, position in positions.items():
            value = alias.meta.get("val")
            if alias is out or position >= positions[node] or not isinstance(value, torch.Tensor):
                continue
            if value.untyped_storage()._cdata == storage and any(
                positions[user] > positions[node] for user in alias.users
            ):
                raise ValueError("A pre-existing output alias needs its own post-write dependency")
        later_users = [user for user in out.users if positions[user] > positions[node]]
        with graph.inserting_before(node):
            ordered = graph.call_function(ordered_op, node.args, node.kwargs)
            ordered.meta = copy(node.meta)
            # The original op returns None, whereas the adapter returns an HPU
            # tensor. Let propagation recompute the backend's return metadata.
            for key in ("output_device", "output_dtypes", "output_layouts", "output_shapes", "output_strides",
                        "output_contiguous", "output_offset", "tensor_meta"):
                ordered.meta.pop(key, None)
            ordered.meta["val"] = out.meta["val"]
        for user in later_users:
            user.replace_input_with(out, ordered)
        graph.erase_node(node)
        replaced += 1
    graph.lint()
    graph_module.recompile()
    return replaced
