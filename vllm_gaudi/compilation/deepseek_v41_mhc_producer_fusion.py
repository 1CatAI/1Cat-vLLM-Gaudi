# SPDX-License-Identifier: Apache-2.0
"""Move an independent mHC controller into the preceding compute recipe.

The ordinary overlap pass has already separated the pure residual-only branch
from the peer-dependent consumer. Keep that separation, but let the previous
producer's MME work overlap the controller inside one recipe. Communication and
the consumer remain outside; no arithmetic or reduction order is changed.
"""

import operator

import torch

from vllm_gaudi.compilation.deepseek_v41_overlap import (
    _partition_metadata,
    _pure,
    crosses_mutable_storage,
)


def _getitem(node):
    return node.op == "call_function" and node.target == operator.getitem


def _outputs(module):
    value = next(node for node in module.graph.nodes if node.op == "output").args[0]
    if isinstance(value, torch.fx.Node):
        return (value, ), False
    if isinstance(value, tuple) and all(isinstance(item, torch.fx.Node) for item in value):
        return value, True
    raise ValueError("non-tensor partition output")


def _tuple_users(call, count):
    return all(_getitem(user) and isinstance(user.args[1], int) and 0 <= user.args[1] < count for user in call.users)


def _plan(module, call, exchanges):
    nodes = list(module.graph.nodes)
    before = nodes[:nodes.index(call)]
    producer_call = next((node for node in reversed(before) if node.op == "call_module"), None)
    if producer_call is None or producer_call.kwargs:
        raise ValueError("no preceding positional compute producer")
    between = before[before.index(producer_call) + 1:]
    if not any(node.op == "call_function" and node.target in exchanges for node in between):
        raise ValueError("no intervening peer point")
    unsafe = [
        node for node in between
        if not (_getitem(node) or _pure(node) or (node.op == "call_function" and node.target in exchanges))
    ]
    if unsafe:
        raise ValueError("intervening operations lack a pure storage contract: " + str([(node.op, str(node.target))
                                                                                        for node in unsafe]))
    if sum(node.op == "call_module" and node.target == producer_call.target for node in nodes) != 1:
        raise ValueError("producer module is shared by multiple calls")
    producer = module.get_submodule(producer_call.target)
    control = module.get_submodule(call.target)
    if not isinstance(producer, torch.fx.GraphModule):
        raise ValueError("producer is not an FX partition")
    if any(node.op not in ("placeholder", "get_attr", "output")
           and not (node.op == "call_function" and (getattr(node.target, "_schema", None) or _getitem(node)))
           for node in producer.graph.nodes):
        raise ValueError("producer has an operation without a storage contract")
    if any(node.op not in ("placeholder", "output") and not (_pure(node) or _getitem(node)) or node.op == "get_attr"
           for node in control.graph.nodes):
        raise ValueError("controller has state, mutation, or an unknown operation")
    old_outputs, old_tuple = _outputs(producer)
    control_outputs, control_tuple = _outputs(control)
    if old_tuple and not _tuple_users(producer_call, len(old_outputs)):
        raise ValueError("producer tuple escapes")
    if control_tuple and not _tuple_users(call, len(control_outputs)):
        raise ValueError("controller tuple escapes")

    graph, copied = torch.fx.Graph(), {}
    for node in producer.graph.nodes:
        if node.op == "output":
            continue
        copied[node] = graph.node_copy(node, copied.__getitem__)
    env = {}
    producer_placeholders = [node for node in producer.graph.nodes if node.op == "placeholder"]
    if len(producer_placeholders) != len(producer_call.args):
        raise ValueError("producer argument count differs from its graph")
    if not producer_placeholders:
        raise ValueError("constant-only producer has no activation handoff")
    existing = {
        arg: copied[node]
        for node, arg in zip(producer_placeholders, producer_call.args) if isinstance(arg, torch.fx.Node)
    }
    placeholders = [node for node in control.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(call.args):
        raise ValueError("controller argument count differs from its graph")
    parent_args = list(producer_call.args)
    lifted_attributes = []
    # Bridge fused partitions may interleave placeholders and operators. The
    # callable ABI follows placeholder order, not the first compute position.
    # New parent arguments are appended, so their placeholders must be too.
    last_placeholder = copied[producer_placeholders[-1]]
    available = set(before[:before.index(producer_call)])
    for placeholder, argument in zip(placeholders, call.args):
        projection = (isinstance(argument, torch.fx.Node) and _getitem(argument) and argument.args[0] is producer_call
                      and old_tuple)
        if projection:
            env[placeholder] = copied[old_outputs[argument.args[1]]]
        elif argument is producer_call and not old_tuple:
            env[placeholder] = copied[old_outputs[0]]
        elif isinstance(argument, torch.fx.Node) and argument in existing:
            env[placeholder] = existing[argument]
        else:
            if not isinstance(argument, (torch.fx.Node, int, float, bool, type(None))):
                raise ValueError("controller input is not a tensor node or scalar literal")
            if isinstance(argument, torch.fx.Node) and argument not in available:
                if argument.op == "get_attr":
                    # Moving an attribute lookup does not copy its tensor.
                    # All intervening operations are read-only; storage writes
                    # inside the producer are checked on the merged graph.
                    lifted_attributes.append(argument)
                else:
                    raise ValueError("controller input is not ready at producer entry: " + str(argument.target))
            with graph.inserting_after(last_placeholder):
                added = graph.placeholder(f"mhc_{placeholder.name}")
                added.meta = dict(placeholder.meta)
                if isinstance(argument, torch.fx.Node):
                    added.meta.update(argument.meta)
            last_placeholder = added
            env[placeholder] = added
            parent_args.append(argument)
            if isinstance(argument, torch.fx.Node):
                existing[argument] = added
    selected = set()
    for node in control.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        env[node] = graph.node_copy(node, env.__getitem__)
        selected.add(env[node])
    graph.output(tuple(copied[node] for node in old_outputs) + tuple(env[node] for node in control_outputs))
    merged = torch.fx.GraphModule(producer, graph)
    if crosses_mutable_storage(merged, selected):
        raise ValueError("producer writes controller input storage or alias evidence is missing")
    graph.lint()
    return producer_call, merged, tuple(parent_args), len(old_outputs), old_tuple, control_tuple, lifted_attributes


def fuse_mhc_producers(module, exchanges):
    """Return explicit fused/skipped entries; unsupported partitions stay intact."""
    exchanges = exchanges if isinstance(exchanges, tuple) else (exchanges, )
    graph, audit = module.graph, []
    for call in list(graph.nodes):
        if call.op != "call_module" or call.kwargs or "_mhc_submod_" not in str(call.target):
            continue
        child = module.get_submodule(call.target)
        if not isinstance(child, torch.fx.GraphModule):
            continue
        if not any(node.op == "call_function" and ("deepseek_v41_control_rrms_parallel" in str(node.target)
                                                   or "deepseek_v41_control_rrms_swizzled" in str(node.target))
                   for node in child.graph.nodes):
            continue
        try:
            producer, merged, args, offset, old_tuple, control_tuple, lifted = _plan(module, call, exchanges)
        except ValueError as error:
            audit.append(dict(controller=call.target, fused=False, reason=str(error)))
            continue
        old_users = list(producer.users)
        for attribute in lifted:
            producer.prepend(attribute)
        module.set_submodule(producer.target, merged)
        producer.args = args
        producer.meta = {
            key: value
            for key, value in producer.meta.items()
            if not key.startswith("output_") and key not in ("val", "tensor_meta", "_mhc_result_meta")
        }
        producer.meta.update(_partition_metadata(merged))
        if not old_tuple:
            with graph.inserting_after(producer):
                original = graph.call_function(operator.getitem, (producer, 0))
                original.meta = dict(producer.meta["_mhc_result_meta"][0], placement="eager")
            for user in old_users:
                user.replace_input_with(producer, original)
        with graph.inserting_after(producer):
            if control_tuple:
                for user in list(call.users):
                    index = offset + user.args[1]
                    result = graph.call_function(operator.getitem, (producer, index))
                    result.meta = dict(producer.meta["_mhc_result_meta"][index], placement="eager")
                    user.replace_all_uses_with(result)
                    graph.erase_node(user)
            else:
                result = graph.call_function(operator.getitem, (producer, offset))
                result.meta = dict(producer.meta["_mhc_result_meta"][offset], placement="eager")
                call.replace_all_uses_with(result)
        graph.erase_node(call)
        audit.append(
            dict(controller=call.target,
                 producer=producer.target,
                 fused=True,
                 original_outputs=offset,
                 appended_outputs=len(producer.meta["_mhc_result_meta"]) - offset))
    if any(item["fused"] for item in audit):
        graph.lint()
        module.recompile()
    return audit
