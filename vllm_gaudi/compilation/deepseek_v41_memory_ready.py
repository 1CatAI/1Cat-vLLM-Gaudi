# SPDX-License-Identifier: Apache-2.0
"""Cold first-reader annotations for native, device-acquiring recipe plans."""
from dataclasses import dataclass

import operator

import torch
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module

from vllm_gaudi.compilation.deepseek_v41_overlap import _partition_metadata


@dataclass(frozen=True)
class MemoryReadyAnnotation:
    kind: str
    root: int
    peer: int = -1
    line: int = -1
    enabled: int = -1


_READERS = {
    "custom_op.private_memory_ready_post.default",
    "custom_op.custom_deepseek_v41_mhc_post_norm_memory_ready_gaudi2.default",
}
_RESETS = {"custom_op.private_memory_flags_zero.default", "custom_op.custom_deepseek_v41_memory_flags_zero_gaudi2.default"}
_VIEWS = {"aten.view.default", "aten.reshape.default", "aten.alias.default"}


def _origin(node):
    while isinstance(node, torch.fx.Node) and node.op == "call_function" and str(node.target) in _VIEWS:
        node = node.args[0]
    return node


def memory_ready_annotation(module):
    """Require a real root binding and gate every earlier payload read.

    The runtime independently verifies the arriving buffer, reset producer,
    fixed addresses and retirement. This annotation supplies no timer, event,
    tensor copy, communication ordinal or per-token host work.
    """
    nodes = list(module.graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    marked = [n for n in nodes if n.op == "call_function" and str(n.target) in _READERS | _RESETS]
    if not marked:
        return None
    if len(marked) != 1:
        raise ValueError("An acquiring recipe must have exactly one reader or reset: " + str([(str(n.target), n.args[6] if len(n.args) > 6 else None) for n in marked]))
    node = marked[0]
    if str(node.target) in _RESETS:
        result = next(n for n in nodes if n.op == "output").args[0]
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        roots = [i for i, value in enumerate(outputs) if _origin(value) is node]
        if len(roots) != 1:
            raise ValueError("The reset must expose exactly one tracked root output")
        return MemoryReadyAnnotation("reset", roots[0])
    if len(node.args) < 8 or type(node.args[6]) is not int or node.args[6] < 0:
        raise ValueError("The flag line must be a nonnegative cold constant")
    peer, root = _origin(node.args[0]), _origin(node.args[4])
    enabled = _origin(node.args[7])
    if (peer not in placeholders or root not in placeholders or enabled not in placeholders
            or len({peer, root, enabled}) != 3):
        raise ValueError("The reader must bind separate payload, flag-root and admission inputs")
    tainted = {peer}
    for earlier in nodes[:nodes.index(node)]:
        if any(argument in tainted for argument in earlier.all_input_nodes):
            if earlier.op != "call_function" or str(earlier.target) not in _VIEWS:
                raise ValueError("The payload is read before its acquiring kernel")
            tainted.add(earlier)
    return MemoryReadyAnnotation("reader", placeholders.index(root), placeholders.index(peer), node.args[6],
                                 placeholders.index(enabled))


def mark_memory_ready_recipe(native, module, arguments, outputs):
    annotation = memory_ready_annotation(module)
    if annotation is None:
        return
    if len([n for n in module.graph.nodes if n.op == "placeholder"]) != len(arguments):
        raise ValueError("Prepared reader bindings differ from its compiled placeholders")
    if annotation.kind == "reset":
        if not hasattr(native, "mark_last_memory_ready_clear"):
            raise RuntimeError("Native runtime lacks the reset-producer contract")
        native.mark_last_memory_ready_clear(outputs[annotation.root])
    else:
        if not hasattr(native, "mark_last_memory_ready"):
            raise RuntimeError("Native runtime lacks the acquiring-reader contract")
        native.mark_last_memory_ready(arguments[annotation.root], arguments[annotation.peer], annotation.line,
                                      arguments[annotation.enabled])


def isolate_memory_resets(module):
    graph = module.graph
    changed = 0
    for call in list(graph.nodes):
        if call.op != "call_module" or call.kwargs:
            continue
        child = module.get_submodule(call.target)
        if not isinstance(child, torch.fx.GraphModule):
            continue
        tiles = [node for node in child.graph.nodes if str(node.target) in _RESETS]
        if not tiles:
            continue
        assignment, current = {}, 0
        for node in child.graph.nodes:
            if str(node.target) in _RESETS:
                current += 1
                assignment[node] = current
                current += 1
            else:
                assignment[node] = current
        wrapper = split_module(child, child, assignment.__getitem__)
        env = dict(zip((n for n in wrapper.graph.nodes if n.op == "placeholder"), call.args, strict=True))
        with graph.inserting_before(call):
            for node in wrapper.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    result = map_arg(node.args[0], env.__getitem__)
                    if isinstance(result, torch.fx.Node):
                        call.replace_all_uses_with(result)
                    else:
                        for user in list(call.users):
                            if user.op != "call_function" or user.target != operator.getitem:
                                raise RuntimeError("Memory-ready split requires explicit tuple consumers")
                            user.replace_all_uses_with(result[user.args[1]])
                            graph.erase_node(user)
                    continue
                if node.op == "call_module":
                    target = f"{call.target}_memory_reset_{node.target}"
                    partition = wrapper.get_submodule(node.target)
                    bound = any(str(n.target) in _RESETS for n in partition.graph.nodes)
                    changed += bool(bound)
                    module.add_submodule(target, partition)
                    copied = graph.call_module(target, map_arg(node.args, env.__getitem__),
                                               map_arg(node.kwargs, env.__getitem__))
                    copied.meta = {
                        key: value
                        for key, value in call.meta.items()
                        if not key.startswith("output_") and key not in ("val", "tensor_meta")
                    }
                    copied.meta.update(_partition_metadata(partition))
                    if bound:
                        inputs = list(copied.all_input_nodes)
                        order = {n: i for i, n in enumerate(graph.nodes)}
                        if not inputs:
                            raise RuntimeError("Memory reset requires a tracked template input")
                        last = max(inputs, key=order.__getitem__)
                        last.append(copied)
                elif node.op == "get_attr":
                    raise RuntimeError("Unexpected captured attribute in memory reset partition")
                else:
                    copied = graph.node_copy(node, env.__getitem__)
                    if node.target != operator.getitem or "_mhc_result_meta" not in copied.args[0].meta:
                        raise RuntimeError("Unexpected Memory-ready wrapper operation")
                    copied.meta = dict(copied.args[0].meta["_mhc_result_meta"][copied.args[1]])
                    copied.meta["placement"] = "eager"
                env[node] = copied
        graph.erase_node(call)
    if changed:
        graph.lint()
        module.recompile()
    return changed
