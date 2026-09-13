# SPDX-License-Identifier: Apache-2.0
"""Destination-bound GDN update lowering through the normal Bridge pass API."""

from __future__ import annotations

import threading
import os

import torch
from torch.fx import Node

_registered = False
_registration_lock = threading.Lock()
_stats = {"graphs": 0, "updates": 0}
_audit_serial = 0


def state_update_lowering_stats():
    return dict(_stats)


def _tensor(node):
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val", node.meta.get("tensor_meta"))
    return value if isinstance(value, torch.Tensor) else None


def _range(node):
    value = _tensor(node)
    if value is None or not value.is_contiguous():
        return None
    layout = (*value.shape, *value.stride(), value.storage_offset())
    if any(type(dim) is not int for dim in layout):
        return None
    width = value.element_size()
    return (value.untyped_storage()._cdata, value.storage_offset() * width,
            (value.storage_offset() + value.numel()) * width)


def _overlaps(left, right):
    x, y = _tensor(left), _tensor(right)
    if x is not None and y is not None and x.untyped_storage()._cdata != y.untyped_storage()._cdata:
        return False
    a, b = _range(left), _range(right)
    # Unknown aliases cannot establish a safe destination-bound write.
    return a is None or b is None or (a[0] == b[0] and a[1] < b[2] and b[1] < a[2])


def _views():
    return {
        torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default, torch.ops.aten.reshape.default,
        torch.ops.aten.alias.default
    }


def _unwrap(node):
    while isinstance(node, Node) and node.op == "call_function" and node.target in _views():
        node = node.args[0]
    return node


def _mutation_source(node, destination):
    copies = []
    node = _unwrap(node)
    while isinstance(node, Node) and node.target == torch.ops.aten.copy.default:
        if (_range(node.args[0]) != _range(destination) or _tensor(node) is None
                or _tensor(node).dtype != _tensor(destination).dtype
                or tuple(_tensor(node).shape) != tuple(_tensor(destination).shape)):
            break
        copies.append(node)
        node = _unwrap(node.args[1])
    return node, copies


def lower_gdn_state_updates(graph_module):
    """Consume an explicit mutation epilogue after proving state liveness.

    A functional marker lets AOT describe the mutation normally. This pass
    runs after functionalization and before partitioning. It only accepts
    a complete active row, exact destination aliasing, and no old-state read
    between the update and its original copy. It never assumes disjointness
    merely because tensors have different Python identities.
    """
    functional = torch.ops.custom_op.gdn_state_update.default
    out = torch.ops.custom_op.gdn_state_update_out.default
    graph = graph_module.graph
    nodes = list(graph.nodes)
    positions = {node: index for index, node in enumerate(nodes)}
    changed = 0
    for node in nodes:
        if node.op != "call_function" or node.target != functional:
            continue
        destination = node.args[3]
        dst = _tensor(destination)
        if dst is None or tuple(dst.shape) != (1, 24, 128, 128) or _range(destination) is None:
            raise RuntimeError("GDN direct state lowering requires a concrete contiguous active row")
        copies = [
            candidate for candidate in nodes
            if candidate.op == "call_function" and candidate.target == torch.ops.aten.copy_.default
            and _mutation_source(candidate.args[1], destination)[0] is node
            and _range(candidate.args[0]) == _range(destination)
        ]
        if len(copies) != 1:
            raise RuntimeError(f"GDN direct state update requires one explicit mutation epilogue; found {len(copies)}")
        copy = copies[0]
        _, functional_copies = _mutation_source(copy.args[1], destination)
        if positions[copy] <= positions[node]:
            raise RuntimeError("GDN state mutation precedes its computation")
        if any(_overlaps(destination, source) for source in node.args[:3]):
            raise RuntimeError("GDN state destination overlaps an epilogue input")
        for candidate in nodes[positions[node] + 1:positions[copy]]:
            if candidate in functional_copies:
                continue
            if candidate.op == "call_function" and candidate.target in _views():
                continue
            for argument in candidate.all_input_nodes:
                if _tensor(argument) is not None and _overlaps(destination, argument):
                    raise RuntimeError(f"GDN old state remains live at {candidate.name}")
        node.target = out
        node.meta["val"] = dst
        node.meta["placement"] = "hpu_cluster"
        for functional_copy in functional_copies:
            functional_copy.replace_all_uses_with(node)
            graph.erase_node(functional_copy)
        copy.replace_all_uses_with(node)
        graph.erase_node(copy)
        changed += 1
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
        _stats["graphs"] += 1
        _stats["updates"] += changed
    return bool(changed)


def validate_direct_state_views(views):
    """Check scheduler-owned ranges only when the active binding changes."""
    ranges = []
    for view in views:
        if view is None or tuple(view.shape) != (1, 24, 128, 128) or not view.is_contiguous():
            raise RuntimeError("Every direct GDN layer must own one contiguous active TP2 row")
        if view.dtype != torch.float32:
            raise RuntimeError("Direct GDN state must retain FP32 storage")
        # The Python data_ptr facade used by NIXL can expose an allocation's
        # base for a view. Compare the actual storage interval, including its
        # offset, so distinct layers in a shared pool remain distinct.
        begin = view.untyped_storage().data_ptr() + view.storage_offset() * view.element_size()
        end = begin + view.numel() * view.element_size()
        if any(begin < stop and start < end for start, stop in ranges):
            raise RuntimeError("Direct GDN state destinations overlap between layers")
        ranges.append((begin, end))
    return len(ranges)


def register_gdn_state_update_pass(graph_directory=None):
    global _registered
    with _registration_lock:
        if _registered:
            return
        from habana_frameworks.torch.dynamo.compile_backend.passes import (
            OptimizationPassPlacement,
            register_pass_at_optimization_pass,
        )
        graph_directory = graph_directory or os.environ.get("GDN_STATE_DIAGNOSTIC_DIR")
        if graph_directory is not None:
            from pathlib import Path

            graph_directory = Path(graph_directory) / f"rank{os.environ.get('LOCAL_RANK', '0')}"

        def pass_gdn_state_update(context):
            global _audit_serial
            graph = context.graph_module
            if graph_directory is not None:
                from pathlib import Path

                directory = Path(graph_directory)
                directory.mkdir(parents=True, exist_ok=True)
                index = _audit_serial
                _audit_serial += 1
                (directory / f"before-{index}.py").write_text(graph.code)
            changed = lower_gdn_state_updates(graph)
            if graph_directory is not None:
                (directory / f"after-{index}.py").write_text(graph.code)
            return changed

        register_pass_at_optimization_pass(pass_gdn_state_update, OptimizationPassPlacement.PRE_PARTITIONER)
        if graph_directory is not None:

            def dump_state_partitions(context):
                import json

                graph = context.graph_module
                index = len(list(graph_directory.glob("partitions-*.json")))
                rows = []
                for name, module in graph.named_modules():
                    row = {"name": name, "type": type(module).__name__}
                    if hasattr(module, "code"):
                        row["code"] = module.code
                    if hasattr(module, "_fx_module"):
                        row["fx"] = module._fx_module.code
                        row["in_to_out_dups"] = module._in_to_out_dups
                        row["is_reusables"] = module.is_reusables
                        row["jit"] = str(module._jit_ir)
                    rows.append(row)
                (graph_directory / f"partitions-{index}.json").write_text(json.dumps(rows, indent=2))
                return False

            register_pass_at_optimization_pass(dump_state_partitions, OptimizationPassPlacement.POST_PARTITIONER)
        _registered = True
