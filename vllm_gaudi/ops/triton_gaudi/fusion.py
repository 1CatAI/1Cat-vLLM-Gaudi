# SPDX-License-Identifier: Apache-2.0
"""Fail-closed HPU graph fusion for Gaudi2 Triton activation quantization."""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch.fx import GraphModule, Node

_registration_lock = threading.Lock()
_registered = False
_MISSING = object()
_GDN_GRAPH_BATCH_ARGS = {
    "gdn_decode_packed": 1,
    "gdn_decode_value_conv_packed": 3,
    "gdn_qk_conv_packed": 1,
}
_GDN_GRAPH_BATCHES = frozenset((8, ))
_STATEFUL_GDN_OPS = frozenset({
    "gdn_decode_conv_packed",
    "gdn_decode_packed",
    *_GDN_GRAPH_BATCH_ARGS,
})


def _validate_bridge_pass_api(passes_module: Any) -> tuple[Any, Any]:
    required = (
        "OptimizationPassPlacement",
        "register_pass_at_optimization_pass",
        "pass_reinplace_triton_gaudi_gdn_decode",
    )
    missing = [name for name in required if not hasattr(passes_module, name)]
    if missing:
        raise RuntimeError("Gaudi Bridge is missing required hpu_backend passes: " + ", ".join(missing))
    return (
        passes_module.OptimizationPassPlacement,
        passes_module.register_pass_at_optimization_pass,
    )


def _validate_bridge_gdn_graph_policy(shared_layer_module: Any) -> None:
    actual = getattr(shared_layer_module, "TRITON_GAUDI_GDN_BATCH_ARGS", None)
    actual_batches = getattr(shared_layer_module, "TRITON_GAUDI_GDN_BATCHES", None)
    graph_ops = getattr(shared_layer_module, "TRITON_GAUDI_GRAPH_OPS", ())
    shape_gate = getattr(
        shared_layer_module,
        "_is_supported_triton_gaudi_gdn_graph",
        None,
    )
    if (actual != _GDN_GRAPH_BATCH_ARGS or actual_batches != _GDN_GRAPH_BATCHES or not callable(shape_gate)):
        raise RuntimeError("Gaudi Bridge is missing shape-gated GDN placement")
    if _STATEFUL_GDN_OPS.intersection(graph_ops or ()):
        raise RuntimeError("Gaudi Bridge enables unsafe generic GDN placement")


def _resolve_triton_ops() -> tuple[Any, Any, Any]:
    try:
        return (
            torch.ops.triton_gaudi.silu_and_mul.default,
            torch.ops.triton_gaudi.dynamic_quant.default,
            torch.ops.triton_gaudi.silu_and_mul_dynamic_quant.default,
        )
    except AttributeError as exc:
        raise RuntimeError("Gaudi Bridge launch ABI v1.10 fused activation quantization op is missing") from exc


def _view_targets() -> set[Any]:
    targets = {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.alias.default,
    }
    return targets


def _argument(node: Node, position: int, name: str) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > position:
        return node.args[position]
    return _MISSING


def _unwrap_exclusive_views(node: Any) -> tuple[Any, list[Node]]:
    views: list[Node] = []
    targets = _view_targets()
    while isinstance(node, Node):
        is_view = node.op == "call_function" and node.target in targets
        is_view_method = node.op == "call_method" and node.target in (
            "view",
            "reshape",
            "flatten",
        )
        if not is_view and not is_view_method:
            break
        if len(node.users) != 1 or not node.args:
            return None, []
        views.append(node)
        node = node.args[0]
    return node, views


def pass_fuse_triton_gaudi_silu_dynamic_quant(ctx: Any) -> bool:
    """Replace an exclusive SiLU->dynamic-quant chain with one TPC node.

    Only canonical positional/keyword metadata and an alias-only, single-user
    path are accepted. Any extra consumer or mismatched specialization leaves
    the graph untouched.
    """
    silu_op, quant_op, fused_op = _resolve_triton_ops()
    graph_module: GraphModule = ctx.graph_module
    graph = graph_module.graph
    changed = False

    for quant_node in list(graph.nodes):
        if quant_node.op != "call_function" or quant_node.target != quant_op:
            continue
        quant_input = _argument(quant_node, 0, "input")
        n_cols = _argument(quant_node, 3, "n_cols")
        rows = _argument(quant_node, 4, "rows")
        if not isinstance(n_cols, int) or not isinstance(rows, int) or n_cols <= 128 or n_cols > 4096 or rows <= 0:
            continue

        source, _ = _unwrap_exclusive_views(quant_input)
        if (not isinstance(source, Node) or source.op != "call_function" or source.target != silu_op
                or len(source.users) != 1):
            continue
        silu_input = _argument(source, 0, "input")
        silu_n_cols = _argument(source, 3, "n_cols")
        silu_rows = _argument(source, 4, "rows")
        if not isinstance(silu_input, Node) or silu_n_cols != n_cols or silu_rows != rows:
            continue

        from vllm_gaudi.ops.triton_gaudi.kernels import (
            _prepare_silu_and_mul_dynamic_quant, )

        artifact_hash, block_size = _prepare_silu_and_mul_dynamic_quant(n_cols)
        with graph.inserting_before(quant_node):
            fused = graph.call_function(
                fused_op,
                args=(
                    silu_input,
                    artifact_hash,
                    block_size,
                    n_cols,
                    rows,
                ),
            )
            fused.meta = dict(quant_node.meta)
        quant_node.replace_all_uses_with(fused)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return changed


def register_silu_dynamic_quant_fusion_pass() -> None:
    """Register the fusion once, before the first hpu_backend compilation."""
    global _registered
    if _registered:
        return
    with _registration_lock:
        if _registered:
            return
        _resolve_triton_ops()
        try:
            from habana_frameworks.torch.dynamo.compile_backend import (
                passes as bridge_passes,
                shared_layer as bridge_shared_layer,
            )

            OptimizationPassPlacement, register_pass_at_optimization_pass = _validate_bridge_pass_api(bridge_passes)
            _validate_bridge_gdn_graph_policy(bridge_shared_layer)
        except (AttributeError, ImportError, RuntimeError) as exc:
            raise RuntimeError("Gaudi Bridge does not expose the required hpu_backend pass API") from exc
        register_pass_at_optimization_pass(
            pass_fuse_triton_gaudi_silu_dynamic_quant,
            OptimizationPassPlacement.PRE_PLACEMENT,
        )
        _registered = True
