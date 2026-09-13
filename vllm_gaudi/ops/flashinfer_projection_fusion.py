# SPDX-License-Identifier: Apache-2.0
"""Opt-in, fail-closed fusion of a qualified CGUID norm/FP8 projection.

There is deliberately no automatic registration or production allowlist.
The caller must supply experimentally qualified shapes before compilation.
Ordinary quantization, collectives, additional consumers, training, symbolic
shapes and unrecognized metadata are left unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
import operator
import threading

import torch
from torch.fx import Node

_lock = threading.Lock()
_registered_shapes = None
_fused_count = 0
_MISSING = object()


def _target(node, name):
    return isinstance(node, Node) and node.op == "call_function" and str(node.target).removesuffix(".default") == name


def _arg(node, index, name, default=_MISSING):
    return node.kwargs.get(name, node.args[index] if len(node.args) > index else default)


def _tensor(node, shape, dtype):
    if not isinstance(node, Node):
        return False
    meta = node.meta
    return (meta.get("output_shapes") == [torch.Size(shape)] and meta.get("output_dtypes") == [dtype]
            and getattr(meta.get("output_device"), "type", None) == "hpu" and meta.get("output_contiguous") == [True]
            and not getattr(meta.get("val"), "requires_grad", True))


def _getitem(node, index):
    return (isinstance(node, Node) and node.op == "call_function" and node.target is operator.getitem
            and len(node.args) == 2 and node.args[1] == index)


def _exclusive(node, users):
    return set(node.users) == set(users)


def _identity_views(node):
    views = []
    while any(_target(node, name) for name in ("aten.view", "aten.reshape", "aten._unsafe_view", "aten.alias")):
        if not node.args or node.kwargs or not isinstance(node.args[0], Node):
            return None, []
        source = node.args[0]
        if (not node.meta.get("output_shapes") or node.meta.get("output_shapes") != source.meta.get("output_shapes")
                or node.meta.get("output_contiguous") != [True] or source.meta.get("output_contiguous") != [True]):
            return None, []
        views.append(node)
        node = source
    return node, views


def _private_temporary_update(node):
    """Prove a Bridge-created add_ has no externally observable mutation.

    The owner must be a fresh functional add/native residual inside this graph,
    never a placeholder, parameter or view. Every intervening owner must have
    only non-aliasing reads before this update, apart from the update chain.
    Replacing such an internal storage-reuse optimization with a fresh residual
    preserves the original functional model contract.
    """
    if not _target(node, "aten.add_.Tensor") or len(node.args) != 2 or node.kwargs:
        return False
    order = {value: index for index, value in enumerate(node.graph.nodes)}
    current, child, owners = node.args[0], node, []
    while _target(current, "aten.add_.Tensor"):
        if len(current.args) != 2 or current.kwargs:
            return False
        owners.append((current, child))
        child, current = current, current.args[0]
    fresh_native = (_getitem(current, 3) and _target(current.args[0], "custom_op.flashinfer_gaudi_add_rmsnorm_quant"))
    if not _target(current, "aten.add.Tensor") and not fresh_native:
        return False
    owners.append((current, child))
    for owner, next_update in owners:
        for user in owner.users:
            if user is next_update:
                continue
            if user.op != "call_function" or order[user] >= order[node]:
                return False
            schema = getattr(user.target, "_schema", None)
            if schema is None or any(result.alias_info is not None for result in schema.returns):
                return False
            if any(argument.alias_info is not None and argument.alias_info.is_write for argument in schema.arguments):
                return False
    return True


def _fresh_output_value(source):
    value = source.meta["val"]
    mode = getattr(value, "fake_mode", None)
    with mode if mode is not None else nullcontext():
        return torch.empty(source.meta["output_shapes"][0], dtype=source.meta["output_dtypes"][0], device=value.device)


def _match(gemm, allowed_shapes):
    if not _target(gemm, "hpu.fp8_gemm_v2"):
        return None
    if (_arg(gemm, 1, "trans_A") is not False or _arg(gemm, 3, "trans_B") is not True
            or _arg(gemm, 4, "D", None) is not None or _arg(gemm, 5, "out_dtype") != torch.bfloat16
            or _arg(gemm, 8, "bias", None) is not None or _arg(gemm, 9, "accumulate", False) is not False
            or _arg(gemm, 10, "B_scale_shape", None) is not None):
        return None
    q, scale = _arg(gemm, 0, "A"), _arg(gemm, 6, "A_scale_inv")
    if not _getitem(q, 0) or not _target(scale, "aten._to_copy"):
        return None
    cast = q.args[0]
    if not _target(cast, "hpu.cast_to_fp8_v2") or scale.kwargs != {"dtype": torch.float32} or len(scale.args) != 1:
        return None
    if (len(cast.args) != 5 or cast.kwargs or cast.args[2] is not False or cast.args[3] is not False
            or cast.args[4] != torch.float8_e4m3fn):
        return None
    normed, inverse = cast.args[:2]
    raw_normed, views = _identity_views(normed)
    if not _getitem(raw_normed, 0) or not _target(inverse,
                                                  "aten.mul.Tensor") or inverse.args[1:] != (1.0, ) or inverse.kwargs:
        return None
    reciprocal = inverse.args[0]
    add_scale = scale.args[0]
    if (not _target(reciprocal, "aten.reciprocal") or reciprocal.args != (add_scale, ) or reciprocal.kwargs
            or not _target(add_scale, "aten.add.Tensor") or add_scale.args[1:] != (1e-8 / 240., ) or add_scale.kwargs):
        return None
    cguid, norm = add_scale.args[0], raw_normed.args[0]
    if (not _target(cguid, "hpu.calculate_scale_for_cast") or cguid.kwargs
            or cguid.args not in ((normed, 2, 0, -1, True, 240.), (normed, 2, 0, -1, True, 240., 1.))
            or not _target(norm, "hpu.rms_norm") or norm.kwargs or len(norm.args) not in (3, 5)
            or (len(norm.args) == 5 and norm.args[3:] != (None, False)) or norm.args[2] != 1e-6):
        return None
    summed, gamma = norm.args[:2]
    if (not (_target(summed, "aten.add.Tensor") or _private_temporary_update(summed)) or len(summed.args) != 2
            or summed.kwargs):
        return None
    x, residual = summed.args
    sizes = gemm.meta.get("output_shapes", [])
    if len(sizes) != 1 or len(sizes[0]) != 2 or any(type(size) is not int for size in sizes[0]):
        return None
    batch, projection = sizes[0]
    width = 5120
    if (batch, width, projection) not in allowed_shapes:
        return None
    weight, weight_scale = _arg(gemm, 2, "B"), _arg(gemm, 7, "B_scale_inv")
    if (not all(_tensor(node, (batch, width), torch.bfloat16) for node in (x, residual, summed, normed))
            or not _tensor(gamma, (width, ), torch.bfloat16) or not _tensor(weight,
                                                                            (projection, width), torch.float8_e4m3fn)
            or not _tensor(weight_scale, (projection, ), torch.float32)
            or not _tensor(scale, (batch, 1), torch.float32) or not _tensor(q, (batch, width), torch.float8_e4m3fn)):
        return None
    # No extra norm, scale, quantized or inverse-RMS consumer may disappear.
    chain = (raw_normed, *reversed(views))
    edges = ((norm, (raw_normed, )), (normed, (cguid, cast)), (cguid, (add_scale, )), (add_scale, (reciprocal, scale)),
             (reciprocal, (inverse, )), (inverse, (cast, )), (cast, (q, )), (q, (gemm, )), (scale, (gemm, )))
    if not all(_exclusive(node, users) for node, users in edges):
        return None
    if any(not _exclusive(parent, (child, )) for parent, child in zip(chain, chain[1:])):
        return None
    return x, residual, gamma, summed, norm, normed, q, scale


def fuse_projection_graph(graph_module, allowed_shapes):
    """Return the rewrite count; unknown graphs are not changed."""
    if not allowed_shapes:
        return 0
    # Only this decode projection has jointly qualified against its matching
    # CGUID consumer and the independent formula. No ordinary-scale promotion.
    if not set(allowed_shapes).issubset({(8, 5120, 34816)}):
        raise ValueError("Unqualified projection shape requested")
    graph = graph_module.graph
    count = 0
    for gemm in list(graph.nodes):
        match = _match(gemm, allowed_shapes)
        if match is None:
            continue
        x, residual, gamma, summed, norm, normed, q, scale = match
        order = {node: index for index, node in enumerate(graph.nodes)}
        if any(order[user] < order[norm] for user in summed.users):
            continue
        from flashinfer_gaudi._native import add_rmsnorm_quant_op
        native = add_rmsnorm_quant_op()
        if native is None:
            raise RuntimeError("Requested projection fusion requires the native norm extension")
        with graph.inserting_before(norm):
            fused = graph.call_function(native.default, (x, residual, gamma, 1e-6, False))
            replacements = []
            sources = (q, scale, normed, summed)
            values = tuple(_fresh_output_value(source) for source in sources)
            for index, source in enumerate(sources):
                node = graph.call_function(operator.getitem, (fused, index))
                node.meta = dict(source.meta)
                node.meta["val"] = values[index]
                node.meta["placement"] = "hpu_cluster"
                replacements.append(node)
            fused.meta = dict(norm.meta)
            fused.meta.pop("tensor_meta", None)
            fused.meta["val"] = values
            fused.meta["placement"] = "hpu_cluster"
            for name in ("output_shapes", "output_dtypes", "output_layouts", "output_strides", "output_contiguous",
                         "output_offset"):
                if all(name in source.meta for source in sources):
                    fused.meta[name] = [source.meta[name][0] for source in sources]
        q.replace_all_uses_with(replacements[0])
        scale.replace_all_uses_with(replacements[1])
        summed.replace_all_uses_with(replacements[3])
        if _target(summed, "aten.add_.Tensor"):
            # FX DCE deliberately retains impure nodes, even after their last
            # use disappears. The private-owner proof above permits removal
            # of this obsolete storage update; leaving it can mutate the
            # new native producer's input before that producer executes.
            graph.erase_node(summed)
        count += 1
        graph.eliminate_dead_code()
    if count:
        graph.lint()
        graph_module.recompile()
    return count


def register_projection_fusion_pass(qualified_shapes=()):
    """Explicit experimental registration; an empty/default request is a no-op."""
    global _registered_shapes
    shapes = frozenset(tuple(shape) for shape in qualified_shapes)
    if not shapes:
        return
    if not shapes.issubset({(8, 5120, 34816)}):
        raise ValueError("Unqualified projection shape requested")
    with _lock:
        if _registered_shapes is not None:
            if _registered_shapes != shapes:
                raise RuntimeError("Cannot change projection fusion shapes after registration")
            return
        from habana_frameworks.torch.dynamo.compile_backend import passes

        def fuse(ctx):
            global _fused_count
            if getattr(ctx, "is_training", False) or getattr(ctx, "is_backward", False):
                return False
            count = fuse_projection_graph(ctx.graph_module, shapes)
            if count:
                # This callback runs after Bridge's layout/view rewrites.
                # Re-executing the whole graph with FakeTensor propagation
                # is invalid (e.g. flattened HPU BMM inputs are no longer a
                # valid aten.bmm program). Only the new native nodes receive
                # fresh, non-aliasing values and complete placement metadata.
                _fused_count += count
            return bool(count)

        passes.register_pass_at_optimization_pass(fuse, passes.OptimizationPassPlacement.PRE_PLACEMENT)
        _registered_shapes = shapes


def projection_fusion_stats():
    return {
        "compiled_matches": _fused_count,
        "qualified_shapes": sorted(_registered_shapes or ()),
        "production_default": False
    }
