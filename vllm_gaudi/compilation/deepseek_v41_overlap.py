# SPDX-License-Identifier: Apache-2.0
"""Expose residual-only mHC work between a TP producer and its real consumer.

Only partitions containing the existing control/Sinkhorn operators are split.
Arithmetic and reductions are copied verbatim; no approximate fused formula is
introduced. Engram writes and all other mutable operations remain boundaries.
"""

import threading

import torch
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module

_lock = threading.RLock()


def _result_metadata(node):
    """Keep canonical Bridge offsets; reconstruct only newly created nodes."""
    meta = dict(node.meta)
    value = meta.get("val")
    if isinstance(value, torch.Tensor):
        defaults = dict(output_device=value.device,
                        output_dtypes=[value.dtype],
                        output_layouts=[value.layout],
                        output_shapes=[list(value.shape)],
                        output_strides=[list(value.stride())],
                        output_offset=[value.storage_offset()],
                        output_contiguous=[value.is_contiguous()])
        for key, default in defaults.items():
            meta.setdefault(key, default)
    return meta


def _partition_metadata(child):
    output = next(n for n in child.graph.nodes if n.op == "output").args[0]
    if isinstance(output, torch.fx.Node):
        return _result_metadata(output)
    if not isinstance(output, tuple) or not all(isinstance(n, torch.fx.Node) for n in output):
        raise RuntimeError("mHC partitions require explicit tensor outputs")
    items = [_result_metadata(n) for n in output]
    meta = {"val": tuple(item.get("val") for item in items), "_mhc_result_meta": items}
    if items:
        meta["output_device"] = items[0].get("output_device")
        for key in ("output_dtypes", "output_layouts", "output_shapes", "output_strides", "output_offset",
                    "output_contiguous"):
            meta[key] = [value for item in items for value in item.get(key, [])]
    return meta


def deduplicate_float_casts(module, selected):
    seen = {}
    removed = 0
    for node in list(module.graph.nodes):
        if node not in selected:
            continue
        if node.op != "call_function" or str(node.target) != "aten._to_copy.default":
            continue
        if node.kwargs.get("dtype") != torch.float32:
            continue
        key = (node.args, tuple(sorted(node.kwargs.items())))
        if key in seen:
            node.replace_all_uses_with(seen[key])
            module.graph.erase_node(node)
            removed += 1
        else:
            source = node.args[0]
            base = (source.args[0]
                    if isinstance(source, torch.fx.Node) and source.target == torch.ops.aten.view.default else None)
            base_key = ((base, ), key[1])
            if base is not None and base_key in seen:
                # Casting the residual once and viewing the FP32 result keeps
                # element order and every later reduction/rounding boundary.
                with module.graph.inserting_before(node):
                    viewed = module.graph.call_function(torch.ops.aten.view.default, (seen[base_key], source.args[1]))
                    viewed.meta = dict(node.meta)
                node.replace_all_uses_with(viewed)
                module.graph.erase_node(node)
                seen[key] = viewed
                removed += 1
            else:
                seen[key] = node
    if removed:
        module.graph.eliminate_dead_code()
        module.graph.lint()
        module.recompile()
    return removed


def _pure(node):
    if node.op in ("placeholder", "get_attr"):
        return True
    if node.op != "call_function":
        return False
    schema = getattr(node.target, "_schema", None)
    return schema is not None and not schema.is_mutable and not node.is_impure()


def _private_add(node):
    """Bridge reinplacement of a fresh, exclusively owned arithmetic result.

    Input/view/state mutations never qualify. The source allocation must have
    no schema alias and exactly this user, so restoring the functional add
    cannot change another observer or any input's storage.
    """
    if node.op != "call_function" or node.target != torch.ops.aten.add_.Tensor:
        return False
    value = node.args[0]
    if not isinstance(value, torch.fx.Node) or not _pure(value) or value.op != "call_function":
        return False
    schema = value.target._schema
    return (len(value.users) == 1 and node in value.users and len(schema.returns) == 1
            and schema.returns[0].alias_info is None)


def independent_mhc_nodes(module, dependent_inputs):
    """Return a closed, pure mHC branch that cannot read a peer result."""
    nodes = list(module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    tainted = {placeholders[index] for index in dependent_inputs}
    for node in nodes:
        if any(argument in tainted for argument in node.all_input_nodes):
            tainted.add(node)
    controls = ("deepseek_v41_control_gemv", "deepseek_v41_control_batch4_f32", "deepseek_v41_control_prefetch_f32")
    seeds = [
        node for node in nodes
        if node not in tainted and any(name in str(node.target) for name in (*controls, "deepseek_v4_sinkhorn4"))
    ]
    # A sinkhorn-free/quantized/draft graph is not this candidate's contract.
    if not any(any(name in str(node.target) for name in controls) for node in seeds):
        return set()
    selected = set()

    def include(node):
        if node.op in ("placeholder", "get_attr"):
            return True
        if node in selected:
            return True
        if node in tainted or not (_pure(node) or _private_add(node)):
            return False
        if not all(include(argument) for argument in node.all_input_nodes):
            return False
        selected.add(node)
        return True

    if not all(include(seed) for seed in seeds):
        return set()
    # Retain the residual mixing branch through the last independent arithmetic
    # node. A dependent merge is deliberately left in the consumer partition.
    for node in nodes:
        if (node not in tainted and any(argument in selected for argument in node.all_input_nodes)
                and (_pure(node) or _private_add(node))):
            include(node)
    return selected


def crosses_mutable_storage(module, selected):
    reads = {argument for node in selected for argument in node.all_input_nodes}
    for node in module.graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if node.op != "call_function" or schema is None or not schema.is_mutable:
            continue
        for index, argument in enumerate(schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
            writes = []
            map_arg(value, lambda item, writes=writes: writes.append(item))
            for written in writes:
                tensor = written.meta.get("val")
                if not isinstance(tensor, torch.Tensor):
                    return True
                for read in reads:
                    other = read.meta.get("val")
                    if other is None or (isinstance(other, torch.Tensor) and torch._C._is_alias_of(tensor, other)):
                        return True
    return False


def split_mhc_consumers(module, exchange):
    graph = module.graph

    def peer_value(node):
        if not isinstance(node, torch.fx.Node) or node.op != "call_function":
            return False
        return node.target == exchange or any(peer_value(argument) for argument in node.all_input_nodes)

    audit = []
    for call in list(graph.nodes):
        if call.op != "call_module" or call.kwargs:
            continue
        child = module.get_submodule(call.target)
        if not isinstance(child, torch.fx.GraphModule):
            continue
        dependent = [index for index, value in enumerate(call.args) if peer_value(value)]
        if not dependent:
            continue
        selected = independent_mhc_nodes(child, dependent)
        if not selected:
            continue
        private_adds = [node for node in selected if _private_add(node)]
        for node in private_adds:
            node.target = torch.ops.aten.add.Tensor
        if crosses_mutable_storage(child, selected):
            for node in private_adds:
                node.target = torch.ops.aten.add_.Tensor
            continue
        casts_removed = deduplicate_float_casts(child, selected)
        if casts_removed:
            selected = independent_mhc_nodes(child, dependent)
        wrapper = split_module(child, child, lambda node, selected=selected: 0 if node in selected else 1)
        env = dict(zip((n for n in wrapper.graph.nodes if n.op == "placeholder"), call.args, strict=True))
        with graph.inserting_before(call):
            for node in wrapper.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    result = map_arg(node.args[0], env.__getitem__)
                    if not isinstance(result, torch.fx.Node):
                        # Original compiled partition outputs are tuples. Keep
                        # every original getitem user without allocating data.
                        for user in list(call.users):
                            import operator
                            if user.op != "call_function" or user.target != operator.getitem:
                                raise RuntimeError("mHC split requires explicit tuple-output consumers")
                            user.replace_all_uses_with(result[user.args[1]])
                            graph.erase_node(user)
                    else:
                        call.replace_all_uses_with(result)
                    continue
                if node.op == "call_module":
                    target = f"{call.target}_mhc_{node.target}"
                    partition = wrapper.get_submodule(node.target)
                    module.add_submodule(target, partition)
                    copied = graph.call_module(target, map_arg(node.args, env.__getitem__),
                                               map_arg(node.kwargs, env.__getitem__))
                    copied.meta = {
                        key: value
                        for key, value in call.meta.items()
                        if not key.startswith("output_") and key not in ("val", "tensor_meta")
                    }
                    copied.meta.update(_partition_metadata(partition))
                elif node.op == "get_attr":
                    raise RuntimeError("Unexpected mHC split captured attribute")
                else:
                    copied = graph.node_copy(node, env.__getitem__)
                    import operator
                    if node.target != operator.getitem or "_mhc_result_meta" not in copied.args[0].meta:
                        raise RuntimeError("Unexpected mHC partition wrapper operation")
                    copied.meta = dict(copied.args[0].meta["_mhc_result_meta"][copied.args[1]])
                    copied.meta["placement"] = "eager"
                env[node] = copied
        graph.erase_node(call)
        audit.append({
            "partition": call.target,
            "independent_nodes": len(selected),
            "casts_removed": casts_removed,
            "private_adds_restored": len(private_adds),
            "operators": [str(node.target) for node in child.graph.nodes if node in selected]
        })
    if audit:
        graph.lint()
        module.recompile()
    return audit


def make_backend():
    from habana_frameworks.torch.dynamo.compile_backend import passes
    from habana_frameworks.torch.dynamo.compile_backend.backends import hpu_backend
    from vllm_gaudi.extension.logger import logger

    def transform(ctx):
        import os
        from pathlib import Path
        directory = os.environ.get("VLLM_HPU_TP2_PLAN_DUMP_DIR")
        if directory:
            root = Path(directory) / f"rank{os.environ.get('LOCAL_RANK', '0')}"
            root.mkdir(parents=True, exist_ok=True)
            (root / f"overlap-input-{id(ctx.graph_module)}.py").write_text(
                ctx.graph_module.print_readable(print_output=False))
        audit = split_mhc_consumers(ctx.graph_module, torch.ops.vllm_gaudi.tp2_exchange_peer.default)
        from vllm_gaudi import envs
        tile_partitions = 0
        if envs.VLLM_HPU_DSV41_REINDEX_BOUNDED_PLAN:
            from vllm_gaudi.compilation.deepseek_v41_reindex import split_reindex_tiles
            tile_partitions = split_reindex_tiles(ctx.graph_module)
        logger().info("V4.1 TP/mHC partition transform examined %d modules, split %d",
                      sum(node.op == "call_module" for node in ctx.graph_module.graph.nodes), len(audit))
        if audit or tile_partitions:
            # Full fake propagation after Bridge partitioning replays already
            # canonicalized views and rejects their saved storage offsets.
            # New calls/getitems inherit the exact child-output contract above.
            passes.pass_add_fused_op_metadata(ctx)
            logger().info("V4.1 TP/mHC independent partitions: %s", audit)
            if tile_partitions:
                logger().info("V4.1 isolated optional Reindex tile recipes: %d", tile_partitions)
        return bool(audit or tile_partitions)

    def backend(graph, inputs, **kwargs):
        with _lock:
            passes.custom_pass_at_fuse_partition.append(transform)
            try:
                return hpu_backend(graph, inputs, **kwargs)
            finally:
                passes.custom_pass_at_fuse_partition.remove(transform)

    return backend
