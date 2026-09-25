# SPDX-License-Identifier: Apache-2.0
"""Isolate optional Reindex tiles from state writers and mandatory consumers."""
import operator

import torch
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module

from vllm_gaudi.compilation.deepseek_v41_overlap import _partition_metadata

TILE_NAME = "custom_op.custom_deepseek_v41_reindex_tile_gaudi2.default"
TILE_NAMES = frozenset((TILE_NAME, "custom_op.custom_deepseek_v41_reindex_tiled_gaudi2.default"))


def optional_tile_bound(module):
    calls = [node for node in module.graph.nodes if node.op == "call_function"]
    tiles = [node for node in calls if str(node.target) in TILE_NAMES]
    if not tiles:
        return 0
    # A recipe is omittable only when the entire operation is the pure tile.
    # Other arithmetic, mutations or communication must remain mandatory.
    if len(tiles) != 1 or len(calls) != 1 or any(node.op not in ("placeholder", "output", "call_function")
                                                 for node in module.graph.nodes):
        raise RuntimeError("Optional Reindex tile was fused with mandatory work")
    ordinal = tiles[0].args[-1]
    if type(ordinal) is not int or not 0 <= ordinal <= 8:
        raise RuntimeError("Optional Reindex ordinal is not a fixed tile 0..7 or whole scorer 8")
    return ordinal + 1


def split_reindex_tiles(module):
    graph = module.graph
    changed = 0
    for call in list(graph.nodes):
        if call.op != "call_module" or call.kwargs:
            continue
        child = module.get_submodule(call.target)
        if not isinstance(child, torch.fx.GraphModule):
            continue
        tiles = [node for node in child.graph.nodes if str(node.target) in TILE_NAMES]
        if not tiles:
            continue
        assignment, current = {}, 0
        for node in child.graph.nodes:
            if str(node.target) in TILE_NAMES:
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
                                raise RuntimeError("Reindex split requires explicit tuple consumers")
                            user.replace_all_uses_with(result[user.args[1]])
                            graph.erase_node(user)
                    continue
                if node.op == "call_module":
                    target = f"{call.target}_reindex_{node.target}"
                    partition = wrapper.get_submodule(node.target)
                    bound = optional_tile_bound(partition)
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
                elif node.op == "get_attr":
                    raise RuntimeError("Unexpected captured attribute in Reindex partition")
                else:
                    copied = graph.node_copy(node, env.__getitem__)
                    if node.target != operator.getitem or "_mhc_result_meta" not in copied.args[0].meta:
                        raise RuntimeError("Unexpected Reindex wrapper operation")
                    copied.meta = dict(copied.args[0].meta["_mhc_result_meta"][copied.args[1]])
                    copied.meta["placement"] = "eager"
                env[node] = copied
        graph.erase_node(call)
    if changed:
        graph.lint()
        module.recompile()
    return changed
