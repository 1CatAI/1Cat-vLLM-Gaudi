# SPDX-License-Identifier: Apache-2.0
"""Keep literal scalar recipe inputs resident in the TP4 compiled decoder."""
from itertools import count
import json
import os
from pathlib import Path

import torch

from vllm_gaudi.compilation.deepseek_v4 import _compile_lock
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()
_audit = {"graphs": 0, "scalars": 0}
_diagnostic_graphs = count()


def scalar_stats():
    return dict(_audit)


def _archive_submission_graph(gm):
    """Archive the actual cold-path host graph alongside requested recipe plans."""
    directory = os.environ.get("VLLM_HPU_TP2_PLAN_DUMP_DIR")
    if not directory:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    root = Path(directory) / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for node in gm.graph.nodes:
        record = dict(name=node.name, op=node.op, target=str(node.target),
                      inputs=[value.name for value in node.all_input_nodes])
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            record["tensor"] = dict(shape=[str(size) for size in value.shape], dtype=str(value.dtype),
                                    stride=list(value.stride()))
        if node.op == "call_module":
            child = gm.get_submodule(node.target)
            record.update(module_type=type(child).__name__, recipe_id=getattr(child, "_recipe_id", None),
                          input_output_aliases=getattr(child, "_in_to_out_dups", None))
        records.append(record)
    stem = root / f"tp4-submission-{next(_diagnostic_graphs):04d}"
    stem.with_suffix(".py.txt").write_text(gm.code)
    stem.with_suffix(".json").write_text(json.dumps(dict(nodes=records), indent=2, default=str) + "\n")


def hoist_recipe_scalars(gm, *, device_type="hpu"):
    """Hoist only literal scalars read by static, non-mutating recipe inputs.

    The HPU partitioner leaves these scalar_tensor nodes in its host graph.
    With eager caching disabled each invocation otherwise compiles a tiny
    constant recipe, including the codecs' immutable NaN sentinel.
    Tensor-valued/request-dependent scalars stay untouched.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten.scalar_tensor.default:
            continue
        if len(node.args) != 1 or type(node.args[0]) not in (bool, int, float):
            continue
        device = torch.device(node.kwargs.get("device", "cpu"))
        if device.type != device_type or node.kwargs.get("requires_grad", False) or not node.users:
            continue
        supported = True
        for user in node.users:
            if user.op != "call_module":
                supported = False
                break
            recipe = gm.get_submodule(user.target)
            if not hasattr(recipe, "_recipe_id") or recipe.is_dynamic or recipe._has_randoms:
                supported = False
                break
            aliases = recipe._in_to_out_dups or {}
            indices = [i for i, arg in enumerate(user.args) if arg is node]
            if not indices or any(index in aliases for index in indices) or user.kwargs:
                supported = False
                break
        if not supported:
            continue
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily(), torch.inference_mode():
            # Copy a host literal once; no eager fill or process-wide cache
            # setting is required by the subsequent decoder invocations.
            value = torch.tensor(node.args[0], dtype=node.kwargs.get("dtype", torch.get_default_dtype()),
                                 device="cpu").to(device)
        name = f"_tp4_recipe_scalar_{count}"
        while hasattr(gm, name):
            name += "_"
        gm.register_buffer(name, value, persistent=False)
        with gm.graph.inserting_before(node):
            replacement = gm.graph.get_attr(name)
        replacement.meta = dict(node.meta)
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        count += 1
    if count:
        gm.graph.lint()
        gm.recompile()
    return count


def make_backend():
    from habana_frameworks.torch.dynamo.compile_backend import passes
    from habana_frameworks.torch.dynamo.compile_backend.backends import hpu_backend

    def resident_scalars(context):
        count = hoist_recipe_scalars(context.graph_module)
        if count:
            _audit["graphs"] += 1
            _audit["scalars"] += count
            logger.info("TP4 compiler retained %d immutable scalar recipe inputs", count)
        _archive_submission_graph(context.graph_module)
        return bool(count)

    def backend(gm, inputs, **kwargs):
        with _compile_lock:
            passes.custom_pass_at_post_partition.append(resident_scalars)
            try:
                return hpu_backend(gm, inputs, **kwargs)
            finally:
                passes.custom_pass_at_post_partition.remove(resident_scalars)

    return backend
