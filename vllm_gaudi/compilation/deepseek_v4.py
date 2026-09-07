# SPDX-License-Identifier: Apache-2.0
"""Decode output dependency lowering in the normal HPU compiler pipeline."""

import threading

import torch

from vllm_gaudi.compilation.functionalization import lower_functionalized
from vllm_gaudi.compilation.ordered_boundary import attention_out_op, make_output_dependency
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()
_compile_lock = threading.RLock()


def make_backend():
    from habana_frameworks.torch.dynamo.compile_backend import passes
    from habana_frameworks.torch.dynamo.compile_backend.backends import hpu_backend

    op = torch.ops.vllm.deepseek_v4_attention.default
    ordered_op = attention_out_op()
    hop = torch.ops.higher_order.auto_functionalized_v2

    def transform(ctx):
        nodes = [node for node in ctx.graph_module.graph.nodes if node.target is hop and node.args[0] is op]
        if not nodes:
            logger.debug("DeepSeek V4 compile graph has no mutable attention boundaries")
            return False
        # Only the qualified q1 output contract is lowered. Prefill and other
        # shapes retain the normal HPU backend, without a runtime fallback.
        shapes = [tuple(base.meta["val"].shape) for node in nodes for base in node.kwargs["_all_bases"]]
        if not all(shape == (1, 64, 512) for shape in shapes):
            if any(shape and shape[0] == 1 for shape in shapes):
                raise RuntimeError(f"Unsupported DeepSeek V4 single-token output contract: {shapes}")
            logger.info("DeepSeek V4 retained prefill attention boundaries with shapes %s",
                        shapes)
            return False
        candidate, audit = lower_functionalized(ctx.graph_module, (op,), reinplace=False)
        ctx.graph_module.graph = candidate.graph
        ctx.graph_module.recompile()
        passes.pass_fake_propagation(ctx)
        dependencies = make_output_dependency(ctx.graph_module, op, ordered_op, 2)
        if dependencies != len(audit["nodes"]):
            raise RuntimeError("DeepSeek V4 output mutation is missing a producer dependency")
        passes.pass_fake_propagation(ctx)
        passes.pass_mark_placement(ctx)
        for node in ctx.graph_module.graph.nodes:
            for key in ("downstream_allreduce_name", "upstream_waittensor_name", "partition_assigned",
                        "merge_path_color"):
                node.meta.pop(key, None)
        passes.pass_allreduce_parents(ctx)
        logger.info("DeepSeek V4 compiler lowered %d ordered decode boundaries", dependencies)
        return True

    def backend(gm, inputs, **kwargs):
        with _compile_lock:
            passes.custom_pass_at_pre_stagepasses.append(transform)
            try:
                return hpu_backend(gm, inputs, **kwargs)
            finally:
                passes.custom_pass_at_pre_stagepasses.remove(transform)

    return backend
