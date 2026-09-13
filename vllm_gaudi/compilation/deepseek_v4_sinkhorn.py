# SPDX-License-Identifier: Apache-2.0
"""Fuse the fixed C1 mHC Sinkhorn chain without moving rounding boundaries."""
import copy

import torch
from torch.fx import Node


def _call(node, target):
    return isinstance(node, Node) and node.op == "call_function" and node.target is target


def _epsilon_add(node):
    return (_call(node, torch.ops.aten.add.Tensor) and len(node.args) == 2
            and isinstance(node.args[1], (int, float)) and node.args[1] == 1e-6
            and node.kwargs.get("alpha", 1) == 1)


def _shape(node):
    value = node.meta.get("val", node.meta.get("tensor_meta"))
    return value is not None and tuple(value.shape) == (1, 4, 4) and value.dtype == torch.float32


def fuse_sinkhorn4(module, replacement=None):
    """Replace only complete, unobserved39-step chains after softmax+epsilon."""
    replacement = replacement or torch.ops.custom_op.custom_deepseek_v4_sinkhorn4_gaudi2.default
    graph = module.graph
    count = 0
    for final in reversed(list(graph.nodes)):
        if not _call(final, torch.ops.aten.div.Tensor) or not _shape(final):
            continue
        current, chain = final, set()
        for step in range(38, -1, -1):
            if not _call(current, torch.ops.aten.div.Tensor) or len(current.args) != 2:
                break
            source, denominator = current.args
            if not _epsilon_add(denominator):
                break
            summed = denominator.args[0]
            if not _call(summed, torch.ops.aten.sum.dim_IntList) or len(summed.args) != 3:
                break
            expected_dim = 1 if step % 2 == 0 else 2
            if (summed.args[0] is not source or summed.args[2] is not True
                    or list(summed.args[1]) not in ([expected_dim], [expected_dim - 3])
                    or summed.kwargs.get("dtype") not in (None, torch.float32)):
                break
            chain.update((current, denominator, summed))
            current = source
        else:
            if not _epsilon_add(current) or not _shape(current):
                continue
            softmax = current.args[0]
            if not (_call(softmax, torch.ops.aten._softmax.default)
                    and len(softmax.args) == 3 and softmax.args[1] in (-1, 2)
                    and softmax.args[2] is False):
                continue
            if any(user not in chain for node in chain if node is not final for user in node.users):
                continue
            with graph.inserting_before(final):
                fused = graph.call_function(replacement, (current,))
                fused.meta = copy.copy(final.meta)
            final.replace_all_uses_with(fused)
            for node in reversed(list(graph.nodes)):
                if node in chain:
                    graph.erase_node(node)
            count += 1
    if count:
        graph.lint()
        module.recompile()
    return count
