# SPDX-License-Identifier: Apache-2.0
"""Join a pure mHC branch into the recipe that produces the peer payload.

The earlier consumer split creates another recipe. This transform instead
extends the existing producer, retaining both its outputs and the branch's
arithmetic. It does not change transport, ordering or the shared C1 plan.
"""
import operator
from types import SimpleNamespace

import torch
from torch.fx.node import map_arg

from vllm_gaudi.compilation.deepseek_v41_overlap import (
    _partition_metadata, _pure, crosses_mutable_storage,
)


def _merge_one(module, partition, exchanges, index, *, mark_tensor_ready=False):
    branch = next((n for n in module.graph.nodes if n.op == 'call_module'
                   and n.target == partition['independent_partition']), None)
    if branch is None:
        return None
    order = {n: i for i, n in enumerate(module.graph.nodes)}
    preceding = [n for n in module.graph.nodes if n.op == 'call_function'
                 and n.target in exchanges and order[n] < order[branch]]
    if not preceding:
        return None
    exchange = preceding[-1]
    payload = exchange.args[0]
    payload_index = None
    views = (torch.ops.aten.view.default, torch.ops.aten.reshape.default,
             torch.ops.aten._unsafe_view.default, torch.ops.aten.as_strided.default)
    while payload.op == 'call_function' and payload.target in (operator.getitem, *views):
        # Keep these original wire views outside the merged recipe. Following
        # their dependency identifies the producer; it never changes layout.
        if payload.target == operator.getitem:
            if payload_index is not None or type(payload.args[1]) is not int:
                return None
            payload_index = payload.args[1]
        payload = payload.args[0]
    if payload.op != 'call_module' or payload.kwargs or branch.kwargs:
        return None
    producer = payload
    old = module.get_submodule(producer.target)
    independent = module.get_submodule(branch.target)
    if not isinstance(old, torch.fx.GraphModule) or not isinstance(independent, torch.fx.GraphModule):
        return None
    # Moving across any other mutation/opaque call needs a different proof.
    interval = [n for n in module.graph.nodes if order[producer] < order[n] < order[branch]]
    if any(n is not exchange and not _pure(n)
           and not (n.op == 'call_function' and n.target == operator.getitem) for n in interval):
        return None
    graph = torch.fx.Graph()
    outer_inputs, inputs, memo = [], {}, {}

    def external(value):
        if not isinstance(value, torch.fx.Node):
            return value
        if value not in inputs:
            outer_inputs.append(value)
            first = next((n for n in graph.nodes if n.op != 'placeholder'), None)
            if first is None:
                inputs[value] = graph.placeholder(f'input_{len(inputs)}')
            else:
                with graph.inserting_before(first):
                    inputs[value] = graph.placeholder(f'input_{len(inputs)}')
            inputs[value].meta = dict(value.meta)
        return inputs[value]

    old_env = {}
    for n, arg in zip((n for n in old.graph.nodes if n.op == 'placeholder'), producer.args, strict=True):
        old_env[n] = external(arg)
    old_result = None
    for n in old.graph.nodes:
        if n.op == 'placeholder':
            continue
        if n.op == 'output':
            old_result = map_arg(n.args[0], old_env.__getitem__)
        else:
            old_env[n] = graph.node_copy(n, old_env.__getitem__)
    if not isinstance(old_result, tuple) or not all(isinstance(n, torch.fx.Node) for n in old_result):
        return None
    if mark_tensor_ready:
        if payload_index is None or not 0 <= payload_index < len(old_result):
            raise RuntimeError('Tensor-ready producer needs an explicit peer payload output')
        packet = old_result[payload_index]
        value = packet.meta.get('val')
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or not value.is_contiguous():
            raise RuntimeError('Tensor-ready peer payload must retain its contiguous BF16 contract')
        if not 0 < value.numel() <= 32768 or value.numel() % 128:
            raise RuntimeError('Tensor-ready marker requires a small vector-aligned peer packet')

        def metadata(result):
            meta = {key: entry for key, entry in packet.meta.items()
                    if not key.startswith('output_') and key not in ('val', 'tensor_meta')}
            meta['val'] = result
            return _partition_metadata_for_value(meta)

        marked = graph.call_function(torch.ops.custom_op.custom_deepseek_v41_peer_ready_identity_gaudi2.default,
                                     (packet, True))
        # The registration has no alias schema: metadata must represent its
        # new allocation, rather than silently making the signal output a
        # view of the preceding MME result.
        marked_value = value.clone()
        marked.meta = metadata(marked_value)
        old_result = tuple(marked if i == payload_index else item for i, item in enumerate(old_result))

    def resolve(value):
        if not isinstance(value, torch.fx.Node):
            return value
        if value in memo:
            return memo[value]
        if value.op == 'call_function' and value.target == operator.getitem and value.args[0] is producer:
            result = old_result[value.args[1]]
        elif order[value] < order[producer]:
            result = external(value)
        elif value.op == 'call_function' and _pure(value):
            result = graph.node_copy(value, resolve)
        else:
            raise ValueError('mHC input is unavailable to the producer')
        memo[value] = result
        return result

    try:
        branch_env = {n: resolve(arg) for n, arg in
                      zip((n for n in independent.graph.nodes if n.op == 'placeholder'),
                          branch.args, strict=True)}
    except ValueError:
        return None
    added, branch_result = set(), None
    for n in independent.graph.nodes:
        if n.op == 'placeholder':
            continue
        if n.op == 'output':
            branch_result = map_arg(n.args[0], branch_env.__getitem__)
        elif n.op == 'get_attr':
            raise RuntimeError('Unexpected captured attribute in independent mHC')
        else:
            branch_env[n] = graph.node_copy(n, branch_env.__getitem__)
            added.add(branch_env[n])
    if not isinstance(branch_result, tuple):
        branch_result = (branch_result,)
    graph.output((*old_result, *branch_result))
    joined = torch.fx.GraphModule(old, graph)
    if crosses_mutable_storage(joined, added):
        return None
    target = f'{producer.target}_mhc_producer_{index}'
    module.add_submodule(target, joined)
    root = module.graph
    with root.inserting_before(producer):
        call = root.call_module(target, tuple(outer_inputs))
        call.meta = dict(producer.meta)
        call.meta.update(_partition_metadata(joined))
        replacements = []
        for index, item in enumerate(call.meta['_mhc_result_meta']):
            output = root.call_function(operator.getitem, (call, index))
            output.meta = dict(item)
            output.meta['placement'] = 'eager'
            replacements.append(output)
    for owner, offset in ((producer, 0), (branch, len(old_result))):
        for user in tuple(owner.users):
            if user.op != 'call_function' or user.target != operator.getitem:
                raise RuntimeError('mHC producer merge requires explicit tuple outputs')
            user.replace_all_uses_with(replacements[offset + user.args[1]])
            root.erase_node(user)
        root.erase_node(owner)
    root.eliminate_dead_code()
    root.lint()
    module.recompile()
    return dict(producer=producer.target, independent=branch.target,
                merged=target, recipes_removed=1, arithmetic_unchanged=True,
                tensor_ready_output=payload_index if mark_tensor_ready else None)


def _partition_metadata_for_value(meta):
    from vllm_gaudi.compilation.deepseek_v41_overlap import _result_metadata

    return _result_metadata(SimpleNamespace(meta=meta))


def merge_mhc_producers(module, partitions, exchanges, *, mark_tensor_ready=False):
    audit = []
    for partition in partitions:
        result = _merge_one(module, partition, exchanges, len(audit), mark_tensor_ready=mark_tensor_ready)
        if result is not None:
            audit.append(result)
    return audit
