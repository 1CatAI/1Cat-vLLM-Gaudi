# SPDX-License-Identifier: Apache-2.0
"""Prepared decoder segments and native compute/collective graph replay."""
from __future__ import annotations

from contextlib import contextmanager
import atexit
import copy
from dataclasses import dataclass
import operator
import os
import threading
import weakref

import torch

_registered = False
_lock = threading.Lock()
_local = threading.local()
_modules = weakref.WeakSet()
_native_graphs = {}
_native_entries = weakref.WeakKeyDictionary()
_native_entry_replays = 0


@dataclass
class _Slot:
    index: int
    warm: object


def prepared_group_stats():
    joint = []
    for graph in _native_graphs.values():
        if hasattr(graph, "joint_info"):
            joint.append(graph.joint_info())
        elif os.environ.get("VLLM_HPU_TP2_NATIVE_JOINT_PLAN", "0") == "1":
            raise RuntimeError("Native joint plan counters are unavailable")
    return {
        "prepares":
        sum(module.prepares for module in tuple(_modules)),
        "native_joint_replays":
        sum(info[0] for info in joint),
        "native_joint_compute_pages":
        sum(info[1] for info in joint),
        "native_joint_compute_submissions":
        sum(info[6] for info in joint),
        "native_joint_hcl_callbacks":
        sum(info[11] for info in joint),
        "native_entry_replays":
        _native_entry_replays,
        "replays":
        sum(module.replays for module in tuple(_modules)),
        "native_graphs":
        len(_native_graphs),
        "native_replays":
        sum(graph.replay_count() for graph in _native_graphs.values()),
        "native_segments":
        sum(graph.segment_count() for graph in _native_graphs.values()),
        "native_collectives":
        sum(graph.collective_count() for graph in _native_graphs.values()),
        "external_collectives":
        sum(graph.external_collective_count() for graph in _native_graphs.values()),
        "native_commands":
        sum(graph.captured_command_count() for graph in _native_graphs.values()),
        "native_relocations":
        sum(graph.captured_relocation_count() for graph in _native_graphs.values()),
        "native_global_program_bytes":
        sum(graph.global_program_bytes() for graph in _native_graphs.values()),
        "native_arc_program_bytes":
        sum(graph.arc_program_bytes() for graph in _native_graphs.values()),
        "native_hcl_command_bytes_per_replay":
        sum(graph.hcl_command_bytes_per_replay() for graph in _native_graphs.values()),
        "native_hcl_stream_ccb_bytes":
        sum(graph.hcl_stream_ccb_bytes() for graph in _native_graphs.values()),
        "native_hcl_replay_bytes":
        sum(graph.hcl_replay_bytes() for graph in _native_graphs.values()),
        "native_hcl_ccb_wraps":
        sum(graph.hcl_ccb_wrap_count() for graph in _native_graphs.values()),
        "native_hcl_submissions":
        sum(graph.hcl_submission_count() for graph in _native_graphs.values())
    }


def _signature_key(value):
    if isinstance(value, torch.Tensor):
        state = ((value.dtype == torch.float32 and value.numel() >= 393216)
                 or (value.dtype == torch.bfloat16 and value.shape == (1, 3, 5120)))
        address = (value.untyped_storage().data_ptr() +
                   value.storage_offset() * value.element_size() if state else None)
        return (str(value.device), str(value.dtype), tuple(value.shape), value.stride(), value.storage_offset(),
                address)
    return (type(value).__name__, value)


def _runtime():
    from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime

    bridge, backend, _ = _resolve_runtime()
    if not hasattr(bridge, "PreparedGroupPlan"):
        raise RuntimeError("The TP2 bridge lacks the version-locked prepared replay API")
    return bridge, backend


def native_decode_graph_qualification_groups():
    groups = int(os.environ.get("VLLM_HPU_NATIVE_DECODE_GRAPH_QUALIFICATION_GROUPS", "8"))
    if groups not in (1, 8):
        raise RuntimeError("Native decoder qualification requires one or eight groups")
    return groups


def native_compute_coverage_matches(segments, reductions, groups, compiled_consumer):
    if compiled_consumer:
        # The partial group starts with an external producer's exchange. The
        # full decoder retains one initial computation after embedding exchange.
        return segments == reductions + int(groups == 8)
    return segments > reductions


def _flush():
    pending = getattr(_local, "pending", None)
    if pending:
        calls = pending[:]
        pending.clear()
        bridge, _ = _runtime()[:2]
        plans = [x[0] for x in calls]
        inputs = [x[1] for x in calls]
        from vllm_gaudi import envs as gaudi_envs

        if gaudi_envs.VLLM_HPU_NATIVE_DECODE_GRAPH and len(plans) == 8:
            groups = native_decode_graph_qualification_groups()
            start = 1 if groups == 1 else 0
            stop = start + groups
            if start:
                bridge.replay_prepared_groups(plans[:start], inputs[:start])
            native_plans, native_inputs = plans[start:stop], inputs[start:stop]
            key = (groups, *(id(plan) for plan in native_plans))
            graph = _native_graphs.get(key)
            if graph is None:
                if not bridge.native_decode_graph_available():
                    raise RuntimeError("Native compute/HCL replay symbols are unavailable; no fallback was executed")
                graph = bridge.NativeDecodeGraph()
                _native_graphs[key] = graph
                graph.capture(native_plans, native_inputs)
                graph.instantiate()
                context = getattr(_local, "native_context", None)
                if groups == 8 and context is not None and context.get("outputs") is not None:
                    from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs

                    bindings = FixedDecodeInputs(context["owner"], context, inputs)
                    graph.bind_dynamic_inputs(bindings.tensors())
                    _native_entries[context["owner"]] = (graph, bindings, context["outputs"], _runtime()[1])
            else:
                graph.update_inputs(native_plans, native_inputs)
                graph.replay()
            if stop < len(plans):
                bridge.replay_prepared_groups(plans[stop:], inputs[stop:])
        else:
            # Cold plan discovery can flush a prefix while later groups are
            # still compiling. It is outside scored decode and cannot form the
            # required complete decoder graph.
            bridge.replay_prepared_groups(plans, inputs)


def record_native_decoder_outputs(hidden_states, residual):
    context = getattr(_local, "native_context", None)
    if context is not None:
        context["outputs"] = hidden_states, residual


def replay_native_decoder(owner, **roots):
    """Replay from the model entry, without traversing compiled group wrappers."""
    global _native_entry_replays
    entry = _native_entries.get(owner)
    if entry is None:
        return None
    graph, bindings, outputs, communicator = entry
    if _runtime()[1] is not communicator:
        invalidate_prepared_group_plans()
        return None
    updates = bindings.updates(roots)
    if updates is None:
        # Layout/bucket changes are discovered before copying or state mutation.
        invalidate_prepared_group_plans()
        return None
    try:
        bindings.apply(updates)
        graph.replay_fixed()
    except BaseException:
        _native_entries.pop(owner, None)
        raise
    _native_entry_replays += 1
    return outputs


@contextmanager
def collect_prepared_group_replays(**native_context):
    """Collect consecutive prepared groups, flushing before any cold capture."""
    if getattr(_local, "pending", None) is not None:
        yield
        return
    _local.pending = []
    _local.native_context = native_context or None
    try:
        yield
    except BaseException:
        _local.pending.clear()
        raise
    else:
        _flush()
    finally:
        _local.pending = None
        _local.native_context = None


def invalidate_prepared_group_plans():
    _flush()
    _native_entries.clear()
    if _native_graphs:
        for graph in tuple(_native_graphs.values()):
            graph.reset_slots()
            graph.close()
        _native_graphs.clear()
    for module in tuple(_modules):
        for plan in module.plans:
            plan.invalidate()
        module.plans.clear()
        module.signature_keys.clear()
        module.communicator_backend = None


def shutdown_prepared_group_plans():
    """Retire plan-owned communicators before Bridge/device teardown."""
    if not _native_graphs and not any(module.plans for module in tuple(_modules)):
        return
    _flush()
    torch.hpu.synchronize()
    invalidate_prepared_group_plans()


def _is_clear(target):
    return (getattr(target, "__name__", "") == "list_clear"
            or (getattr(target, "__qualname__", "") == "pass_make_boxed_graph.<locals>.<lambda>"
                and getattr(target, "__module__", "") == "habana_frameworks.torch.dynamo.compile_backend.passes"))


def _view_targets():
    return (torch.ops.aten.view.default, torch.ops.aten.as_strided.default)


def _verify_identity_view(source, result):
    if (tuple(source.shape) != tuple(result.shape) or source.stride() != result.stride()
            or source.storage_offset() != result.storage_offset() or source.dtype != result.dtype
            or source.untyped_storage()._cdata != result.untyped_storage()._cdata):
        raise RuntimeError(f"Prepared view changes layout: {tuple(source.shape)}/{source.stride()}/"
                           f"{source.storage_offset()} -> {tuple(result.shape)}/{result.stride()}/"
                           f"{result.storage_offset()}")


def _is_norm_singleton_view(source, result):
    shapes = {(1, 5120), (1, 1, 5120)}
    return (tuple(source.shape) in shapes and tuple(result.shape) in shapes
            and source.dtype == result.dtype == torch.bfloat16 and source.is_contiguous() and result.is_contiguous()
            and source.storage_offset() == result.storage_offset()
            and source.untyped_storage()._cdata == result.untyped_storage()._cdata)


def _eligible(graph):
    collective = torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm.default
    peer_exchange = torch.ops.vllm_gaudi.tp2_exchange_peer.default
    count = sum(node.target in (collective, peer_exchange) for node in graph.graph.nodes)
    if count < 2:
        return False
    # V1 only captures groups already qualified for destination-bound GDN.
    # Ordinary/reference groups retain the normal compiler execution path.
    native = torch.ops.custom_op.gdn_state_update_out.default
    if not any(child.op == "call_function" and child.target == native
               for module in graph.modules() if type(module).__name__ == "HabanaGraphModule"
               for child in module.fx_module.graph.nodes):
        return False
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            if type(module).__name__ != "HabanaGraphModule" or module.is_dynamic or module._has_randoms:
                raise RuntimeError("Prepared TP2 groups require static deterministic compiled recipes")
            continue
        if node.op == "call_function" and (node.target in (operator.getitem, collective, peer_exchange,
                                                           *_view_targets()) or _is_clear(node.target)):
            continue
        raise RuntimeError(f"Unsupported operation in prepared TP2 group: {node.op}:{node.target}")
    return True


class PreparedGroupModule(torch.nn.Module):

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.plans = []
        self.signature_keys = []
        self.communicator_backend = None
        self.prepares = 0
        self.replays = 0
        _modules.add(self)

    def _prepare(self, inputs):
        from habana_frameworks.torch.dynamo.compile_backend._recipe_compiler_C import batch_empty
        from vllm_gaudi.distributed.tp2_fused_ar_norm import _allocate_outputs

        signature = tuple(_signature_key(value) for value in inputs)
        directory = os.environ.get("GDN_STATE_DIAGNOSTIC_DIR")
        if directory and self.plans:
            import json
            from pathlib import Path

            path = Path(directory) / f"rank{os.environ.get('LOCAL_RANK', '0')}"
            path.mkdir(parents=True, exist_ok=True)
            changes = [[{
                "input": index,
                "expected": old,
                "actual": new
            } for index, (old, new) in enumerate(zip(previous, signature, strict=True)) if old != new]
                       for previous in self.signature_keys]
            (path / f"plan-miss-{id(self)}-{self.prepares}.json").write_text(json.dumps(changes, indent=2))
        bridge, backend = _runtime()
        self.communicator_backend = backend
        native = bridge.PreparedGroupPlan()
        env = {}

        def slot(warm, *, planned=None, external=False):
            return _Slot(native.add_slot(warm if planned is None else planned, external), warm)

        def resolve(value):
            if isinstance(value, torch.fx.Node):
                return env[value]
            if isinstance(value, (tuple, list)):
                return type(value)(resolve(x) for x in value)
            return value

        def warm(value):
            if isinstance(value, _Slot):
                return value.warm
            if isinstance(value, (tuple, list)):
                return type(value)(warm(x) for x in value)
            return value

        def slots(values):
            return [x if isinstance(x, _Slot) else slot(x) for x in values]

        collective = torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm.default
        peer_exchange = torch.ops.vllm_gaudi.tp2_exchange_peer.default
        results = None
        for node in self.original.graph.nodes:
            if node.op == "placeholder":
                env[node] = [slot(value, external=True) for value in inputs]
            elif node.op == "get_attr":
                value = self.original
                for component in node.target.split("."):
                    value = getattr(value, component)
                env[node] = slot(value)
            elif node.op == "call_function" and _is_clear(node.target):
                env[node] = None
            elif node.op == "call_function" and node.target == operator.getitem:
                value, index = resolve(node.args)
                env[node] = value[index]
            elif node.op == "call_function" and node.target in _view_targets():
                arguments = resolve(node.args)
                source = arguments[0]
                actual = node.target(*warm(arguments), **node.kwargs)
                if _is_norm_singleton_view(source.warm, actual) and tuple(source.warm.shape) != tuple(actual.shape):
                    # Keep a separate TensorImpl for each logical shape while
                    # retaining the exact same prepared allocation and range.
                    index = native.add_norm_view(source.index, list(actual.shape))
                    env[node] = _Slot(index, actual)
                    continue
                _verify_identity_view(source.warm, actual)
                # Fixed signatures revalidate the source layout on every replay.
                # An identity alias requires no tensor/descriptor or device node.
                env[node] = source
            elif node.op == "call_module":
                child = self.original.get_submodule(node.target)
                arguments = slots(resolve(node.args))
                if node.kwargs:
                    raise RuntimeError("Prepared compute expects positional tensor/scalar bindings")
                observed = child(*warm(arguments))
                visible = list(observed) if isinstance(observed, (list, tuple)) else [observed]
                allocations = iter(batch_empty(child._outputs_batch_data))
                duplicates = {out: inp for inp, out in (child._in_to_out_dups or {}).items()}
                visible_slots, allocated_slots = [], []
                for index, actual in enumerate(visible):
                    if index in duplicates:
                        current = arguments[duplicates[index]]
                    else:
                        current = slot(actual, planned=next(allocations))
                        allocated_slots.append(current.index)
                    visible_slots.append(current)
                native.add_compute(child._recipe_id, [x.index for x in arguments], allocated_slots)
                env[node] = tuple(visible_slots) if isinstance(
                    observed, tuple) else (visible_slots if isinstance(observed, list) else visible_slots[0])
            elif node.op == "call_function" and node.target == peer_exchange:
                arguments = slots(resolve(node.args))
                if len(arguments) != 1 or not hasattr(native, "add_peer_exchange"):
                    raise RuntimeError("Prepared TP2 runtime lacks the exchange-only node")
                actual = peer_exchange(*warm(arguments))
                result = slot(actual, planned=torch.empty_like(actual))
                native.add_peer_exchange(arguments[0].index, result.index)
                env[node] = result
            elif node.op == "call_function" and node.target == collective:
                resolved = resolve(node.args)
                arguments = slots(resolved[:3])
                epsilon = resolved[3]
                actual = collective(*warm(arguments), epsilon)
                allocated = _allocate_outputs(arguments[0].warm, arguments[2].warm)
                results_ = [
                    slot(actual[0] if i == 1 else actual[1] if i == 2 else value, planned=value)
                    for i, value in enumerate(allocated)
                ]
                native.add_exchange([x.index for x in arguments], [x.index for x in results_], epsilon)
                env[node] = results_[1], results_[2]
            elif node.op == "output":
                results = resolve(node.args[0])
            else:
                raise RuntimeError(f"Invalid prepared node {node}")
        if not isinstance(results, tuple) or not all(
                isinstance(x, _Slot) and isinstance(x.warm, torch.Tensor) for x in results):
            raise RuntimeError("Prepared decoder output must be a flat tuple of tensors")
        native.prepare(backend, [x.index for x in results])
        self.plans.append(native)
        self.signature_keys.append(signature)
        self.prepares += 1
        return warm(results)

    def forward(self, input_list):
        inputs = list(input_list)
        input_list.clear()
        if self.plans and _runtime()[1] is not self.communicator_backend:
            invalidate_prepared_group_plans()
        for plan in self.plans:
            if plan.matches(inputs):
                pending = getattr(_local, "pending", None)
                if pending is None:
                    _runtime()[0].replay_prepared_groups([plan], [inputs])
                else:
                    pending.append((plan, inputs))
                self.replays += 1
                return tuple(plan.outputs())
        # A cold path needs the preceding group's real producers submitted.
        # Capture is ordinary execution once; a failed mutation is not retried.
        _flush()
        return self._prepare(inputs)


def register_tp2_prepared_group_pass():
    global _registered
    with _lock:
        if _registered:
            return
        from habana_frameworks.torch.dynamo.compile_backend.passes import (
            OptimizationPassPlacement,
            register_pass_at_optimization_pass,
        )

        def prepare_group_pass(context):
            graph = context.graph_module
            if not _eligible(graph):
                return False
            original = torch.fx.GraphModule(graph, copy.deepcopy(graph.graph))
            prepared = PreparedGroupModule(original)
            graph.add_module("_tp2_prepared_group", prepared)
            replacement = torch.fx.Graph()
            inputs = replacement.placeholder("input_list")
            result = replacement.call_module("_tp2_prepared_group", (inputs, ))
            replacement.output(result)
            graph.graph = replacement
            graph.delete_all_unused_submodules()
            graph.recompile()
            return True

        register_pass_at_optimization_pass(prepare_group_pass, OptimizationPassPlacement.POST_PARTITIONER)
        atexit.register(shutdown_prepared_group_plans)
        _registered = True
