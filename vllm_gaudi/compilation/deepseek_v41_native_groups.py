# SPDX-License-Identifier: Apache-2.0
"""Native command replay for the unchanged TP4 compiled layer groups."""
import atexit
from contextlib import contextmanager
import copy
import operator
from types import SimpleNamespace
import weakref

import torch

from vllm_gaudi.ops import tp2_prepared_plan as prepared


def _storage(value):
    return value.untyped_storage()._cdata


def _mutable_buffers(owner):
    if hasattr(owner, "shared") and hasattr(owner, "layers"):
        from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors
        return stage_state_tensors(owner)
    # Small standalone owners do not publish the model's state schema.
    return tuple(owner.buffers())


class _DirectGroupEntry:
    """Reuse the TP2 input contract at an already captured TP4 group boundary."""

    def __init__(self, owner, arguments, group, graph, key, inputs, plan, bridge):
        from torch.utils import _pytree
        from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs
        self.owner = weakref.ref(owner)
        self.group = weakref.ref(group)
        self.graph, self.key = graph, key
        used = {_storage(value) for value in inputs if isinstance(value, torch.Tensor)}
        state = {id(value) for value in _mutable_buffers(owner) if _storage(value) in used}
        self.state_fields = []
        for name, value in owner.named_buffers():
            if id(value) in state:
                parent, _, leaf = name.rpartition(".")
                self.state_fields.append((owner.get_submodule(parent), leaf))
        self.bindings = FixedDecodeInputs(owner, self.roots(arguments), [inputs], native_bridge=bridge)
        # Every changing tensor actually used by the graph must be covered,
        # including separate views of the same packed Engram allocation.
        external, _ = _pytree.tree_flatten(arguments)
        destinations = {_storage(value) for value in self.bindings.tensors()}
        if any(isinstance(value, torch.Tensor) and _storage(value) in used
               and _storage(value) not in destinations for value in external):
            raise RuntimeError("TP4 direct entry left a changing input unbound")
        self.allowed_outputs = used | {_storage(value) for value in plan.outputs()}
        self.result = None

    def roots(self, arguments):
        from torch.utils import _pytree
        owner = self.owner()
        if owner is None:
            raise RuntimeError("TP4 direct entry outlived its state owner")
        flat, _ = _pytree.tree_flatten(arguments)
        return dict(metadata={str(index): value for index, value in enumerate(flat)},
                    state_generation=(owner.generation, getattr(owner, "precision_fingerprint", None)),
                    state_tensors=tuple(getattr(module, name) for module, name in self.state_fields),
                    adapter=SimpleNamespace(name="deepseek_v41_tp4_group"))

    def finish(self, arguments, result):
        from torch.utils import _pytree
        external, _ = _pytree.tree_flatten(arguments)
        input_ids = {id(value): index for index, value in enumerate(external) if isinstance(value, torch.Tensor)}
        values, self.result_spec = _pytree.tree_flatten(result)
        outputs = []
        for value in values:
            if isinstance(value, torch.Tensor):
                if _storage(value) in self.allowed_outputs:
                    # A returned input can be mutated by the captured graph.
                    # Its current value lives in the staged destination.
                    outputs.append(("held", value))
                elif id(value) in input_ids:
                    outputs.append(("input", input_ids[id(value)]))
                else:
                    # An unrecorded output allocation may represent work in
                    # an outer wrapper; preserve that wrapper in this case.
                    return None
            else:
                outputs.append(("held", value))
        self.result = outputs
        return self

    def replay(self, arguments):
        from torch.utils import _pytree
        if prepared._native_graphs.get(self.key) is not self.graph:
            return False, None
        updates = self.bindings.updates(self.roots(arguments))
        if updates is None:
            return False, None
        # No fallback is permitted after staging or state mutation begins.
        self.bindings.apply(updates, self.graph)
        self.graph.replay_fixed()
        prepared._tp4_direct_group_replays += 1
        group = self.group()
        if group is not None:
            group.replays += 1
        external, _ = _pytree.tree_flatten(arguments)
        return True, _pytree.tree_unflatten(
            [external[value] if kind == "input" else value for kind, value in self.result], self.result_spec)


@contextmanager
def capture_group_entry(owner, arguments):
    previous = getattr(prepared._local, "tp4_group_entry", None)
    context = SimpleNamespace(owner=owner, arguments=arguments, entry=None)
    prepared._local.tp4_group_entry = context
    try:
        yield context
    finally:
        prepared._local.tp4_group_entry = previous


def _collectives():
    ops = torch.ops._c10d_functional
    return ops.all_reduce_.default, ops.all_reduce.default, ops.all_gather_into_tensor.default


def _fresh_recipe_output(gm, node):
    if node.op == "call_function" and node.target in prepared._view_targets():
        return _fresh_recipe_output(gm, node.args[0])
    if node.op == "call_function" and node.target is operator.getitem:
        source, index = node.args
        if source.op != "call_module" or type(index) is not int:
            return False
        recipe = gm.get_submodule(source.target)
        return hasattr(recipe, "_recipe_id") and index not in (recipe._in_to_out_dups or {}).values()
    if node.op == "call_module":
        recipe = gm.get_submodule(node.target)
        return hasattr(recipe, "_recipe_id") and not recipe._in_to_out_dups
    return False


def eligible_collectives(gm):
    """Reject wider buckets and externally observable in-place reductions."""
    targets = _collectives()
    nodes = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in targets]
    if not nodes:
        return ()
    last_compute = None
    for node in gm.graph.nodes:
        if node.op == "call_module":
            last_compute = node
        if node in nodes:
            last_compute = None
            value = node.args[0].meta.get("val")
            if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1
                    or value.dtype != torch.bfloat16 or not value.is_contiguous()
                    or not 0 < value.numel() <= 32768):
                return ()
            if node.target in targets[:2]:
                if value.numel() % 4 or node.args[1] != "sum":
                    return ()
                source = node.args[0]
                if set(source.users) != {node} or not _fresh_recipe_output(gm, source):
                    return ()
            elif node.args[1] != 4:
                return ()
    # A replay boundary includes the compiled consumer of the last collective.
    return tuple(nodes) if last_compute is not None else ()


def _lower_collectives(gm, nodes):
    gather = _collectives()[2]
    for node in nodes:
        value = node.args[0].meta["val"]
        shape = tuple(node.meta["val"].shape)
        target = (torch.ops.vllm_gaudi.tp4_allgather_plain.default if node.target == gather
                  else torch.ops.vllm_gaudi.tp4_allreduce_plain.default)
        with gm.graph.inserting_before(node):
            flat = gm.graph.call_function(torch.ops.aten.view.default, (node.args[0], [1, value.numel()]))
            flat.meta = dict(node.args[0].meta, val=value.reshape(1, -1))
            exchanged = gm.graph.call_function(target, (flat,))
            exchanged.meta = dict(node.meta, val=node.meta["val"].reshape(1, -1))
            result = gm.graph.call_function(torch.ops.aten.view.default, (exchanged, list(shape)))
            result.meta = dict(node.meta)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops._c10d_functional.wait_tensor.default:
            # Prepared/native producers register completion on the destination.
            # Its downstream compiled recipe keeps that dependency explicitly.
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


class NativeTP4Group(prepared.PreparedGroupModule):
    """Own fixed command addresses, mutable state and retirement together."""

    def __init__(self, original, owner, collective_count, bridge, backend):
        super().__init__(original, tp4=True)
        self.owner = weakref.ref(owner)
        self.collective_count = collective_count
        self.runtime = bridge, backend
        self.captured = {}
        self.binding_info = {}

    def _runtime(self):
        return self.runtime

    def _owner_key(self):
        owner = self.owner()
        if owner is None:
            raise RuntimeError("Native TP4 group outlived its model state owner")
        return id(owner), id(self), owner.generation

    def _pinned_inputs(self, inputs):
        owner = self.owner()
        storage = {value.untyped_storage()._cdata for value in owner.buffers()}
        return tuple(index for index, value in enumerate(inputs)
                     if isinstance(value, torch.Tensor) and value.untyped_storage()._cdata in storage)

    @staticmethod
    def _same_pinned(inputs, captured, pinned):
        return all(inputs[index].data_ptr() == captured[index].data_ptr() for index in pinned)

    def forward(self, input_list):
        inputs = list(input_list)
        input_list.clear()
        owner_key = self._owner_key()
        for plan, plan_owner in zip(self.plans, self.plan_owners, strict=True):
            if plan_owner != owner_key or not plan.matches(inputs):
                continue
            key = ("tp4_group", id(self), id(plan))
            entry = self.captured.get(id(plan))
            if entry is not None and not self._same_pinned(inputs, entry[0], entry[1]):
                continue
            bridge, _ = self.runtime
            graph = prepared._native_graphs.get(key)
            if graph is None:
                graph = bridge.NativeDecodeGraph()
                prepared._native_graphs[key] = graph
                prepared._native_graph_owners[key] = self.owner
                graph.configure_topology(1, self.collective_count, False)
                graph.capture([plan], [inputs])
                graph.instantiate()
                pinned = self._pinned_inputs(inputs)
                owner = self.owner()
                used = {_storage(value) for value in inputs if isinstance(value, torch.Tensor)}
                states = [value for value in _mutable_buffers(owner) if _storage(value) in used]
                # Constant weights remain retained by the plan. Only changing
                # roots and mutable state participate in replay dependencies.
                tensors = [value for index, value in enumerate(inputs)
                           if index not in pinned and isinstance(value, torch.Tensor) and value.device.type == "hpu"]
                graph.bind_dynamic_inputs(tensors)
                if states:
                    graph.bind_state_tensors(states)
                context = getattr(prepared._local, "tp4_group_entry", None)
                if context is not None and context.owner is owner:
                    if context.entry is not None:
                        raise RuntimeError("TP4 direct entry requires one complete native group")
                    context.entry = _DirectGroupEntry(
                        owner, context.arguments, self, graph, key, inputs, plan, bridge)
                self.captured[id(plan)] = tuple(inputs), pinned
                self.binding_info[id(plan)] = dict(captured_inputs=len(inputs), dynamic_inputs=len(tensors),
                                                  mutable_state=len(states), owned_input_views=len(pinned),
                                                  direct_roots=len(context.entry.bindings.tensors())
                                                  if context is not None and context.entry is not None else 0)
                prepared._native_captures += 1
            else:
                sources, destinations = [], []
                for value, captured in zip(inputs, entry[0], strict=True):
                    if isinstance(value, torch.Tensor) and value.data_ptr() != captured.data_ptr():
                        sources.append(value)
                        destinations.append(captured)
                if sources:
                    graph.stage_fixed_inputs(sources, destinations)
                graph.replay_fixed()
            self.replays += 1
            return tuple(plan.outputs())
        return self._prepare(inputs)


def prepare_native_group(gm, owner):
    """Select replay from installed capabilities and the compiled graph contract."""
    nodes = eligible_collectives(gm)
    if not nodes:
        return False
    from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
    bridge, backend = resolve_device_runtime(4)
    if getattr(bridge, "tp4_prepared_group_version", 0) != 1:
        return False
    if not bridge.native_tp4_decode_graph_available():
        raise RuntimeError("Installed TP4 group adapter lacks its pinned native runtime")
    # Resolve the actual process group before replacing any operation.
    from torch.distributed.distributed_c10d import _resolve_process_group
    for node in nodes:
        group = _resolve_process_group(node.args[-1])
        if group.size() != 4 or group._get_backend(torch.device("hpu")) is not backend:
            raise RuntimeError("Compiled collective belongs to a different TP4 communicator")
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp4_allreduce_runtime
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    with unset_fake_temporarily(), torch.inference_mode():
        initialize_tp4_allreduce_runtime()
    original = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    lowered = eligible_collectives(original)
    _lower_collectives(original, lowered)
    if not prepared._eligible(original, tp4=True):
        raise RuntimeError("Eligible TP4 graph lost its prepared replay contract")
    name = "_tp4_native_group"
    gm.add_module(name, NativeTP4Group(original, owner, len(nodes), bridge, backend))
    replacement = torch.fx.Graph()
    inputs = replacement.placeholder("input_list")
    result = replacement.call_module(name, (inputs,))
    replacement.output(result)
    gm.graph = replacement
    gm.delete_all_unused_submodules()
    gm.recompile()
    return True


atexit.register(prepared.shutdown_prepared_group_plans)
