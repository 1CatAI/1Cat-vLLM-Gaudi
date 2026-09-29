# SPDX-License-Identifier: Apache-2.0
"""Prepare static TP4 bindings around the unchanged normal HPU compiled graph."""
from dataclasses import dataclass
from itertools import count
from types import FunctionType, MethodType

import torch
from torch.utils import _pytree

_entries = count()


def input_contract(arguments):
    """Keep shape/layout and input alias relationships without pinning addresses."""
    values, structure = _pytree.tree_flatten(arguments)
    storages, signature = {}, []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.requires_grad:
                raise ValueError("Prepared TP4 groups require strided inference inputs")
            storage = value.untyped_storage()._cdata
            alias = storages.setdefault(storage, len(storages))
            signature.append((tuple(value.shape), value.stride(), value.storage_offset(), value.dtype,
                              value.device, alias))
        elif value is None or type(value) in (bool, int, float, str):
            signature.append((type(value), value))
        else:
            raise TypeError(f"Unsupported prepared TP4 input: {type(value).__name__}")
    return structure, tuple(signature)


@dataclass
class _BoundCall:
    function: object
    inputs: tuple
    replacements: tuple
    result_spec: object
    result_bindings: tuple

    def __call__(self, arguments):
        external, _ = _pytree.tree_flatten(arguments)
        inputs = list(self.inputs)
        for destination, source in self.replacements:
            inputs[destination] = external[source]
        outputs, _ = _pytree.tree_flatten(self.function(*inputs))
        result = [outputs[index] if kind == 'output' else external[index] if kind == 'input' else index
                  for kind, index in self.result_bindings]
        return _pytree.tree_unflatten(result, self.result_spec)


class PreparedTP4Group:
    """Reuse one ordinary compiled graph for each model-owned static contract.

    Weight/physical-state rebinding and unloading increment owner.generation.
    Search geometry, input shapes/layouts and alias relationships remain part of
    the key. Request-slot buffer contents stay mutable, with normal AOT mutation
    and HPU producer dependencies preserved by the original compiled callable.
    """

    def __init__(self, module, owner, backend):
        self.module, self.owner, self.backend = module, owner, backend
        self.generation = None
        self.variants = {}
        self.preparations = self.calls = 0
        self._capture = None
        method = module.forward
        function = method.__func__
        name = f"{function.__name__}_tp4_prepared_{next(_entries)}"
        entry = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name,
                             function.__defaults__, function.__closure__)
        entry.__kwdefaults__, entry.__module__ = function.__kwdefaults__, function.__module__
        self.entry = torch.compile(MethodType(entry, module), backend=self._backend, fullgraph=True, dynamic=False)

    def _backend(self, graph, inputs, **kwargs):
        compiled = self.backend(graph, inputs, **kwargs)

        def execute(*arguments):
            outputs = compiled(*arguments)
            if self._capture is not None:
                self._capture.append((compiled, tuple(arguments), outputs))
            return outputs
        return execute

    def _prepare(self, arguments):
        self._capture = []
        try:
            result = self.entry(*arguments)
            if len(self._capture) != 1:
                raise RuntimeError("Prepared TP4 binding requires exactly one complete compiled entry")
            function, bound_inputs, bound_outputs = self._capture[0]
        finally:
            self._capture = None
        external, _ = _pytree.tree_flatten(arguments)
        dynamic_ids = {id(value): index for index, value in enumerate(external) if isinstance(value, torch.Tensor)}
        replacements = tuple((index, dynamic_ids[id(value)]) for index, value in enumerate(bound_inputs)
                             if isinstance(value, torch.Tensor) and id(value) in dynamic_ids)
        outputs, _ = _pytree.tree_flatten(bound_outputs)
        output_ids = {id(value): index for index, value in enumerate(outputs) if isinstance(value, torch.Tensor)}
        result_values, result_spec = _pytree.tree_flatten(result)
        bindings = []
        for value in result_values:
            if isinstance(value, torch.Tensor):
                if id(value) in output_ids:
                    bindings.append(('output', output_ids[id(value)]))
                elif id(value) in dynamic_ids:
                    bindings.append(('input', dynamic_ids[id(value)]))
                else:
                    raise RuntimeError("Compiled TP4 output requires an unprepared Python tensor transformation")
            elif value is None or type(value) in (bool, int, float, str):
                bindings.append(('constant', value))
            else:
                raise TypeError("Prepared TP4 output must contain tensors or immutable scalars")
        return result, _BoundCall(function, bound_inputs, replacements, result_spec, tuple(bindings))

    def __call__(self, *arguments):
        if not torch.is_inference_mode_enabled() or torch.is_autocast_enabled("hpu"):
            raise RuntimeError("Prepared TP4 execution requires inference mode without autocast")
        generation = (self.owner.generation, getattr(self.owner, "precision_fingerprint", None))
        if generation != self.generation:
            self.variants.clear()
            self.generation = generation
        key = (self.owner.search_length, getattr(self.owner, "decode_token_bound", None),
               self.module.training, input_contract(arguments))
        function = self.variants.get(key)
        self.calls += 1
        if function is None:
            result, function = self._prepare(arguments)
            self.variants[key] = function
            self.preparations += 1
            return result
        return function(arguments)
