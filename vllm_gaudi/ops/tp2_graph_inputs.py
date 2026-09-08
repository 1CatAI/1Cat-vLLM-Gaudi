# SPDX-License-Identifier: Apache-2.0
"""Persistent decoder input bindings; discovery is restricted to preparation."""
from dataclasses import dataclass, fields, is_dataclass

import torch


def _layout(tensor):
    return tensor.device, tensor.dtype, tuple(tensor.shape), tensor.stride()


def _storage(tensor):
    return tensor.untyped_storage()._cdata


def _address(tensor):
    # The NIXL facade can make Tensor.data_ptr report a pool allocation base.
    return tensor.untyped_storage().data_ptr() + tensor.storage_offset() * tensor.element_size()


def _metadata_items(metadata):
    # The model runner's processed metadata is a generated named tuple.
    # Discovery happens once; replay reads only the selected tensor fields.
    if isinstance(metadata, tuple) and hasattr(metadata, "_fields"):
        return ((name, getattr(metadata, name)) for name in metadata._fields)
    if is_dataclass(metadata):
        return ((field.name, getattr(metadata, field.name)) for field in fields(metadata))
    return vars(metadata).items()


@dataclass
class _Binding:
    source: str
    name: str
    owner: object
    destination: torch.Tensor
    layout: tuple

    def read(self, roots):
        if self.source == "root":
            return roots[self.name]
        if self.source == "metadata":
            return getattr(roots["metadata"], self.name)
        return getattr(self.owner, self.name)


class FixedDecodeInputs:
    """Only copy changing decoder inputs; weights and state allocations are held.

    The graph owner invalidates the plan before replacing state/KV allocations.
    Binding discovery uses storage identity so lifted views of metadata remain
    connected to the persistent input storage. Different views are not merged
    merely because they share one storage (e.g. separate cosine/sine slices).
    """

    def __init__(self, model, roots, captured_inputs):
        used = {_storage(value) for row in captured_inputs for value in row if isinstance(value, torch.Tensor)}
        candidates = []
        for name in ("positions", "hidden_states", "residual"):
            candidates.append(("root", name, None, roots[name]))
        metadata = roots["metadata"]
        for name, value in _metadata_items(metadata):
            if isinstance(value, torch.Tensor):
                candidates.append(("metadata", name, None, value))
        # Rotary preparation is already compiled, but replaces these buffers
        # before each decoder invocation. Their values must reach captured views.
        for module in model.modules():
            for name in ("cos", "sin"):
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor):
                    candidates.append(("buffer", name, module, value))
        self.bindings = []
        seen = set()
        for source, name, owner, value in candidates:
            if not isinstance(value, torch.Tensor) or _storage(value) not in used:
                continue
            key = (_storage(value), value.dtype, value.storage_offset(), tuple(value.shape), value.stride())
            if key in seen:
                continue
            seen.add(key)
            self.bindings.append(_Binding(source, name, owner, value, _layout(value)))
        self.metadata_type = type(metadata)
        self.static_metadata = {
            name: getattr(metadata, name, None)
            for name in ("is_prompt", "direct_gdn_state", "block_size")
        }
        self.input_copies = 0

    def updates(self, roots):
        """Preflight every changing binding before copying any input."""
        metadata = roots["metadata"]
        if type(metadata) is not self.metadata_type:
            return None
        if any(getattr(metadata, name, None) != value for name, value in self.static_metadata.items()):
            return None
        pending = []
        for binding in self.bindings:
            source = binding.read(roots)
            if not isinstance(source, torch.Tensor) or _layout(source) != binding.layout:
                return None
            if _address(source) != _address(binding.destination):
                if _storage(source) == _storage(binding.destination):
                    # Avoid overwriting a later input view in the same update
                    # transaction. Rebuild before any mutation instead.
                    return None
                pending.append((binding.destination, source))
        return pending

    def apply(self, updates):
        for destination, source in updates:
            destination.copy_(source)
        self.input_copies += len(updates)

    def tensors(self):
        return [binding.destination for binding in self.bindings]
