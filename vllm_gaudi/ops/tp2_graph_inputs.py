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
    if isinstance(metadata, dict):
        return metadata.items()
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
            value = roots["metadata"]
            for name in self.name:
                value = value[name] if isinstance(value, dict) or isinstance(name, int) else getattr(value, name)
            return value
        if self.source == "attention_inputs":
            return roots["attention_inputs"][self.name]
        return getattr(self.owner, self.name)


class FixedDecodeInputs:
    """Only copy changing decoder inputs; weights and state allocations are held.

    The graph owner invalidates the plan before replacing state/KV allocations.
    Binding discovery uses storage identity so lifted views of metadata remain
    connected to the persistent input storage. Different views are not merged
    merely because they share one storage (e.g. separate cosine/sine slices).
    """

    def __init__(self, model, roots, captured_inputs, native_bridge=None):
        used = {_storage(value) for row in captured_inputs for value in row if isinstance(value, torch.Tensor)}
        candidates = []
        for name in ("positions", "hidden_states", "residual", "pre_mix", "input_ids", "metadata_pack", "pp_wire"):
            destination = (roots.get("metadata_destination", roots.get(name))
                           if name == "metadata_pack" else roots.get(name))
            candidates.append(("root", name, None, destination))
        metadata = roots["metadata"]
        if "attention_inputs" in roots:
            pack = roots.get("metadata_destination", roots.get("metadata_pack"))
            for index, value in enumerate(roots["attention_inputs"]):
                if pack is None or _storage(value) != _storage(pack):
                    candidates.append(("attention_inputs", index, None, value))
        else:

            def discover(value, path=()):
                for name, item in _metadata_items(value):
                    if isinstance(item, torch.Tensor):
                        candidates.append(("metadata", (*path, name), None, item))
                    elif isinstance(item, dict) or is_dataclass(item):
                        discover(item, (*path, name))

            discover(metadata)
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
        self.state_generation = roots.get("state_generation")
        self.state_tensors = tuple(roots.get("state_tensors", ()))
        self.state_signatures = tuple(
            (_storage(value), _address(value), _layout(value)) for value in self.state_tensors)
        adapter_name = getattr(roots.get("adapter"), "name", "")
        self.native_staging = adapter_name == "deepseek_v4" or adapter_name.startswith("deepseek_v41_")
        self.native_preflight = None
        if adapter_name.startswith("deepseek_v41_"):
            from vllm_gaudi import envs
            if envs.VLLM_HPU_DSV41_NATIVE_INPUT_PREFLIGHT:
                if (getattr(native_bridge, "fixed_input_preflight_api_version", None) != 1
                        or not hasattr(native_bridge, "FixedInputPreflight")):
                    raise RuntimeError("V4.1 native input preflight requires the version 1 bridge API")
                self.native_preflight = native_bridge.FixedInputPreflight(self.state_tensors, self.tensors())
        self.last_invalidation_reason = None

    def updates(self, roots, sources=None):
        """Preflight bindings, optionally changing only selected source classes."""
        sources = None if sources is None else frozenset(sources)
        self.last_invalidation_reason = None
        metadata = roots["metadata"]
        if roots.get("state_generation") != self.state_generation:
            self.last_invalidation_reason = "state_generation"
            return None
        state = tuple(roots.get("state_tensors", ()))
        if self.native_preflight is None and (len(state) != len(self.state_tensors) or any(
                not isinstance(value, torch.Tensor) or (_storage(value), _address(value), _layout(value)) != signature
                for value, signature in zip(state, self.state_signatures))):
            self.last_invalidation_reason = "state_allocation"
            return None
        if type(metadata) is not self.metadata_type:
            self.last_invalidation_reason = "metadata_type"
            return None
        if any(getattr(metadata, name, None) != value for name, value in self.static_metadata.items()):
            self.last_invalidation_reason = "metadata_contract"
            return None
        if self.native_preflight is not None:
            values = [binding.read(roots) if sources is None or binding.source in sources else binding.destination
                      for binding in self.bindings]
            changed = self.native_preflight.updates(state, values)
            if changed is None:
                self.last_invalidation_reason = "native_preflight"
                return None
            return [(self.bindings[index].destination, values[index]) for index in changed]
        pending = []
        for binding in self.bindings:
            source = (binding.read(roots)
                      if sources is None or binding.source in sources else binding.destination)
            if not isinstance(source, torch.Tensor) or _layout(source) != binding.layout:
                self.last_invalidation_reason = f"input_layout:{binding.name}"
                return None
            if _address(source) != _address(binding.destination):
                if _storage(source) == _storage(binding.destination):
                    # Avoid overwriting a later input view in the same update
                    # transaction. Rebuild before any mutation instead.
                    self.last_invalidation_reason = f"input_overlap:{binding.name}"
                    return None
                pending.append((binding.destination, source))
        return pending

    def apply(self, updates, graph=None):
        if self.native_staging and graph is not None:
            graph.stage_fixed_inputs([source for _, source in updates], [destination for destination, _ in updates])
        else:
            for destination, source in updates:
                destination.copy_(source)
        self.input_copies += len(updates)

    def tensors(self):
        return [binding.destination for binding in self.bindings]
