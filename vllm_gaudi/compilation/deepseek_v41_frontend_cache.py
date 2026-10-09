# SPDX-License-Identifier: Apache-2.0
"""Guarded frontend artifacts with fresh backend and tensor ownership.

This capability is not enabled by the serving entrypoint. It saves Dynamo's
graph and input/output mapping, not recipes, device addresses or native plans.
The caller must supply a namespace bound to its sources, runtime and model.
"""
import hashlib
from functools import lru_cache
import json
import os
import pickle
from pathlib import Path
import time
import uuid

import torch
from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable, SerializableCallable


class ReboundFrontend(SerializableCallable):
    """Rebuild the unchanged backend against this process's actual inputs."""

    def __init__(self, graph_bytes):
        self.graph_bytes = graph_bytes
        self.backend = None
        self.compiled = None
        self.backend_seconds = 0.0

    @classmethod
    def from_graph(cls, graph, example_inputs=None):
        # Runtime weights/state must arrive as placeholders. Serializing a
        # concrete tensor would retain the previous process's model state.
        if any(isinstance(value, torch.Tensor) for _, value in graph.named_buffers()):
            raise ValueError("Frontend artifacts cannot contain concrete tensor buffers")
        if any(True for _ in graph.named_parameters()):
            raise ValueError("Frontend artifacts cannot contain model parameters")
        for node in graph.graph.nodes:
            if node.op == "get_attr":
                value = graph
                for part in node.target.split("."):
                    value = getattr(value, part)
                if isinstance(value, torch.Tensor):
                    raise ValueError("Frontend tensor constants require explicit current-process binding")
        sources = {node.name: node._dynamo_source for node in graph.graph.nodes
                   if node.op == "placeholder" and hasattr(node, "_dynamo_source")}
        data = GraphModuleSerializableCallable.serialize_compile_artifacts(GraphModuleSerializableCallable(graph))
        # GraphPickler retains node.meta but not Node's custom attributes.
        # AOTDispatch requires the original placeholder sources to preserve
        # static weight classification and argument/alias guard ownership.
        return cls(pickle.dumps((data, sources)))

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        return fn.graph_bytes

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        return cls(data)

    def bind(self, backend):
        if self.compiled is not None or self.backend is not None:
            raise RuntimeError("A restored frontend backend can only be bound once")
        self.backend = backend

    def __call__(self, *args):
        if self.backend is None:
            raise RuntimeError("Frontend backend must be rebound before execution")
        if self.compiled is None:
            data, sources = pickle.loads(self.graph_bytes)
            graph = GraphModuleSerializableCallable.deserialize_compile_artifacts(data).graph_module
            for node in graph.graph.nodes:
                if node.name in sources:
                    node._dynamo_source = sources[node.name]
            started = time.perf_counter()
            self.compiled = self.backend(graph, list(args))
            self.backend_seconds += time.perf_counter() - started
        return self.compiled(*args)


def owner_bindings(owner):
    """External references are resolved from the newly constructed owner."""
    values = {"owner": owner}
    if isinstance(owner, torch.nn.Module):
        values.update((f"module:{name}", value) for name, value in owner.named_modules())
        values.update((f"buffer:{name}", value) for name, value in owner.named_buffers())
        values.update((f"parameter:{name}", value) for name, value in owner.named_parameters())
    return values


class GuardedFrontendEntry:
    """Finite guarded variants; capture precedes any state-mutating execution."""

    def __init__(self, function, owner, backend, directory, *, identity, max_variants=32):
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError("Frontend cache requires a complete dependency fingerprint")
        if not 1 <= max_variants <= 128:
            raise ValueError("Frontend cache variant limit exceeds its bounded inventory")
        self.function, self.owner, self.backend = function, owner, backend
        self.directory = Path(directory)
        self.identity, self.max_variants = identity, max_variants
        self.variants = []
        self.stats = dict(captures=0, restores=0, hits=0, capture_seconds=0.0, restore_seconds=0.0)
        if self.directory.exists():
            self._restore()

    def _restore(self):
        from torch._dynamo.aot_compile import AOTCompiledFunction

        manifests = sorted(self.directory.glob("*.json"))
        if len(manifests) > self.max_variants:
            raise ValueError("Frontend cache contains too many variants")
        for path in manifests:
            record = json.loads(path.read_text())
            if record["identity"] != self.identity or record["schema"] != 1:
                raise ValueError("Frontend dependency fingerprint changed")
            binary = path.with_suffix(".bin")
            if binary.stat().st_size > 16 << 20:
                raise ValueError("Frontend artifact exceeds the metadata-only size bound")
            data = binary.read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Frontend artifact digest differs from its publication")
            started = time.perf_counter()
            artifact = AOTCompiledFunction.deserialize(data, self.function.__globals__, owner_bindings(self.owner))
            artifact._artifacts.compiled_fn.bind(self.backend)
            self.variants.append(artifact)
            self.stats["restores"] += 1
            self.stats["restore_seconds"] += time.perf_counter() - started

    def __call__(self, *args):
        from torch._dynamo.aot_compile import AOTCompiledFunction, aot_compile_fullgraph
        from torch._dynamo.hooks import Hooks

        arguments = (self.owner, *args)
        for artifact in self.variants:
            if artifact.guard_check(*arguments):
                self.stats["hits"] += 1
                return artifact(*arguments)
        if len(self.variants) == self.max_variants:
            raise RuntimeError("Frontend variant capacity exhausted before execution")
        started = time.perf_counter()
        artifact = aot_compile_fullgraph(self.function, (arguments, {}),
                                         Hooks(),
                                         ReboundFrontend.from_graph,
                                         dynamic=False)
        data = AOTCompiledFunction.serialize(artifact, owner_bindings(self.owner)).serialized_data
        if len(data) > 16 << 20:
            raise ValueError("Frontend artifact exceeds the metadata-only size bound")
        self.stats["capture_seconds"] += time.perf_counter() - started
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        temporary = self.directory / f".{stem}.tmp"
        binary = self.directory / f"{stem}.bin"
        temporary.write_bytes(data)
        temporary.replace(binary)
        record = dict(schema=1, identity=self.identity, sha256=hashlib.sha256(data).hexdigest())
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(binary.with_suffix(".json"))
        artifact._artifacts.compiled_fn.bind(self.backend)
        self.variants.append(artifact)
        self.stats["captures"] += 1
        return artifact(*arguments)


@lru_cache(maxsize=4)
def immutable_package_sources(package):
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }


def cached_group_entry(function, group, backend, directory, *, native, shared_coordinates, memory_ready):
    """Bind one common TP stage entry; topology remains an ordinary contract."""
    from vllm_gaudi.extension.logger import logger

    package = Path(__file__).resolve().parents[1]
    sources = immutable_package_sources(package)
    if isinstance(backend, str):
        from torch._dynamo.backends.registry import lookup_backend

        backend = lookup_backend(backend)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(("VLLM_HPU_", "PT_HPU_", "HCCL_",
                            "HCL_")) and name not in ("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", "PT_HPU_RECIPE_CACHE_CONFIG")
    }
    contract = dict(sources=sources,
                    torch_version=torch.__version__,
                    environment=environment,
                    runtime=os.environ.get("DSV41_SERVING_RUNTIME"),
                    serving_identity=os.environ.get("DSV41_SERVING_COMPILE_IDENTITY"),
                    function=function.__qualname__,
                    layers=[layer.layer for layer in group.layers],
                    native=native,
                    shared_coordinates=shared_coordinates,
                    memory_ready=memory_ready)
    if not contract["runtime"] or not contract["serving_identity"]:
        raise RuntimeError("Persistent frontend reuse requires the validated serving runtime and model identity")
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    root = Path(directory) / identity / f"rank{os.environ.get('LOCAL_RANK', '0')}"
    result = GuardedFrontendEntry(function, group, backend, root, identity=identity)
    logger().info("V4.1 guarded frontend initialized: layers=%s restored=%d restore_seconds=%.3f", contract["layers"],
                  result.stats["restores"], result.stats["restore_seconds"])
    return result
