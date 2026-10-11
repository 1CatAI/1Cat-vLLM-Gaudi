# SPDX-License-Identifier: Apache-2.0
"""Guarded frontend artifacts with fresh backend and tensor ownership.

Normal serving installs this cache alongside its persistent recipes. It saves
Dynamo's graph and input/output mapping, not recipes, device addresses or native plans.
The caller must supply a namespace bound to its sources, runtime and model.
"""
import hashlib
from itertools import chain
from functools import lru_cache
import dataclasses
import io
import json
import os
import pickle
from pathlib import Path
import time
from types import FunctionType
import uuid

import torch
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable, SerializableCallable

_LOCATION_ENVIRONMENT = frozenset((
    "VLLM_HPU_DSV41_FRONTEND_CACHE_DIR",
    "PT_HPU_RECIPE_CACHE_CONFIG",
    "VLLM_HPU_DSV4_WORKER_CPUS",
    "VLLM_HPU_DSV4_WORKER_HELPER_CPUS",
))


def cache_rank():
    # vLLM starts workers itself; torchrun's LOCAL_RANK is not its contract.
    # Use the initialized process group for both TP and PP installations.
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("LOCAL_RANK", "0"))


class _SourcePickler(pickle.Pickler):

    def reducer_override(self, value):
        from torch._guards import Source

        if isinstance(value, Source) and dataclasses.is_dataclass(value):
            fields = dataclasses.fields(value)
            if any(not field.init for field in fields):
                # Cached-hash Source reducers include derived fields, although
                # their constructor only accepts the initialization fields.
                return type(value), tuple(getattr(value, field.name) for field in fields if field.init)
        return NotImplemented


def _restore_defaults_source(base, idx_key, is_kw=False, field=None, name=None):
    from torch._dynamo.source import DefaultsSource

    result = DefaultsSource(base, idx_key, is_kw)
    if field is not None and (result.field != field or result._name != name):
        raise ValueError("Serialized default source disagrees with its reconstructed guard")
    return result


class _SourceUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == "torch._dynamo.source" and name == "DefaultsSource":
            return _restore_defaults_source
        return super().find_class(module, name)


def _source_dumps(value):
    buffer = io.BytesIO()
    _SourcePickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def _source_loads(data):
    return _SourceUnpickler(io.BytesIO(data)).load()


def _default_signature(value):
    if value is None or type(value) in (bool, int, float, str):
        return [type(value).__name__, value]
    if isinstance(value, (torch.dtype, torch.device)):
        return [type(value).__name__, str(value)]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_default_signature(item) for item in value]]
    raise ValueError("Frontend default argument needs an explicit immutable signature")


def _default_value(artifact, arguments, name):
    return eval(name, {"L": artifact.prepare_f_locals(*arguments), "G": artifact.fn.__globals__})


def _capture_defaults(artifact, arguments):
    from torch._dynamo.source import DefaultsSource

    guards = _source_loads(artifact._artifacts.guards_state)
    result = {}
    for guard in guards.output_graph.guards:
        source = guard.originating_source
        while source is not None:
            if isinstance(source, DefaultsSource):
                name = source.name
                result[name] = dict(signature=_default_signature(_default_value(artifact, arguments, name)),
                                    base=source.base.name,
                                    key=source.idx_key,
                                    keyword=source.is_kw)
            source = getattr(source, "base", None)
    return result


def _capture_tensor_attributes(artifact, arguments):
    from torch._dynamo.source import AttrSource

    guards = _source_loads(artifact._artifacts.guards_state)
    result = {}
    for guard in guards.output_graph.guards:
        source = guard.originating_source
        while source is not None:
            if isinstance(source, AttrSource):
                base = _default_value(artifact, arguments, source.base.name)
                if isinstance(base, torch.Tensor) and source.member in vars(base):
                    result[source.name] = dict(base=source.base.name,
                                               member=source.member,
                                               signature=_default_signature(getattr(base, source.member)))
            source = getattr(source, "base", None)
    return result


def _owner_scalar_guards(guards):
    """Index existing equality guards without loading executable frontends."""
    saved = {"L": guards.output_graph.local_scope, "G": guards.output_graph.global_scope}
    result = {}
    for guard in guards.output_graph.guards:
        create = getattr(guard.create_fn, "func", guard.create_fn)
        if getattr(create, "__name__", "") not in ("EQUALS_MATCH", "CONSTANT_MATCH", "BOOL_MATCH"):
            continue
        source = guard.originating_source
        if source is None or not source.name.startswith("L['self']"):
            continue
        try:
            result[source.name] = _default_signature(eval(source.name, saved))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return result


def _owner_scalars_match(owner, globals_, expected):
    scope = {"L": {"self": owner}, "G": globals_}
    try:
        # Search geometry rejects incompatible variants before their many
        # unchanged model/configuration scalar guards need evaluation.
        ordered = sorted(expected.items(), key=lambda pair: not pair[0].endswith(".search_length"))
        return all(_default_signature(eval(name, scope)) == value for name, value in ordered)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _defaults_match(artifact, arguments):
    try:
        return all(
            _default_signature(_default_value(artifact, arguments, name)) == expected["signature"]
            for name, expected in (*artifact._frontend_defaults.items(), *artifact._frontend_tensor_attributes.items()))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _signature_value(signature):
    kind, value = signature
    if kind in ("NoneType", "bool", "int", "float", "str"):
        return value
    if kind == "dtype":
        return getattr(torch, value.removeprefix("torch."))
    if kind == "device":
        return torch.device(value)
    if kind in ("tuple", "list"):
        items = [_signature_value(item) for item in value]
        return tuple(items) if kind == "tuple" else items
    raise ValueError("Unsupported saved default argument signature")


def _restore_guard_defaults(guards, defaults, owner, globals_):
    saved = {"L": guards.output_graph.local_scope, "G": guards.output_graph.global_scope}
    current = {"L": {"self": owner}, "G": globals_}
    for record in defaults.values():
        if not record["keyword"]:
            continue
        function = eval(record["base"], saved)
        if (not isinstance(function, FunctionType)
                or function.__kwdefaults__ is not None and record["key"] in function.__kwdefaults__):
            continue
        if function is eval(record["base"], current):
            raise ValueError("Current function definition lacks its captured keyword defaults")
        # Torch's nested-function reducer omits __kwdefaults__. This function
        # belongs only to the loaded guard snapshot, never the live model.
        if function.__kwdefaults__ is None:
            function.__kwdefaults__ = {}
        function.__kwdefaults__[record["key"]] = _signature_value(record["signature"])


def _restore_guard_tensor_attributes(guards, attributes):
    from torch._subclasses.fake_tensor import FakeTensor

    saved = {"L": guards.output_graph.local_scope, "G": guards.output_graph.global_scope}
    for record in attributes.values():
        tensor = eval(record["base"], saved)
        if not isinstance(tensor, FakeTensor):
            raise ValueError("Serialized tensor guard must own metadata only")
        if record["member"] not in vars(tensor):
            # Torch's FakeTensor serialization retains shape/stride, but drops
            # custom scalar layout qualifications. Repair only its guard copy.
            setattr(tensor, record["member"], _signature_value(record["signature"]))


@dataclasses.dataclass
class _GuardedCompiledFunction(AOTCompiledFunction):
    _frontend_defaults: dict = dataclasses.field(default_factory=dict)
    _frontend_tensor_attributes: dict = dataclasses.field(default_factory=dict)
    _frontend_owner: object = None

    @classmethod
    def deserialize(cls, data, globals_, bindings, defaults, attributes):
        from torch._dynamo.aot_compile import AOTCompileUnpickler, CompileArtifacts
        from torch._dynamo.package import SerializedCode

        state = AOTCompileUnpickler(bindings, io.BytesIO(data)).load()
        state["runtime_env"] = dataclasses.replace(state["runtime_env"],
                                                   bytecode=SerializedCode.to_code_object(
                                                       state["runtime_env"].bytecode))
        deserializer, compiled = state["compiled_fn"]
        state["compiled_fn"] = deserializer(compiled)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])
        return cls(CompileArtifacts(**state),
                   _extra_globals=globals_,
                   _frontend_defaults=defaults,
                   _frontend_tensor_attributes=attributes,
                   _frontend_owner=bindings["owner"])

    def __post_init__(self):
        from torch._dynamo.package import load_guard_manager

        self._artifacts.check_compatibility()
        self.fn = self._artifacts.runtime_env.forward_callable(self._artifacts.backend_id,
                                                               self._artifacts.compiled_fn,
                                                               extra_globals=self._extra_globals)
        if self._artifacts.guard_manager is None:
            # Keep Torch's tensor/alias reducers intact. Only the cached Source
            # constructor mismatch is repaired when loading this local artifact.
            guards = _source_loads(self._artifacts.guards_state)
            _restore_guard_defaults(guards, self._frontend_defaults, self._frontend_owner, self.fn.__globals__)
            _restore_guard_tensor_attributes(guards, self._frontend_tensor_attributes)
            self._artifacts.guard_manager = load_guard_manager(guards, self._artifacts.original_code,
                                                               self.fn.__globals__)


class ReboundFrontend(SerializableCallable):
    """Rebuild the unchanged backend against this process's actual inputs."""

    def __init__(self, graph_bytes):
        self.graph_bytes = graph_bytes
        self.backend = None
        self.compiled = None
        self.backend_seconds = 0.0
        self.backend_directory = None

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
        sources = {
            node.name: node._dynamo_source
            for node in graph.graph.nodes if node.op == "placeholder" and hasattr(node, "_dynamo_source")
        }
        data = GraphModuleSerializableCallable.serialize_compile_artifacts(GraphModuleSerializableCallable(graph))
        # GraphPickler retains node.meta but not Node's custom attributes.
        # AOTDispatch requires the original placeholder sources to preserve
        # static weight classification and argument/alias guard ownership.
        return cls(_source_dumps((data, sources)))

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        return fn.graph_bytes

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        return cls(data)

    def bind(self, backend, *, directory=None):
        if self.compiled is not None or self.backend is not None:
            raise RuntimeError("A restored frontend backend can only be bound once")
        self.backend = backend
        self.backend_directory = directory

    def __call__(self, *args):
        if self.backend is None:
            raise RuntimeError("Frontend backend must be rebound before execution")
        if self.compiled is None:

            def restore_graph():
                data, sources = _source_loads(self.graph_bytes)
                graph = GraphModuleSerializableCallable.deserialize_compile_artifacts(data).graph_module
                for node in graph.graph.nodes:
                    if node.name in sources:
                        node._dynamo_source = sources[node.name]
                return graph

            started = time.perf_counter()
            if self.backend_directory is not None and os.environ.get("VLLM_HPU_DSV41_BACKEND_CACHE", "1") == "1":
                from vllm_gaudi.compilation.deepseek_v41_backend_cache import restore_or_compile
                from vllm_gaudi.compilation.deepseek_v41_cache_identity import lowered_keys

                key, compatible = lowered_keys(self.graph_bytes)
                self.compiled = restore_or_compile(self.backend,
                                                   None,
                                                   list(args),
                                                   self.backend_directory,
                                                   key,
                                                   compatible=compatible,
                                                   graph_factory=restore_graph)
            else:
                self.compiled = self.backend(restore_graph(), list(args))
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


def _input_signature(value):
    if isinstance(value, torch.Tensor):
        return [
            "tensor",
            list(value.shape),
            list(value.stride()),
            str(value.dtype),
            str(value.device), value.requires_grad
        ]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_input_signature(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[_default_signature(key), _input_signature(item)] for key, item in value.items()]]
    return _default_signature(value)


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
        self.pending = []
        self.stats = dict(captures=0, restores=0, hits=0, capture_seconds=0.0, restore_seconds=0.0)
        self.guard_miss_reported = False
        if self.directory.exists():
            self._index()

    def _index(self):
        manifests = sorted(self.directory.glob("*.json"))
        for path in manifests:
            try:
                record = json.loads(path.read_text())
                if record["identity"] != self.identity or record["schema"] != 3:
                    raise ValueError("Frontend dependency fingerprint changed")
                binary = path.with_suffix(".bin")
                if binary.stat().st_size > 16 << 20:
                    raise ValueError("Frontend artifact exceeds the metadata-only size bound")
            except (ValueError, KeyError, OSError) as error:
                self._reject(path, error)
                continue
            self.pending.append((binary, record))
        if len(self.pending) > self.max_variants:
            raise ValueError("Frontend cache contains too many variants")

    def _reject(self, manifest, error):
        from vllm_gaudi.extension.logger import logger

        logger().warning("V4.1 frontend artifact rejected; rebuilding affected entry: %s", error)
        suffix = f".rejected-{uuid.uuid4().hex}"
        for path in (manifest, manifest.with_suffix(".bin")):
            if path.exists():
                path.rename(path.with_name(path.name + suffix))

    def _restore(self, args):
        signature = _input_signature(args)
        for binary, record in tuple(self.pending):
            if record["inputs"] != signature or not _owner_scalars_match(self.owner, self.function.__globals__,
                                                                         record.get("owner_scalars", {})):
                continue
            started = time.perf_counter()
            try:
                data = binary.read_bytes()
                if hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("Frontend artifact digest differs from its publication")
                artifact = _GuardedCompiledFunction.deserialize(data, self.function.__globals__,
                                                                owner_bindings(self.owner), record["defaults"],
                                                                record["tensor_attributes"])
            except (ValueError, KeyError, OSError, EOFError, ImportError, AttributeError, TypeError,
                    pickle.UnpicklingError) as error:
                self.pending.remove((binary, record))
                self._reject(binary.with_suffix(".json"), error)
                continue
            artifact._artifacts.compiled_fn.bind(self.backend, directory=self.directory / "lowered")
            artifact._frontend_input_signature = record["inputs"]
            artifact._frontend_owner_scalars = record.get("owner_scalars", {})
            self.variants.append(artifact)
            self.pending.remove((binary, record))
            self.stats["restores"] += 1
            self.stats["restore_seconds"] += time.perf_counter() - started
            from vllm_gaudi.extension.logger import logger

            logger().info("V4.1 guarded frontend restored: entry=%s seconds=%.3f", self.function.__qualname__,
                          time.perf_counter() - started)
            # Deserialize one candidate at a time. A compatible guard ends
            # lookup immediately; other finite variants remain indexed for
            # their actual input/state contract instead of being rebuilt on
            # every newly created stage owner.
            yield artifact

    def __call__(self, *args):
        from torch._dynamo.aot_compile import AOTCompiledFunction, aot_compile_fullgraph
        from torch._dynamo.hooks import Hooks

        arguments = (self.owner, *args)
        # Snapshot existing variants: restoration appends to the live list.
        # Check already-loaded guards before reading any further artifacts.
        for artifact in chain(tuple(self.variants), self._restore(args)):
            defaults_match = _defaults_match(artifact, arguments)
            if defaults_match and artifact.guard_check(*arguments):
                self.stats["hits"] += 1
                return artifact(*arguments)
            same_geometry = (artifact._frontend_input_signature == _input_signature(args) and _owner_scalars_match(
                self.owner, self.function.__globals__, artifact._frontend_owner_scalars))
            if same_geometry and not self.guard_miss_reported:
                from vllm_gaudi.extension.logger import logger

                reason = (artifact._artifacts.guard_manager.check_verbose(artifact.prepare_f_locals(
                    *arguments)) if defaults_match else "callable default or tensor qualification differs")
                logger().warning("V4.1 frontend guard rejected: entry=%s reason=%s", self.function.__qualname__, reason)
                self.guard_miss_reported = True
        if len(self.variants) + len(self.pending) == self.max_variants:
            raise RuntimeError("Frontend variant capacity exhausted before execution")
        started = time.perf_counter()
        artifact = aot_compile_fullgraph(self.function, (arguments, {}),
                                         Hooks(),
                                         ReboundFrontend.from_graph,
                                         dynamic=False)
        artifact._frontend_defaults = _capture_defaults(artifact, arguments)
        artifact._frontend_tensor_attributes = _capture_tensor_attributes(artifact, arguments)
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
        record = dict(schema=3,
                      identity=self.identity,
                      sha256=hashlib.sha256(data).hexdigest(),
                      defaults=artifact._frontend_defaults,
                      tensor_attributes=artifact._frontend_tensor_attributes,
                      owner_scalars=_owner_scalar_guards(_source_loads(artifact._artifacts.guards_state)),
                      inputs=_input_signature(args))
        artifact._frontend_input_signature = record["inputs"]
        artifact._frontend_owner_scalars = record["owner_scalars"]
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(binary.with_suffix(".json"))
        artifact._artifacts.compiled_fn.bind(self.backend, directory=self.directory / "lowered")
        self.variants.append(artifact)
        self.stats["captures"] += 1
        from vllm_gaudi.extension.logger import logger

        logger().info("V4.1 guarded frontend captured: entry=%s seconds=%.3f", self.function.__qualname__,
                      time.perf_counter() - started)
        return artifact(*arguments)


class TensorCallableOwner(torch.nn.Module):
    """Bind a pure tensor region or method to the current process's owner."""

    def __init__(self, function, owner=None):
        super().__init__()
        self.function = function
        self.method_owner = owner

    def forward(self, arguments, keywords):
        if self.method_owner is None:
            return self.function(*arguments, **keywords)
        return self.function(self.method_owner, *arguments, **keywords)


def _contract_directory(directory, contract):
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import frontend_contract_keys

    current, compatible = frontend_contract_keys(contract)
    directory = Path(directory)
    for identity in (current, *compatible):
        root = directory / identity
        if root.is_dir():
            return root, identity
    return directory / current, current


def cached_tensor_entry(function, arguments, keywords, *, owner=None):
    """Persist the existing per-contract prefill/protocol compilation boundary."""
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import computation_dependencies, runtime_content_identity
    from torch._dynamo.backends.registry import lookup_backend

    directory = os.environ.get("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR")
    if not directory:
        return None
    identity = os.environ.get("DSV41_SERVING_COMPILE_IDENTITY")
    profile = os.environ.get("DSV41_RUNTIME_PROFILE")
    runtime = (runtime_content_identity(json.loads(Path(profile).read_text()))
               if profile else os.environ.get("DSV41_SERVING_RUNTIME"))
    if not identity or not runtime:
        raise RuntimeError("Tensor region reuse requires the validated serving identity")
    proxy = TensorCallableOwner(function, owner)
    contract = dict(sources=computation_dependencies(function, proxy),
                    runtime=runtime,
                    model=identity,
                    arguments=_input_signature((arguments, keywords)),
                    torch_version=torch.__version__)
    root, key = _contract_directory(Path(directory) / "tensor-regions", contract)
    entry = GuardedFrontendEntry(TensorCallableOwner.forward,
                                 proxy,
                                 lookup_backend("hpu_backend"),
                                 root / f"rank{cache_rank()}",
                                 identity=key)

    def call(*args, **kwargs):
        return entry(args, kwargs)

    return call


@lru_cache(maxsize=4)
def immutable_package_sources(package):
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }


def cached_group_entry(function, group, backend, directory, *, native, shared_coordinates, memory_ready):
    """Bind one common TP stage entry; topology remains an ordinary contract."""
    from vllm_gaudi.extension.logger import logger
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import (computation_dependencies, runtime_content_identity,
                                                                    semantic_environment)

    sources = computation_dependencies(function, group)
    if isinstance(backend, str):
        from torch._dynamo.backends.registry import lookup_backend

        backend = lookup_backend(backend)
    environment = semantic_environment(os.environ)
    contract = dict(sources=sources,
                    torch_version=torch.__version__,
                    environment=environment,
                    runtime=(runtime_content_identity(json.loads(Path(os.environ["DSV41_RUNTIME_PROFILE"]).read_text()))
                             if os.environ.get("DSV41_RUNTIME_PROFILE") else os.environ.get("DSV41_SERVING_RUNTIME")),
                    serving_identity=os.environ.get("DSV41_SERVING_COMPILE_IDENTITY"),
                    function=function.__qualname__,
                    layers=[layer.layer for layer in group.layers],
                    native=native,
                    shared_coordinates=shared_coordinates,
                    memory_ready=memory_ready)
    if not contract["runtime"] or not contract["serving_identity"]:
        raise RuntimeError("Persistent frontend reuse requires the validated serving runtime and model identity")
    root, identity = _contract_directory(directory, contract)
    result = GuardedFrontendEntry(function, group, backend, root / f"rank{cache_rank()}", identity=identity)
    logger().info("V4.1 guarded frontend indexed: layers=%s cached_variants=%d", contract["layers"],
                  len(result.pending))
    return result
