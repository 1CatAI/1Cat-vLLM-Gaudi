# SPDX-License-Identifier: Apache-2.0
"""Persist lowered inference callables without process-local recipe handles."""
import copy
from contextvars import ContextVar
import hashlib
import io
import importlib
import json
import pickle
from pathlib import Path
import time
import uuid
import re
from types import ModuleType

import cloudpickle
import torch
from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable

_literal_inputs = ContextVar("lowered_literal_inputs", default=())


def frontend_literals(graph):
    """Only literal constructors/arguments may certify folded tensor constants."""
    result = []
    if graph is None:
        return result
    scalars = set()

    def visit(value):
        if type(value) in (bool, int, float):
            scalars.add((type(value), value))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for node in graph.graph.nodes:
        visit(node.args)
        visit(node.kwargs)
        if node.op != "call_function" or node.target is not torch.tensor or not node.args:
            continue
        try:

            def scalar_literal(value):
                return (type(value) in (bool, int, float)
                        or isinstance(value, (tuple, list)) and all(scalar_literal(item) for item in value))

            if not scalar_literal(node.args[0]):
                continue
            value = torch.tensor(node.args[0], dtype=node.kwargs.get("dtype"), device="cpu")
            if value.numel() <= 4096:
                result.append(value)
        except (TypeError, ValueError):
            continue
    for kind, value in scalars:
        result.append(torch.tensor(value, device="cpu"))
        if kind is float:
            result.append(torch.tensor(value, dtype=torch.float64, device="cpu"))
    return result


def literal_record(value, candidates):
    matching = [item for item in candidates if item.shape == value.shape and item.dtype == value.dtype]
    if not matching:
        raise ValueError("Lowered tensor constant has no explicit frontend literal certificate")
    actual = value.detach().cpu()
    if not any(torch.equal(actual, expected) for expected in matching):
        raise ValueError("Lowered tensor constant differs from its frontend literal")
    return dict(values=actual.tolist(),
                shape=list(value.shape),
                dtype=str(value.dtype).removeprefix("torch."),
                device=str(value.device))


def restore_literal(record):
    return torch.tensor(record["values"], dtype=getattr(torch, record["dtype"]),
                        device=record["device"]).reshape(record["shape"])


def portable_jit(graph, candidates):
    records = []
    text = graph.str(print_source_info=True)
    for node in graph.nodes():
        if node.kind() == "prim::Constant" and "value" in node.attributeNames() and node.kindOf("value") == "t":
            records.append(literal_record(node.t("value"), candidates))
            pattern = (r"(%" + re.escape(node.output().debugName()) +
                       r"\s*:[^\n]*?=\s*prim::Constant)\[value=.*?\]\(\)")
            text, replaced = re.subn(pattern, r"\1[value=<Tensor>]()", text, flags=re.S)
            if replaced != 1:
                raise ValueError("JIT literal metadata does not cover its tensor constant")
    return text, records


def restore_jit(text, records):
    graph = torch._C.parse_ir(text, parse_tensor_constants=bool(records))
    tensors = [
        node for node in graph.nodes()
        if node.kind() == "prim::Constant" and "value" in node.attributeNames() and node.kindOf("value") == "t"
    ]
    if len(tensors) != len(records):
        raise ValueError("JIT tensor constant count differs from its certificate")
    for node, record in zip(tensors, records):
        node.t_("value", restore_literal(record))
    return graph


def _restore_graph(data):
    graph = GraphModuleSerializableCallable.deserialize_compile_artifacts(data).graph_module
    for module in graph.modules():
        if isinstance(module, torch.fx.GraphModule):
            for name, record in module.meta.pop("dsv41_literal_buffers", {}).items():
                module._buffers[name] = restore_literal(record)
            changed = False
            for node in module.graph.nodes:
                if node.meta.pop("dsv41_boxed_list_clear", False):
                    node.op, node.target = "call_function", list_clear
                    changed = True
            # GraphPickler already reconstructs the executable forward. Buffer
            # rebinding does not change that code; only edited nodes need it.
            if changed:
                module.recompile()
    return graph


def list_clear(values):
    """Portable equivalent of the Bridge's locally defined boxed-input lambda."""
    values.clear()


def _restore_fake(data):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx._graph_pickler import GraphPickler
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return GraphPickler.loads(data, FakeTensorMode(shape_env=ShapeEnv()))


class DeferredRecipe(torch.nn.Module):
    """Immutable recipe description; construct the Bridge owner on first use."""

    def __init__(self, description):
        super().__init__()
        self.description = description
        self.runtime = None

    def forward(self, *args):
        return self.materialize()(*args)

    def materialize(self):
        if self.runtime is None:
            from habana_frameworks.torch.dynamo.compile_backend.recipe_compiler import HabanaGraphModule
            from habana_frameworks.torch._torch_jit_C import jit

            data = dict(self.description)
            data["jit_ir"] = jit.createFromUpstreamGraph(restore_jit(data["jit_ir"], data.pop("jit_literals", ())))
            data["graph_module"] = _restore_graph(data["graph_module"])
            self.runtime = HabanaGraphModule(**data)
        return self.runtime


class DeferredPreparedGroup(torch.nn.Module):
    """Restore the prepared operation sequence with fresh communicator ownership."""

    def __init__(self, original, tp4):
        super().__init__()
        self.original_bytes, self.tp4 = original, tp4
        self.runtime = None

    def forward(self, inputs):
        if self.runtime is None:
            from vllm_gaudi.ops.tp2_prepared_plan import PreparedGroupModule

            original = _restore_graph(self.original_bytes)
            for name, child in tuple(original.named_children()):
                if isinstance(child, DeferredRecipe):
                    original._modules[name] = child.materialize()
            self.runtime = PreparedGroupModule(original, tp4=self.tp4)
        return self.runtime(inputs)


def _recipe_description(module):
    graph = _portable_graph(module.fx_module)
    # HabanaGraphModule has already removed duplicated outputs from its
    # metadata. Restore the constructor's original output list instead.
    from habana_frameworks.torch.dynamo.compile_backend.recipe_compiler import get_outputs_metadata

    jit_ir, jit_literals = portable_jit(module._jit_ir, _literal_inputs.get())
    return dict(jit_ir=jit_ir,
                jit_literals=jit_literals,
                graph_module=GraphModuleSerializableCallable.serialize_compile_artifacts(
                    GraphModuleSerializableCallable(graph)),
                parent_graph_name=module.name,
                outputs_metadata=get_outputs_metadata(graph),
                symbolic_metadata={},
                pholder_symbolic_dict={},
                const_input_indexes=module._const_input_indexes,
                is_training=not module.is_inference,
                dynamic=False,
                force_static_compile=module._force_static_compile,
                has_random_ops=module._has_randoms,
                is_reusables=module.is_reusables)


def _portable_graph(module):
    from habana_frameworks.torch.dynamo.compile_backend.recipe_compiler import HabanaGraphModule
    from vllm_gaudi.ops.tp2_prepared_plan import PreparedGroupModule

    result = torch.fx.GraphModule(module, copy.deepcopy(module.graph))
    result.meta = dict(module.meta)
    for node in result.graph.nodes:
        if (node.op == "call_function"
                and getattr(node.target, "__qualname__", "") == "pass_make_boxed_graph.<locals>.<lambda>"
                and getattr(node.target, "__module__", "") == "habana_frameworks.torch.dynamo.compile_backend.passes"):
            # FX's metadata pickler cannot encode locally defined functions.
            # A reversible method representation preserves the exact boxed
            # list lifetime operation without modifying the live graph.
            node.op, node.target = "call_method", "clear"
            node.meta["dsv41_boxed_list_clear"] = True
    result.recompile()
    for name, child in module.named_children():
        if isinstance(child, HabanaGraphModule):
            if child.is_dynamic or not child.is_inference:
                raise ValueError("Persistent lowered recipes require static inference")
            result._modules[name] = DeferredRecipe(_recipe_description(child))
        elif isinstance(child, torch.fx.GraphModule):
            result._modules[name] = _portable_graph(child)
        elif isinstance(child, PreparedGroupModule):
            original = GraphModuleSerializableCallable.serialize_compile_artifacts(
                GraphModuleSerializableCallable(_portable_graph(child.original)))
            result._modules[name] = DeferredPreparedGroup(original, child.tp4)
        else:
            raise ValueError(f"Unqualified lowered child: {type(child).__name__}")
    if any(True for _ in result.named_parameters()):
        raise ValueError("Lowered artifacts must receive current model parameters as inputs")
    literals = {}
    for name, value in tuple(result._buffers.items()):
        if value is None or not re.fullmatch(r"_tensor_constant\d+", name):
            continue
        candidates = [item for item in _literal_inputs.get() if item.shape == value.shape and item.dtype == value.dtype]
        if not candidates:
            continue
        actual = value.detach().cpu()
        if any(torch.equal(actual, expected) for expected in candidates):
            literals[name] = dict(values=actual.tolist(),
                                  shape=list(value.shape),
                                  dtype=str(value.dtype).removeprefix("torch."),
                                  device=str(value.device))
            # Preserve the get_attr binding in FX without persisting its
            # allocation. Restore a fresh literal tensor before recipe use.
            result._buffers[name] = None
    if literals:
        result.meta["dsv41_literal_buffers"] = literals
    buffers = [(name, tuple(value.shape), str(value.dtype)) for name, value in result.named_buffers()]
    if buffers:
        raise ValueError(f"Lowered tensor constants need explicit current-process binding: {buffers}")
    return result


class _CallablePickler(cloudpickle.CloudPickler):

    def reducer_override(self, value):
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx._graph_pickler import GraphPickler, Options

        if isinstance(value, ModuleType):
            if importlib.import_module(value.__name__) is not value:
                raise ValueError("Lowered callable references an unregistered configuration module")
            return importlib.import_module, (value.__name__, )
        if isinstance(value, FakeTensor):
            return _restore_fake, (GraphPickler.dumps(value, Options(ops_filter=None)), )
        if isinstance(value, torch.Tensor):
            raise ValueError("Lowered callable cannot retain concrete model or state tensors")
        if isinstance(value, torch.fx.GraphModule):
            data = GraphModuleSerializableCallable.serialize_compile_artifacts(
                GraphModuleSerializableCallable(_portable_graph(value)))
            return _restore_graph, (data, )
        return super().reducer_override(value)


def serialize_callable(function, *, literals=()):
    buffer = io.BytesIO()
    token = _literal_inputs.set(tuple(literals))
    try:
        _CallablePickler(buffer, protocol=5).dump(function)
    finally:
        _literal_inputs.reset(token)
    return buffer.getvalue()


def restore_or_compile(backend, graph, arguments, directory, identity, *, compatible=(), graph_factory=None):
    """Cache only validated metadata; native capture still binds current state."""
    from vllm_gaudi.extension.logger import logger

    directory = Path(directory)
    binary, manifest = directory / f"{identity}.bin", directory / f"{identity}.json"
    started = time.perf_counter()
    for candidate in (identity, *compatible):
        saved_binary, saved_manifest = directory / f"{candidate}.bin", directory / f"{candidate}.json"
        if not saved_manifest.exists() or not saved_binary.exists():
            continue
        try:
            record, data = json.loads(saved_manifest.read_text()), saved_binary.read_bytes()
            if (record["schema"] != 1 or record["identity"] != candidate or len(data) > 32 << 20
                    or hashlib.sha256(data).hexdigest() != record["sha256"]):
                raise ValueError("Lowered callable certificate mismatch")
            function = cloudpickle.loads(data)
            logger().info("V4.1 lowered backend restored: seconds=%.3f", time.perf_counter() - started)
            return function
        except (ValueError, KeyError, EOFError, ImportError, TypeError, AttributeError, OSError,
                pickle.UnpicklingError) as error:
            logger().warning("V4.1 lowered backend rejected: %s", error)
    if graph_factory is not None:
        graph = graph_factory()
    function = backend(graph, arguments)
    try:
        data = serialize_callable(function, literals=frontend_literals(graph))
        if len(data) > 32 << 20:
            raise ValueError("Lowered callable exceeds metadata budget")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(data)
        temporary.replace(binary)
        temporary.write_text(
            json.dumps(dict(schema=1, identity=identity, sha256=hashlib.sha256(data).hexdigest())) + "\n")
        temporary.replace(manifest)
        logger().info("V4.1 lowered backend published: seconds=%.3f", time.perf_counter() - started)
    except (ValueError, TypeError, AttributeError, RuntimeError) as error:
        logger().warning("V4.1 lowered backend not cacheable: %s", error)
    return function
