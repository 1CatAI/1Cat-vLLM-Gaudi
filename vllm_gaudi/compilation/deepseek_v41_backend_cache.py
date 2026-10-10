# SPDX-License-Identifier: Apache-2.0
"""Persist lowered inference callables without process-local recipe handles."""
import copy
import hashlib
import io
import importlib
import json
from pathlib import Path
import time
import uuid
from types import ModuleType

import cloudpickle
import torch
from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable


def _restore_graph(data):
    graph = GraphModuleSerializableCallable.deserialize_compile_artifacts(data).graph_module
    for module in graph.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if node.meta.pop("dsv41_boxed_list_clear", False):
                    node.op, node.target = "call_function", list_clear
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
            data["jit_ir"] = jit.createFromUpstreamGraph(torch._C.parse_ir(data["jit_ir"]))
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

    return dict(jit_ir=module.graph_str_repr_with_source_info,
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
    if any(True for _ in result.named_buffers()):
        raise ValueError("Lowered tensor constants need explicit current-process binding")
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


def serialize_callable(function):
    buffer = io.BytesIO()
    _CallablePickler(buffer, protocol=5).dump(function)
    return buffer.getvalue()


def restore_or_compile(backend, graph, arguments, directory, identity):
    """Cache only validated metadata; native capture still binds current state."""
    from vllm_gaudi.extension.logger import logger

    directory = Path(directory)
    binary, manifest = directory / f"{identity}.bin", directory / f"{identity}.json"
    started = time.perf_counter()
    if manifest.exists() and binary.exists():
        try:
            record, data = json.loads(manifest.read_text()), binary.read_bytes()
            if (record["schema"] != 1 or record["identity"] != identity or len(data) > 32 << 20
                    or hashlib.sha256(data).hexdigest() != record["sha256"]):
                raise ValueError("Lowered callable certificate mismatch")
            function = cloudpickle.loads(data)
            logger().info("V4.1 lowered backend restored: seconds=%.3f", time.perf_counter() - started)
            return function
        except (ValueError, KeyError, EOFError, ImportError, TypeError) as error:
            logger().warning("V4.1 lowered backend rejected: %s", error)
    function = backend(graph, arguments)
    try:
        data = serialize_callable(function)
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
