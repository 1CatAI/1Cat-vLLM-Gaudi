# SPDX-License-Identifier: Apache-2.0
"""Content identities for computation, independent of deployment locations."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from functools import lru_cache
from types import FunctionType, MethodType, ModuleType


@lru_cache(maxsize=512)
def normalized_source(value):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(value))), include_attributes=False)


def computation_dependencies(function, owner):
    """Include actual module types, bound callbacks and their Python helpers."""
    pending = [function]
    modules = list(owner.modules()) if hasattr(owner, "modules") else [owner]
    for module in modules:
        if type(module).__module__.startswith("vllm_gaudi."):
            pending.append(type(module))
        pending.extend(value.__func__ if isinstance(value, MethodType) else value for value in vars(module).values()
                       if isinstance(value, (FunctionType, MethodType)))
    result = {}
    while pending:
        value = pending.pop()
        name = f"{value.__module__}:{value.__qualname__}"
        if name in result:
            continue
        source = normalized_source(value)
        result[name] = hashlib.sha256(source.encode()).hexdigest()
        defining = inspect.getmodule(value)
        if defining is None:
            continue
        names = {
            node.id
            for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(value)))) if isinstance(node, ast.Name)
        }
        for used in names:
            dependency = vars(defining).get(used)
            if isinstance(dependency, (FunctionType, type)) and dependency.__module__.startswith("vllm_gaudi."):
                pending.append(dependency)
            elif isinstance(dependency, ModuleType) and dependency.__name__.startswith("vllm_gaudi."):
                path = Path(dependency.__file__)
                result[f"module:{dependency.__name__}"] = hashlib.sha256(
                    ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()).hexdigest()
            elif type(dependency) in (bool, int, float, str, tuple) and used.isupper():
                result[f"constant:{defining.__name__}:{used}"] = hashlib.sha256(repr(dependency).encode()).hexdigest()
        # Deferred imports must be covered even when the chosen numerical
        # branch has not imported their module in this process yet.
        package = Path(__file__).resolve().parents[1]
        tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vllm_gaudi."):
                path = package / (node.module.removeprefix("vllm_gaudi.").replace(".", "/") + ".py")
                if path.is_file():
                    result[f"module:{node.module}"] = hashlib.sha256(
                        ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()).hexdigest()
    return result


def relocated_content(value):
    """Keep content fingerprints while removing machine-specific path spelling."""
    if isinstance(value, dict):
        return {
            (Path(key).name if str(key).startswith("/") else str(key)): relocated_content(item)
            for key, item in value.items() if key != "path" or "sha256" not in value
        }
    if isinstance(value, list):
        return [relocated_content(item) for item in value]
    return value


def stable_serving_contract(model, runtime_identity, arguments):
    return dict(schema=2,
                model=file_content(Path(model) / "manifest.json"),
                runtime=runtime_identity,
                arguments=arguments)


@lru_cache(maxsize=128)
def _file_content(path, device, inode, size, modified):
    path = Path(path)
    if path.suffix == ".json":
        data = json.dumps(relocated_content(json.loads(path.read_text())), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_content(path):
    stat = Path(path).stat()
    return _file_content(str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def runtime_content_identity(profile):
    environment = semantic_environment(profile["environment"])
    records = {}
    for kind in ("additional_libraries", "configuration_files"):
        records[kind] = [(Path(item["path"]).name, file_content(item["path"])) for item in profile.get(kind, ())]
    return hashlib.sha256(json.dumps(dict(environment=environment, records=records),
                                     sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def lowering_dependencies():
    """Backend transformations invalidate lowered artifacts, not frontend captures."""
    import habana_frameworks.torch.dynamo.compile_backend as compiler
    import torch

    package = Path(compiler.__file__).parent
    result = {"torch_version": torch.__version__}
    for path in sorted(package.rglob("*.py")):
        result[f"bridge:{path.relative_to(package)}"] = hashlib.sha256(
            ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()).hexdigest()
    plugin = Path(__file__).resolve().parents[1]
    for name in ("ops/tp2_prepared_plan.py", "compilation/deepseek_v41_prepared.py",
                 "compilation/deepseek_v41_backend_cache.py"):
        result[name] = hashlib.sha256(
            ast.dump(ast.parse((plugin / name).read_text()), include_attributes=False).encode()).hexdigest()
    return result


def semantic_environment(environment):
    ignored = ("TMPDIR", "HABANA_LOGS", "VLLM_HPU_DSV4_WORKER_CPUS", "VLLM_HPU_DSV4_WORKER_HELPER_CPUS",
               "VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", "PT_HPU_RECIPE_CACHE_CONFIG")
    result = {}
    for key, value in environment.items():
        if key in ignored or not key.startswith(("VLLM_HPU_", "PT_HPU_", "HCCL_", "HCL_")):
            continue
        if isinstance(value, str) and value.startswith("/"):
            path = Path(value)
            manifest = path / "manifest.json" if path.is_dir() else path
            if manifest.is_file():
                result[key] = file_content(manifest)
            else:
                # Runtime binary directories are separately content-certified
                # by the serving runtime profile and native build manifest.
                result[key] = "runtime-content-bound"
        else:
            result[key] = value
    return result
