# SPDX-License-Identifier: Apache-2.0
"""Content identities for computation, independent of deployment locations."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import textwrap
from functools import lru_cache
from types import FunctionType, MethodType, ModuleType


@lru_cache(maxsize=512)
def _source_tree(value):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
    except SyntaxError:
        if not isinstance(value, FunctionType) or value.__code__.co_name != "<lambda>":
            raise
        # inspect can return only the final line of a multiline call, with
        # its closing parenthesis. Read the lambda from the complete source;
        # never discard an unparseable computation dependency.
        path = Path(inspect.getsourcefile(value))
        matches = [
            node for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Lambda) and node.lineno == value.__code__.co_firstlineno
        ]
        if len(matches) != 1:
            raise ValueError("Computation dependency has an ambiguous lambda source") from None
        tree = ast.Expression(matches[0])
    return tree


@lru_cache(maxsize=512)
def normalized_source(value):
    return ast.dump(_source_tree(value), include_attributes=False)


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
        names = {node.id for node in ast.walk(_source_tree(value)) if isinstance(node, ast.Name)}
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
        tree = _source_tree(value)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vllm_gaudi."):
                path = package / (node.module.removeprefix("vllm_gaudi.").replace(".", "/") + ".py")
                if path.is_file():
                    result[f"module:{node.module}"] = hashlib.sha256(
                        ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()).hexdigest()
    return result


def frontend_contract_keys(contract):
    """Separate metadata lookup policy from the guarded computation identity.

    TensorCallableOwner and the actual entry/functions remain source-bound.
    The cache reader's module is transport; its schema validates artifacts.
    Read the preceding schema's namespace only when all other computation,
    model, runtime, environment and layout dependencies still match.
    """
    sources = dict(contract["sources"])
    transport = "module:vllm_gaudi.compilation.deepseek_v41_frontend_cache"
    had_transport = transport in sources
    sources.pop(transport, None)
    canonical = dict(contract, sources=sources)

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    current = digest(canonical)
    if not had_transport:
        return current, ()
    previous = dict(sources)
    previous[transport] = "9f120732e0046d5e193fbc6528ef674522d70b0b6599ca30791ee644f6b16e6e"
    return current, (digest(dict(contract, sources=previous)), )


def relocated_content(value):
    """Keep content fingerprints while removing machine-specific path spelling."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            certificate = ("sha256" if key == "path" else key.removesuffix("_path") + "_sha256")
            if isinstance(item, str) and item.startswith("/") and certificate in value:
                # Parent libraries occur as parent/parent_sha256 and
                # parent_gc_path/parent_gc_sha256, not only path/sha256.
                continue
            if key == "link_command" and "binaries" in value and isinstance(item, list):
                # Build operands are provenance. Executable identity remains
                # bound to every binary digest and the original link flags.
                item = [re.sub(r"/[^,\s]+", "<build-path>", argument) for argument in item]
            result[Path(key).name if str(key).startswith("/") else str(key)] = relocated_content(item)
        return result
    if isinstance(value, list):
        return [relocated_content(item) for item in value]
    return value


def _legacy_relocated_content(value):
    """Read the previous namespace certificate solely for validated migration."""
    if isinstance(value, dict):
        return {
            (Path(key).name if str(key).startswith("/") else str(key)): _legacy_relocated_content(item)
            for key, item in value.items() if key != "path" or "sha256" not in value
        }
    if isinstance(value, list):
        return [_legacy_relocated_content(item) for item in value]
    return value


def stable_serving_contract(model, runtime_identity, arguments):
    return dict(schema=2,
                model=file_content(Path(model) / "manifest.json"),
                runtime=runtime_identity,
                arguments=arguments)


@lru_cache(maxsize=128)
def _file_content(path, device, inode, size, modified, legacy):
    path = Path(path)
    if path.suffix == ".json":
        normalize = _legacy_relocated_content if legacy else relocated_content
        data = json.dumps(normalize(json.loads(path.read_text())), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_content(path, *, legacy=False):
    stat = Path(path).stat()
    return _file_content(str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, legacy)


def runtime_content_identity(profile, *, legacy=False):
    environment = semantic_environment(profile["environment"], legacy=legacy)
    records = {}
    for kind in ("additional_libraries", "configuration_files"):
        records[kind] = [(Path(item["path"]).name, file_content(item["path"], legacy=legacy))
                         for item in profile.get(kind, ())]
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
    for name in ("ops/tp2_prepared_plan.py", "compilation/deepseek_v41_prepared.py"):
        result[name] = hashlib.sha256(
            ast.dump(ast.parse((plugin / name).read_text()), include_attributes=False).encode()).hexdigest()
    return result


def lowered_keys(graph_bytes):
    """Keep compatible metadata transport changes out of compiler identities."""
    dependencies = lowering_dependencies()
    key = hashlib.sha256(graph_bytes)
    key.update(json.dumps(dependencies, sort_keys=True).encode())
    # The first metadata schema included its encoder in the compiler key.
    # These artifacts remain usable only when every actual compiler dependency
    # and the entire frontend graph still match. Unknown encoders are rejected.
    legacy = dict(dependencies)
    legacy["compilation/deepseek_v41_backend_cache.py"] = (
        "8f115516a301fa93d090b05f11232291b329f36c055fcada2625f9f40d1c107a")
    previous = hashlib.sha256(graph_bytes)
    previous.update(json.dumps(legacy, sort_keys=True).encode())
    return key.hexdigest(), (previous.hexdigest(), )


def semantic_environment(environment, *, legacy=False):
    ignored = ("TMPDIR", "HABANA_LOGS", "VLLM_HPU_DSV4_WORKER_CPUS", "VLLM_HPU_DSV4_WORKER_HELPER_CPUS",
               "VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", "PT_HPU_RECIPE_CACHE_CONFIG")
    if not legacy:
        ignored += ("VLLM_HPU_DSV41_BACKEND_CACHE", )
    result = {}
    for key, value in environment.items():
        if key in ignored or not key.startswith(("VLLM_HPU_", "PT_HPU_", "HCCL_", "HCL_")):
            continue
        if isinstance(value, str) and value.startswith("/"):
            path = Path(value)
            manifest = path / "manifest.json" if path.is_dir() else path
            if manifest.is_file():
                result[key] = file_content(manifest, legacy=legacy)
            else:
                # Runtime binary directories are separately content-certified
                # by the serving runtime profile and native build manifest.
                result[key] = "runtime-content-bound"
        else:
            result[key] = value
    return result
