# SPDX-License-Identifier: Apache-2.0
"""Materialize a fingerprinted serving installation without editable workspace imports.

Model/checkpoint assets and the system Gaudi SDK remain deployment prerequisites.
The Python environment, engine, plugin, selected native runtime and sidecars are copied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import secrets
import subprocess
import sys
import venv


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def copy_tree(source, target, exclude=()):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", *exclude))


def relocate(value, mapping):
    if isinstance(value, dict):
        return {k: relocate(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, mapping) for v in value]
    if isinstance(value, str):
        for source, target in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if value == source or value.startswith(source + "/"):
                return target + value[len(source):]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--engine-source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--machine-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-output", type=Path,
                        help="Separate immutable sidecar installation on the model filesystem")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Installation destination must be new; existing services are never overwritten")
    assets = args.asset_output.resolve() if args.asset_output else output / "sidecars"
    if assets.exists():
        raise ValueError("Sidecar installation destination must be new")
    profile = json.loads(args.runtime_profile.read_text())
    for item in profile.get("additional_libraries", []) + profile.get("configuration_files", []):
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Input runtime fingerprint mismatch: " + item["path"])
    engine_lock = json.loads((root / "tools/communication/patches/dsv41-serving-engine.json").read_text())
    for item in engine_lock["files"]:
        if digest(args.engine_source / item["path"]) != item["candidate_sha256"]:
            raise ValueError("Engine source differs from the maintained serving lock: " + item["path"])
    native = Path(profile["environment"]["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
    for name in ("deepseek_v4_build.json", "deepseek_v41_build.json"):
        for relative, expected in json.loads((native / name).read_text())["sources"].items():
            if digest(root / relative) != expected:
                raise ValueError("Native build differs from maintained source: " + relative)
    storage_parent = output.parent
    while not storage_parent.exists():
        storage_parent = storage_parent.parent
    if shutil.disk_usage(storage_parent).free < 6 * 2**30:
        raise ValueError("Keep at least 6 GiB free for the Python/runtime installation and startup logs")
    # Copy dependencies, excluding editable import hooks. Never retain external symlinks.
    source_site = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if not source_site.is_dir():
        raise ValueError("Run the installer with the qualified serving virtual environment's Python")
    output.mkdir(parents=True)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(output / "venv")
    site = output / "venv" / source_site.relative_to(sys.prefix)
    for source in source_site.iterdir():
        if source.name.startswith("__editable__") or source.name in ("__pycache__", "vllm", "vllm_gaudi"):
            continue
        destination = site / source.name
        if source.is_dir():
            copy_tree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)
    copy_tree(args.engine_source.resolve() / "vllm", site / "vllm")
    copy_tree(root / "vllm_gaudi", site / "vllm_gaudi", exclude=("lib",))
    for name in ("flashinfer_gaudi",):
        copy_tree(root / name, site / name)
    shutil.copy2(root / "pytest_compat.py", site / "pytest_compat.py")
    # Ignored development lib symlinks must not become the installation's native selection.
    shutil.rmtree(site / "vllm_gaudi/lib", ignore_errors=True)
    library_dir = output / "lib"
    library_dir.mkdir()
    env = profile["environment"]
    native = Path(env["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
    copy_tree(native, output / "native")
    bridge = Path(env["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"])
    shutil.copy2(bridge, library_dir / bridge.name)
    mapping = {str(source_site): str(site), str(native): str(output / "native"),
               str(bridge): str(library_dir / bridge.name)}
    # Preserve first-match loader precedence, omitting shadowed historical builds.
    for directory in env.get("LD_LIBRARY_PATH", "").split(":"):
        source = Path(directory)
        if not directory or str(source).startswith(("/usr/", "/opt/habanalabs/")):
            continue
        if "site-packages" in directory:
            continue
        for binary in sorted(source.glob("*.so*")):
            if binary.name.endswith(".debug") or (library_dir / binary.name).exists():
                continue
            shutil.copy2(binary, library_dir / binary.name)
            mapping[str(binary)] = str(library_dir / binary.name)
    preload = []
    for name in env.get("LD_PRELOAD", "").split(":"):
        if name:
            path = Path(name)
            shutil.copy2(path, library_dir / path.name)
            mapping[str(path)] = str(library_dir / path.name)
            preload.append(str(library_dir / path.name))
    # ABI manifests identify actual copied dependencies, rather than former build paths.
    abi = json.loads(bridge.with_suffix(".abi.json").read_text())
    for item in abi["eager_runtime"]:
        old = Path(item["path"])
        suffix = str(old).split("/site-packages/", 1)[1]
        mapping[str(old)] = str(site / suffix)
    for item in abi.get("bridge_dependencies", {}).values():
        old = Path(item["path"])
        suffix = str(old).split("/site-packages/", 1)[1]
        mapping[str(old)] = str(site / suffix)
    dump(library_dir / bridge.with_suffix(".abi.json").name, relocate(abi, mapping))
    sidecars = {}
    for name, key in (("wo_a_fp8", "VLLM_HPU_DSV41_WO_A_FP8_SIDECAR"),
                      ("attention_dense_fp8", "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"),
                      ("engram_fp8", "VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR")):
        source = Path(env.get(key, str(args.prepared / "sidecars" / name)))
        target = assets / name
        if args.asset_output:
            # Independent directory entries survive deletion of the source build.
            # Assets are immutable; cross-filesystem copies use normal materialization.
            def copy_asset(src, dst):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                if Path(src).stat().st_dev == Path(dst).parent.stat().st_dev:
                    os.link(Path(src).resolve(), dst)
                    return dst
                return shutil.copy2(src, dst)
            shutil.copytree(source, target, copy_function=copy_asset)
        else:
            copy_tree(source, target)
        mapping[str(source)] = str(target)
        sidecars[name] = str(target)
    precision = env.get("VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG")
    if precision:
        target = assets / "attention_dense_fp8/precision.json"
        shutil.copy2(precision, target)
        mapping[precision] = str(target)
    installed_env = relocate(env, mapping)
    for key in list(installed_env):
        if key in ("GRAPH_VISUALIZATION", "VLLM_HPU_DSV41_RAW_TRACE", "VLLM_HPU_DSV41_RAW_SCOPE_ONLY",
                   "VLLM_HPU_PROFILE_REUSE_NATIVE_COMMANDS", "VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG"):
            installed_env.pop(key)
    installed_env["LD_LIBRARY_PATH"] = ":".join(map(str, (library_dir, site / "habana_frameworks/torch/lib",
        site / "habana_frameworks/torch/lib/upstream_pybind", site / "torch/lib",
        Path("/usr/lib/habanalabs"), Path("/opt/habanalabs/openmpi-5.0.8/lib"))))
    if preload:
        installed_env["LD_PRELOAD"] = ":".join(preload)
    # The ordinary entrypoint owns fastpath defaults; installation profiles contain artifacts/SDK settings.
    for key in list(installed_env):
        if key.startswith("VLLM_HPU_DSV41_") and key not in (
                "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR", "VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR"):
            installed_env.pop(key)
    records = [{"path": str(p), "sha256": digest(p)} for p in library_dir.glob("*.so")]
    configurations = [{"path": str(p), "sha256": digest(p)} for p in library_dir.glob("*.json")]
    dump(output / "runtime.json", {"schema": 1, "environment": installed_env,
                                  "additional_libraries": records, "configuration_files": configurations})
    settings = json.loads(args.machine_settings.read_text())
    lock_path = Path(settings["device_lock_dir"])
    if not lock_path.is_absolute():
        settings["device_lock_dir"] = str(output.parent / lock_path)
    if not settings.get("api_key_file"):
        key = output / "api-key"
        descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write("sk-" + secrets.token_hex(24) + "\n")
        settings["api_key_file"] = str(key)
    settings.update(model=str(args.prepared.resolve()), runtime_profile=str(output / "runtime.json"), sidecars=sidecars)
    dump(output / "settings.json", settings)
    copy_tree(root / "tools", output / "tools")
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file() and
             (p.suffix in (".py", ".json") or p.name.endswith(".so"))}
    dump(output / "installation.json", {"schema": 1, "plugin_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "files": files,
        "python": sys.version, "system_sdk_required": True, "prepared_model_is_external": True})
    print(output)


if __name__ == "__main__":
    main()
