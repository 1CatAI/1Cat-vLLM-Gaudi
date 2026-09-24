#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the isolated TP2 all-reduce plus residual RMSNorm bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import torch
from pathlib import Path

import habana_frameworks.torch as htorch
from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--cmake-build-directory", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--backend-library", type=Path)
    parser.add_argument("--native-hccl-library", type=Path)
    parser.add_argument("--synapse-library", type=Path)
    parser.add_argument("--hcl-library", type=Path)
    parser.add_argument("--address-sanitizer", action="store_true", help="Build a diagnostic ASan extension")
    args = parser.parse_args()
    sanitizer_flags = ["-fsanitize=address", "-fno-omit-frame-pointer"] if args.address_sanitizer else []

    source = Path(__file__).with_name("tp2_fused_ar_norm_bridge.cpp")
    torch_package = Path(htorch.__file__).resolve().parent
    pybind_variant = "fork_pybind" if htorch.is_torch_fork else "upstream_pybind"
    backend_name = "libhabana_pytorch_backend.so" if htorch.is_torch_fork else "libhabana_pytorch_backend.upstream.so"
    native_hccl = (args.native_hccl_library.resolve() if args.native_hccl_library else torch_package / "lib" /
                   pybind_variant / "_hccl_eager_C.so")
    backend = (args.backend_library.resolve() if args.backend_library else torch_package / "lib" / backend_name)
    for name, path in (("backend", backend), ("native HCCL binding", native_hccl)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} library does not exist: {path}")
    dependency_libraries = {
        "synapse": args.synapse_library.resolve() if args.synapse_library else None,
        "hcl": args.hcl_library.resolve() if args.hcl_library else None,
    }
    for name, path in dependency_libraries.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name} library does not exist: {path}")

    native_dependency_link = []
    if any(dependency_libraries.values()):
        native_dependency_link.append("-Wl,--no-as-needed")
    if dependency_libraries["synapse"]:
        native_dependency_link.append(str(dependency_libraries["synapse"]))
    if dependency_libraries["hcl"]:
        native_dependency_link.append(str(dependency_libraries["hcl"]))
    else:
        native_dependency_link.extend(("-L/usr/lib/habanalabs", "-lhcl"))
    if any(dependency_libraries.values()):
        native_dependency_link.append("-Wl,--as-needed")
    dependency_rpaths = [f"-Wl,-rpath,{path.parent}" for path in dependency_libraries.values() if path]

    args.build_directory.mkdir(parents=True, exist_ok=True)
    module = load(
        name="tp2_fused_ar_norm_bridge",
        sources=[
            str(source),
            str(source.with_name("gdn_state_update.cpp")),
            str(source.with_name("tp2_dynamic_quant.cpp"))
        ],
        extra_include_paths=[
            str(args.bridge_source.resolve()),
            str((args.bridge_source / "pytorch_helpers").resolve()),
            str((args.bridge_source / "pytorch_helpers/habana_helpers/habana_serialization/include").resolve()),
            str((args.bridge_source / "python_packages/habana_frameworks/torch/jit/csrc").resolve()),
            str(args.cmake_build_directory.resolve()),
            str((args.cmake_build_directory / "_deps").resolve()),
            str((args.cmake_build_directory / "_deps/abseil-cpp-src").resolve()),
            str((args.cmake_build_directory / "_deps/exprtk-src/include").resolve()),
            str((args.cmake_build_directory / "_deps/fmt-src/include").resolve()),
            str((args.cmake_build_directory / "_deps/magic_enum-src/include").resolve()),
            str((args.cmake_build_directory / "_deps/nlohmann_json-src/include").resolve()),
            "/usr/include/habanalabs",
            "/usr/include/habanalabs/hl_logger",
        ],
        extra_cflags=[
            *sanitizer_flags,
            "-O3",
            "-std=c++17",
            "-DFMT_HEADER_ONLY=1",
            "-DGENERIC_HELPERS",
            "-fopenmp",
            "-fpermissive",
        ],
        extra_ldflags=[
            *sanitizer_flags,
            str(native_hccl),
            str(backend),
            *native_dependency_link,
            "-fopenmp",
            f"-Wl,-rpath,{native_hccl.parent}",
            f"-Wl,-rpath,{backend.parent}",
            *dependency_rpaths,
            "-Wl,-rpath,/usr/lib/habanalabs",
        ],
        build_directory=str(args.build_directory.resolve()),
        verbose=True,
    )
    binary = Path(module.__file__).resolve()
    eager_name = ("libhabana_pytorch2_plugin.so" if htorch.is_torch_fork else "libhabana_pytorch2_plugin.upstream.so")
    runtimes = {
        Path(line.split(maxsplit=5)[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[-1].endswith(eager_name)
    }
    if len(runtimes) != 1:
        raise RuntimeError(f"Cannot lock the prepared GraphExec runtime: {runtimes}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    loaded_dependencies = {}
    mapped_files = {
        Path(line.split(maxsplit=5)[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[-1].startswith("/")
    }
    for name, expected in dependency_libraries.items():
        if expected is None:
            continue
        loaded = {path for path in mapped_files if path.name == expected.name}
        if len(loaded) != 1:
            raise RuntimeError(f"Cannot identify loaded {name} runtime: {loaded}")
        actual = loaded.pop()
        if digest(actual) != digest(expected):
            raise RuntimeError(f"Loaded {name} runtime does not match requested build: {actual} != {expected}")
        loaded_dependencies[name] = {"path": str(actual), "sha256": digest(actual)}
    bridge_dependencies = {}
    for name, expected in (("backend", backend), ("native_hccl", native_hccl)):
        loaded = {path for path in mapped_files if path.name == expected.name}
        if len(loaded) != 1:
            raise RuntimeError(f"Cannot identify loaded {name} library: {loaded}")
        actual = loaded.pop()
        if digest(actual) != digest(expected):
            raise RuntimeError(f"Loaded {name} library does not match requested build: {actual} != {expected}")
        bridge_dependencies[name] = {"path": str(actual), "sha256": digest(actual)}
    metadata = {
        "schema": 1,
        "device_engram_shared_mapping_version": getattr(module, "device_engram_shared_mapping_version", 0),
        "address_sanitizer": args.address_sanitizer,
        "torch_version": torch.__version__,
        "binary_sha256": digest(binary),
        "adapter_sources": {
            str(path.resolve()): digest(path)
            for path in (
                source,
                source.with_name("gdn_state_update.cpp"),
                source.with_name("tp2_dynamic_quant.cpp"),
                source.with_name("tp2_input_preflight.h"),
                source.with_name("tp2_prepared_plan.h"),
                source.with_name("tp2_native_decode_graph.h"),
                source.with_name("tp2_native_dependencies.h"),
                source.with_name("tp2_native_graph_topology.h"),
                source.with_name("tp2_native_graph_probe.h"),
                source.with_name("dsv41_verify_timing.h"),
            )
        },
        "eager_runtime": [{
            "path": str(path),
            "sha256": digest(path)
        } for path in runtimes],
        "bridge_dependencies": bridge_dependencies,
        "native_dependencies": loaded_dependencies,
        "headers": {
            str(path.resolve()): digest(path)
            for path in (args.bridge_source / "habana_eager/graph_storage.h",
                         args.bridge_source / "habana_eager/graph_execs_group.h")
        }
    }
    binary.with_suffix(".abi.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
