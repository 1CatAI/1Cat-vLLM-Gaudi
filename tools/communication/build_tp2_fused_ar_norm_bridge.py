#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the isolated TP2 all-reduce plus residual RMSNorm bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import habana_frameworks.torch as htorch
from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--cmake-build-directory", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    args = parser.parse_args()

    source = Path(__file__).with_name("tp2_fused_ar_norm_bridge.cpp")
    torch_package = Path(htorch.__file__).resolve().parent
    pybind_variant = "fork_pybind" if htorch.is_torch_fork else "upstream_pybind"
    backend_name = "libhabana_pytorch_backend.so" if htorch.is_torch_fork else "libhabana_pytorch_backend.upstream.so"
    native_hccl = torch_package / "lib" / pybind_variant / "_hccl_eager_C.so"
    backend = torch_package / "lib" / backend_name

    args.build_directory.mkdir(parents=True, exist_ok=True)
    module = load(
        name="tp2_fused_ar_norm_bridge",
        sources=[str(source)],
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
            "-O3",
            "-std=c++17",
            "-DFMT_HEADER_ONLY=1",
            "-DGENERIC_HELPERS",
            "-fopenmp",
            "-fpermissive",
        ],
        extra_ldflags=[
            str(native_hccl),
            str(backend),
            "-L/usr/lib/habanalabs",
            "-lhcl",
            "-fopenmp",
            f"-Wl,-rpath,{native_hccl.parent}",
            f"-Wl,-rpath,{backend.parent}",
            "-Wl,-rpath,/usr/lib/habanalabs",
        ],
        build_directory=str(args.build_directory.resolve()),
        verbose=True,
    )
    print(Path(module.__file__).resolve())


if __name__ == "__main__":
    main()
