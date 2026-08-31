# SPDX-License-Identifier: Apache-2.0
"""Build the in-tree FlashInfer-Gaudi kernel database and PyTorch op."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_kernel_database(source_root: Path, build_root: Path, output_dir: Path) -> Path:
    cmake_build = build_root / "kernel_db"
    _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=source_root,
    )
    _run(
        ["cmake", "--build", str(cmake_build), "--target", "flashinfer_gaudi_kernels", "--parallel"],
        cwd=source_root,
    )
    built = cmake_build / "libflashinfer_gaudi_kernels.so"
    if not built.is_file():
        raise FileNotFoundError(f"CMake did not produce {built}.")
    destination = output_dir / built.name
    shutil.copy2(built, destination)
    return destination


def build_pytorch_extension(source_root: Path, build_root: Path, output_dir: Path) -> Path:
    pytorch_root = source_root / "pytorch"
    extension_build = build_root / "pytorch"
    _run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(output_dir),
            "--build-temp",
            str(extension_build),
        ],
        cwd=pytorch_root,
    )
    matches = sorted(output_dir.glob("flashinfer_gaudi_ops*.so"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one flashinfer_gaudi_ops extension in {output_dir}, found {matches}.")
    return matches[0]


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "flashinfer_gaudi" / "lib",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=repository_root / "build" / "flashinfer_gaudi",
    )
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--extension-only", action="store_true")
    args = parser.parse_args()
    if args.kernel_only and args.extension_only:
        parser.error("--kernel-only and --extension-only are mutually exclusive")

    source_root = repository_root / "csrc" / "flashinfer_gaudi"
    output_dir = args.output_dir.resolve()
    build_root = args.build_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    if not args.extension_only:
        build_kernel_database(source_root, build_root, output_dir)
    if not args.kernel_only:
        build_pytorch_extension(source_root, build_root, output_dir)


if __name__ == "__main__":
    main()

