# SPDX-License-Identifier: Apache-2.0
"""Build the opt-in mixed-engine adapter against the exact installed Bridge ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from flashinfer_gaudi._bridge import BRIDGE_VERSION, runtime_files, runtime_identity, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--bridge-build", type=Path, required=True)
    args = parser.parse_args()
    source, generated = args.bridge_source.resolve(), args.bridge_build.resolve()
    for path in (source / "hpu_ops/op_backend.h", generated / "generated/env_flags/env_flags_generated.h",
                 generated / "_deps/abseil-cpp-src/absl/container/flat_hash_map.h",
                 generated / "_deps/exprtk-src/include/exprtk.hpp", generated / "_deps/fmt-src/include/fmt/core.h",
                 generated / "_deps/magic_enum-src/include/magic_enum/magic_enum.hpp"):
        if not path.is_file():
            parser.error(f"Missing matching Bridge build dependency: {path}")
    identity = runtime_identity()
    if (identity["bridge_version"] != BRIDGE_VERSION or identity["synapse_version"].split(".")[:3] != ["1", "24", "1"]):
        parser.error(f"Only Bridge {BRIDGE_VERSION} and Synapse 1.24.1 are supported")
    output = root / "flashinfer_gaudi/lib"
    output.mkdir(parents=True, exist_ok=True)
    files = runtime_files(output)
    for name, path in files.items():
        if name != "adapter" and not path.is_file():
            parser.error(f"Missing {name}; build the TPC kernel database first")
    native_source = root / "csrc/flashinfer_gaudi/bridge"
    # SDK headers use version-qualified include paths; reuse the matching
    # Bridge dependency checkout without modifying it or downloading headers.
    layout = root / "build/flashinfer_gaudi/bridge/dependency_layout"
    layout.mkdir(parents=True, exist_ok=True)
    for alias, checkout in (("magic_enum-0.9.7", "magic_enum-src"), ("fmt-9.1.0", "fmt-src")):
        link, target = layout / alias, generated / "_deps" / checkout
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        elif link.resolve() != target.resolve():
            parser.error("Bridge dependency layout belongs to another build; use a clean build directory")
    subprocess.run(
        [
            sys.executable, "setup.py", "build_ext", "--build-lib",
            str(output), "--build-temp",
            str(root / "build/flashinfer_gaudi/bridge")
        ],
        cwd=native_source,
        env={
            **os.environ, "GAUDI_PYTORCH_BRIDGE_SOURCE": str(source),
            "GAUDI_BRIDGE_BUILD": str(generated),
            "GAUDI_BRIDGE_DEPENDENCY_LAYOUT": str(layout)
        },
        check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    manifest = {
        "schema_version":
        1,
        "target":
        "gaudi2",
        "runtime":
        identity,
        "bridge_source_revision":
        revision,
        "bridge_source_diff_sha256":
        hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=source)).hexdigest(),
        "adapter_source_sha256":
        sha256(native_source / "gemm_silu.cpp"),
        "silu_quant_source_sha256":
        sha256(native_source / "silu_quant.cpp"),
        "block_fp8_source_sha256":
        sha256(native_source / "block_fp8.cpp"),
        "sha256": {
            name: sha256(path)
            for name, path in files.items()
        },
    }
    destination = output / "bridge_artifact_v1.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    print("Built ABI-locked mixed-engine adapter; production dispatch remains unchanged.")


if __name__ == "__main__":
    main()
