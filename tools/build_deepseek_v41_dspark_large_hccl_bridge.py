# SPDX-License-Identifier: Apache-2.0
"""Build a private, OFF-by-default C6 stock-HCCL capacity adapter.

Copy the shared sources; retain their hashes, the minimal capacity patch and
the exact serving ABI dependencies. Never rewrite the shared C1 adapter.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-abi", type=Path, required=True)
    parser.add_argument("--cmake-build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    abi = json.loads(args.baseline_abi.read_text())
    shared = Path(__file__).resolve().parent / "communication"
    source = args.output.resolve() / "source"
    source.mkdir(parents=True, exist_ok=False)
    proof, patches = [], []
    for original in sorted(shared.iterdir()):
        if original.suffix in (".cpp", ".h") or original.name == "build_tp2_fused_ar_norm_bridge.py":
            shutil.copy2(original, source / original.name)
            proof.append(dict(path=str(original), sha256=digest(original)))
    edits = {
        "tp2_fused_ar_norm_bridge.cpp": (
            "    TORCH_CHECK(partial.device().type() == at::kHPU && partial.numel() <= 32768 &&",
            '    const char* largeFlag = std::getenv("VLLM_HPU_DSV41_DSPARK_LARGE_HCCL");\n'
            '    const int64_t maximum = all_gather && largeFlag && std::strcmp(largeFlag, "1") == 0\n'
            '        ? 387840 : 32768;\n'
            "    TORCH_CHECK(partial.device().type() == at::kHPU && partial.numel() <= maximum &&",
        ),
        "tp2_prepared_plan.h": (
            '          const int64_t v41Maximum = batchFlag && std::strcmp(batchFlag, "1") == 0 ? 64 * 5120 : 32768;',
            '          const char* largeFlag = std::getenv("VLLM_HPU_DSV41_DSPARK_LARGE_HCCL");\n'
            '          const int64_t v41Maximum = node.all_gather && largeFlag && std::strcmp(largeFlag, "1") == 0\n'
            '              ? 387840 : batchFlag && std::strcmp(batchFlag, "1") == 0 ? 64 * 5120 : 32768;',
        ),
    }
    for name, (old, new) in edits.items():
        path = source / name
        original = path.read_text()
        if original.count(old) != 1:
            raise ValueError(f"Shared capacity guard changed; inspect before patching: {name}")
        updated = original.replace(old, new)
        path.write_text(updated)
        patches.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                           fromfile=f"shared/{name}", tofile=f"private/{name}"))
    (args.output / "private-capacity.patch").write_text("".join(patches))
    proof_path = args.output / "SHARED_SOURCE_PROOF.json"
    proof_path.write_text(json.dumps(dict(shared=proof, flag="VLLM_HPU_DSV41_DSPARK_LARGE_HCCL",
                                         default_enabled=False, transport="existing stock HCCL",
                                         baseline_abi=str(args.baseline_abi.resolve())), indent=2) + "\n")
    header = next(Path(path) for path in abi["headers"] if path.endswith("habana_eager/graph_storage.h"))
    command = [sys.executable, str(source / "build_tp2_fused_ar_norm_bridge.py"),
               "--bridge-source", str(header.parent.parent),
               "--cmake-build-directory", str(args.cmake_build_directory.resolve()),
               "--build-directory", str(args.output.resolve() / "build")]
    for key, option in (("backend", "--backend-library"), ("native_hccl", "--native-hccl-library")):
        entry = abi["bridge_dependencies"][key]
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Serving dependency changed: {key}")
        command.extend((option, entry["path"]))
    for key, option in (("synapse", "--synapse-library"), ("hcl", "--hcl-library")):
        entry = abi["native_dependencies"][key]
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Serving native runtime changed: {key}")
        command.extend((option, entry["path"]))
    (args.output / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    dependency_dirs = [str(Path(abi["native_dependencies"][key]["path"]).parent) for key in ("synapse", "hcl")]
    library_path = os.pathsep.join((*dependency_dirs, os.environ.get("LD_LIBRARY_PATH", "")))
    subprocess.run(command, env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", MAX_JOBS="1",
                                     LD_LIBRARY_PATH=library_path), check=True)
    if any(digest(Path(row["path"])) != row["sha256"] for row in proof):
        raise RuntimeError("Shared source changed during private build")


if __name__ == "__main__":
    main()
