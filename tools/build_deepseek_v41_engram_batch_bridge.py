# SPDX-License-Identifier: Apache-2.0
"""Build the common Engram producer on an unchanged qualified replay/peer core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_region(destination, source, first, last):
    left, right = destination.index(first), destination.index(last)
    begin, end = source.index(first), source.index(last)
    return destination[:left] + source[begin:end] + destination[right:]


def without_producer(text):
    for first, last in (
        ("constexpr uint64_t kDeviceEngramHeads", '#include "tp2_prepared_plan.h"'),
        ('  module.attr("device_engram_shared_mapping_version")', '  py::class_<tp2_native::NativeCompletion'),
    ):
        left, right = text.index(first), text.index(last)
        text = text[:left] + text[right:]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bridge", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--cmake-build-directory", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_bridge.resolve()
    metadata = json.loads(base.with_suffix(".abi.json").read_text())
    if digest(base) != metadata["binary_sha256"]:
        raise ValueError("Qualified bridge changed")
    source_dir = args.output_directory.resolve() / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    for path, expected in metadata["adapter_sources"].items():
        path = Path(path)
        if digest(path) != expected:
            raise ValueError(f"Qualified adapter source changed: {path}")
        shutil.copy2(path, source_dir / path.name)
    implementation = Path(__file__).with_name("communication") / "tp2_fused_ar_norm_bridge.cpp"
    current = implementation.read_text()
    original_path = source_dir / implementation.name
    original = original_path.read_text()
    candidate = original
    for first, last in (
        ("constexpr uint64_t kDeviceEngramHeads", '#include "tp2_prepared_plan.h"'),
        ('  module.attr("device_engram_shared_mapping_version")', '  py::class_<tp2_native::NativeCompletion'),
    ):
        candidate = replace_region(candidate, current, first, last)
    if without_producer(candidate) != without_producer(original):
        raise ValueError("Replay/peer core must remain unchanged")
    original_path.write_text(candidate)
    builder = Path(__file__).with_name("communication") / "build_tp2_fused_ar_norm_bridge.py"
    shutil.copy2(builder, source_dir / builder.name)
    command = [sys.executable, str(source_dir / builder.name),
               "--bridge-source", str(args.bridge_source.resolve()),
               "--cmake-build-directory", str(args.cmake_build_directory.resolve()),
               "--build-directory", str(args.output_directory.resolve() / "build")]
    for argument, key in (("backend-library", "backend"), ("native-hccl-library", "native_hccl")):
        item = metadata["bridge_dependencies"][key]
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Qualified {key} changed")
        command += ["--" + argument, item["path"]]
    for argument, key in (("synapse-library", "synapse"), ("hcl-library", "hcl")):
        item = metadata["native_dependencies"][key]
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Qualified {key} changed")
        command += ["--" + argument, item["path"]]
    evidence = dict(base_bridge=str(base), base_sha256=digest(base),
                    replay_peer_source_unchanged=True, implementation=str(implementation.resolve()),
                    implementation_sha256=digest(implementation), command=command)
    (args.output_directory / "producer-build-contract.json").write_text(json.dumps(evidence, indent=2) + "\n")
    subprocess.run(command, env=dict(os.environ, MAX_JOBS=os.environ.get("MAX_JOBS", "48")), check=True)


if __name__ == "__main__":
    main()
