#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an isolated ABI adapter for Bridge's concat-axis cache-key fix."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--compile-commands", type=Path, required=True)
    parser.add_argument("--frontend-library", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.compile_commands.read_text())
    entry = next(item for item in entries if item["file"].endswith("habana_eager/eager_exec.cpp"))
    old_root = str(Path(entry["file"]).parents[1])
    command = [item.replace(old_root, str(args.bridge_source.resolve()))
               for item in shlex.split(entry["command"])]
    command = command[:command.index("-o")]
    source = Path(__file__).with_name("eager_concat_cache_key.cpp")
    args.build_directory.mkdir(parents=True, exist_ok=True)
    binary = args.build_directory.resolve() / "libeager_concat_cache_key.so"
    command += ["-shared", "-o", str(binary), str(source.resolve()), "-ldl"]
    subprocess.run(command, cwd=entry["directory"], check=True)

    def record(path):
        path = path.resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    metadata = {
        "schema": 1,
        "implementation": "concat-axis-in-eager-cache-key",
        "binary": record(binary),
        "frontend": record(args.frontend_library),
        "source": record(source),
        "headers": [record(args.bridge_source / "habana_eager/eager_exec.h")],
        "command": command,
        "shape_agnostic_cache": "retained",
        "deployment": "private LD_PRELOAD adapter; omit when using the patched frontend",
    }
    (binary.parent / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
