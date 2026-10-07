#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate an optional epoch-aware overlay against the exact parent source."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("hcl", "synapse"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    bundle = Path(__file__).parent / "patches/native-runtime"
    manifest = json.loads((bundle / "receive-prepost.json").read_text())
    entry = manifest["components"][args.component]
    patch = bundle / entry["patch"]
    if digest(patch) != entry["sha256"]:
        raise RuntimeError("Receive overlay differs from its manifest")
    source = args.source.resolve()
    for name, hashes in entry["files"].items():
        if digest(source / name) != hashes["before"]:
            raise RuntimeError(f"Unmatched receive-overlay parent: {name}")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    if args.apply:
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        for name, hashes in entry["files"].items():
            if digest(source / name) != hashes["after"]:
                raise RuntimeError(f"Receive-overlay result differs: {name}")
    print(f"{args.component}: {'applied' if args.apply else 'checked'} receive epoch overlay")


if __name__ == "__main__":
    main()
