#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check or apply a pinned native TP2 runtime patch to an isolated checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("bridge", "synapse", "hcl"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after all checks; default only checks")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent / "patches/native-runtime"
    entry = json.loads((bundle / "manifest.json").read_text())["components"][args.component]
    patch = bundle / entry["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError("Runtime patch fingerprint differs from its manifest")
    source = args.source.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != entry["base_commit"]:
        raise RuntimeError(f"Expected {entry['base_commit']}, found {head}")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source, text=True)
    if status:
        raise RuntimeError("Use a clean isolated checkout; existing changes will not be overwritten")
    subprocess.run(["git", "apply", "--check", "--whitespace=error", str(patch)], cwd=source, check=True)
    if args.apply:
        subprocess.run(["git", "apply", "--whitespace=error", str(patch)], cwd=source, check=True)
    print(f"{args.component}: {'applied' if args.apply else 'checked'} at {head}")


if __name__ == "__main__":
    main()
