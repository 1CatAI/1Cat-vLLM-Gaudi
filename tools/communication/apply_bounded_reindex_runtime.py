#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply bounded publication to an isolated, already patched Synapse source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    bundle = Path(__file__).with_name("patches") / "native-runtime"
    manifest = json.loads((bundle / "synapse-bounded-reindex.json").read_text())
    source = args.source.resolve()
    patch = bundle / manifest["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Bounded runtime patch fingerprint differs")
    for relative, expected in manifest["files"].items():
        path = source / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected["before"]:
            raise RuntimeError(f"Bounded runtime source differs: {relative}")
    command = ["git", "apply", "--whitespace=error", str(patch)]
    subprocess.run(command[:2] + ["--check"] + command[2:], cwd=source, check=True)
    if args.apply:
        subprocess.run(command, cwd=source, check=True)
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected["after"]:
                raise RuntimeError(f"Applied bounded runtime fingerprint differs: {relative}")
    print("Bounded native publication: " + ("applied" if args.apply else "checked"))


if __name__ == "__main__":
    main()
