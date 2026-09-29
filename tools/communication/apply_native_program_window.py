#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep native TPC programs within one hardware PC window in private Synapse builds."""

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
    manifest = json.loads((bundle / "synapse-program-window.json").read_text())
    source = args.source.resolve()
    patch = bundle / manifest["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Native program-window patch fingerprint differs")
    for relative, expected in manifest["files"].items():
        path = source / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected["before"]:
            raise RuntimeError(f"Native program-window source differs: {relative}")
    command = ["git", "apply", "--whitespace=error", str(patch)]
    subprocess.run(command[:2] + ["--check"] + command[2:], cwd=source, check=True)
    if args.apply:
        subprocess.run(command, cwd=source, check=True)
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected["after"]:
                raise RuntimeError(f"Applied native program-window fingerprint differs: {relative}")
    print("Native program-window contract: " + ("applied" if args.apply else "checked"))


if __name__ == "__main__":
    main()
