# SPDX-License-Identifier: Apache-2.0
"""Compose an immutable additive DSpark installation and relocate its profile."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-profile", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gc-library", type=Path, required=True)
    parser.add_argument("--addon", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--set-env", action="append", default=[])
    parser.add_argument("--required-guid", action="append", default=[])
    args = parser.parse_args()
    if args.prefix.exists() or args.profile.exists():
        raise FileExistsError("Installations and profiles are immutable; use new paths")
    profile = json.loads(args.parent_profile.read_text())
    parent = Path(profile["environment"]["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"]).resolve()
    manifest = json.loads((parent / "deepseek_v41_unique_build.json").read_text())
    for name, expected in manifest["binaries"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Inherited library changed: {name}")
    for path in (args.gc_library, args.addon, *args.source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.gc_library.name != "libdeepseek_v41_unique_kernels.so" or not args.addon.name.endswith(".so"):
        raise ValueError("Expected the additive GC library and one registration library")
    updates = {}
    for value in args.set_env:
        name, sep, text = value.partition("=")
        if not sep or not name.startswith("VLLM_HPU_DSV41_"):
            raise ValueError("Use explicit DSpark environment assignments")
        updates[name] = text
    if args.required_guid:
        from check_deepseek_v41_gc_exports import verify

        profile["required_gc_exports"] = verify(
            args.gc_library, parent / "libdeepseek_v4_gaudi2_kernels.so", args.required_guid)
    args.prefix = args.prefix.resolve()
    shutil.copytree(parent, args.prefix)
    shutil.copy2(args.gc_library, args.prefix / args.gc_library.name)
    shutil.copy2(args.addon, args.prefix / args.addon.name)
    command = [sys.executable, str(Path(__file__).with_name("record_deepseek_v41_additive_build.py")),
               "--root", str(args.prefix), "--binary", args.gc_library.name, "--binary", args.addon.name]
    for source in args.source:
        command += ["--source", str(source.resolve())]
    subprocess.run(command, check=True)
    for name, value in profile["environment"].items():
        if isinstance(value, str):
            profile["environment"][name] = value.replace(str(parent), str(args.prefix))
    profile["environment"].update(updates)
    for group in ("additional_libraries", "configuration_files"):
        for artifact in profile.get(group, []):
            relocated = args.prefix / Path(artifact["path"]).name
            if relocated.is_file():
                artifact.update(path=str(relocated), sha256=digest(relocated))
    profile.setdefault("additional_libraries", []).append(
        dict(path=str(args.prefix / args.addon.name), sha256=digest(args.prefix / args.addon.name)))
    profile["additive_installation"] = dict(
        parent_profile=str(args.parent_profile.resolve()), parent_profile_sha256=digest(args.parent_profile),
        prefix=str(args.prefix), new_registration=args.addon.name)
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(dict(profile=str(args.profile.resolve()), prefix=str(args.prefix))))


if __name__ == "__main__":
    main()
