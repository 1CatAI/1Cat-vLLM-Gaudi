# SPDX-License-Identifier: Apache-2.0
"""Finalize explicitly rebuilt DSpark libraries without weakening validation."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", action="append", required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--profile-input", type=Path)
    parser.add_argument("--profile-output", type=Path)
    args = parser.parse_args()
    if bool(args.profile_input) != bool(args.profile_output):
        parser.error("Provide both runtime profile paths")
    profile = None
    if args.profile_input:
        profile = json.loads(args.profile_input.read_text())
        if not isinstance(profile.get("environment"), dict):
            raise ValueError("Runtime profile requires its environment mapping")
        if args.profile_output.exists():
            raise ValueError("Use a new runtime profile; never overwrite a qualified parent")

    root = args.root.resolve()
    manifest = root / "deepseek_v41_unique_build.json"
    data = json.loads(manifest.read_text())
    changed = set(args.binary)
    for name in changed:
        if Path(name).name != name or not name.endswith(".so"):
            raise ValueError("Use an explicitly rebuilt library basename")
    # Inherited artifacts must still match their prior recorded identity.
    for name, expected in data["binaries"].items():
        if name not in changed and digest(root / name) != expected:
            raise ValueError(f"Unrequested inherited library change: {name}")
    previous = digest(manifest)
    updates = {}
    for name in sorted(changed):
        actual = digest(root / name)
        updates[name] = dict(previous=data["binaries"].get(name), rebuilt=actual)
        data["binaries"][name] = actual
        if name.startswith("hpu_dsv41_") and "unique_pt2" not in name:
            extras = data.setdefault("extra_registrations", [])
            if name not in extras:
                extras.append(name)
    for source in args.source:
        source = source.resolve()
        data["sources"][str(source)] = digest(source)
    data["build_manifest_update"] = dict(previous_manifest_sha256=previous, explicit_rebuilt_libraries=updates)
    data["default_enabled"] = False
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    if profile is not None:
        environment = profile["environment"]
        environment["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"] = str(root)
        environment["GC_KERNEL_PATH"] = str(root / "libdeepseek_v41_unique_kernels.so")
        environment["VLLM_HPU_DSV41_UNIQUE_KERNEL"] = environment["GC_KERNEL_PATH"]
        args.profile_output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(dict(root=str(root), manifest_sha256=digest(manifest), updated=updates)))


if __name__ == "__main__":
    main()
