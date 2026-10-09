# SPDX-License-Identifier: Apache-2.0
"""Pin an immutable additive native bundle without mixing kernel databases."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.baseline.read_text())
    old = profile["environment"]["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"]
    bundle = args.bundle.resolve()

    def relocate(value):
        if isinstance(value, str):
            return value.replace(old, str(bundle))
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value

    profile = relocate(profile)
    kernel = Path(profile["environment"]["GC_KERNEL_PATH"])
    if kernel.parent != bundle or not kernel.is_file():
        raise ValueError("Native registration and combined kernel database must use the same bundle")
    manifest = bundle / "deepseek_v41_unique_build.json"
    metadata = json.loads(manifest.read_text())
    for name, expected in metadata["binaries"].items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Bundle binary changed: {name}")
    additions = {kernel, *(bundle / name for name in metadata["extra_registrations"])}
    for key, files in (("additional_libraries", additions), ("configuration_files", {manifest})):
        rows = {item["path"]: item for item in profile.get(key, [])}
        for item in rows.values():
            path = Path(item["path"])
            if path.parent == bundle:
                item["sha256"] = digest(path)
        for path in files:
            rows[str(path)] = dict(path=str(path), sha256=digest(path))
        profile[key] = list(rows.values())
    with args.output.open("x") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
