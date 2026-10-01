# SPDX-License-Identifier: Apache-2.0
"""Apply the complete source-pinned V4.1 serving engine delta before installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate the pinned delta without changing source")
    args = parser.parse_args()
    engine = args.engine.resolve()
    bundle = Path(__file__).resolve().parent / "communication/patches/dsv41-serving-engine"
    patch = bundle.with_suffix(".patch")
    lock = json.loads(bundle.with_suffix(".json").read_text())
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
        raise ValueError("Engine patch fingerprint changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
    if head != lock["base_commit"]:
        raise ValueError("Use the engine commit pinned in dsv41-serving-engine.json")
    matches = all((engine / item["path"]).is_file() and hashlib.sha256(
        (engine / item["path"]).read_bytes()).hexdigest() == item["candidate_sha256"] for item in lock["files"])
    if matches:
        print("Pinned V4.1 engine delta already present")
        return
    with tempfile.TemporaryDirectory(prefix="dsv41-engine-index-") as temp:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        subprocess.run(["git", "read-tree", "HEAD"], cwd=engine, env=environment, check=True)
        subprocess.run(["git", "apply", "--cached", str(patch)], cwd=engine, env=environment, check=True)
        for item in lock["files"]:
            blob = subprocess.check_output(["git", "show", ":" + item["path"]], cwd=engine, env=environment)
            if hashlib.sha256(blob).hexdigest() != item["candidate_sha256"]:
                raise ValueError("Engine candidate fingerprint differs from the pinned patch")
    if not args.check:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=engine).strip():
            raise ValueError("Use a clean engine checkout; existing changes remain untouched")
        subprocess.run(["git", "apply", str(patch)], cwd=engine, check=True)
    print("Pinned V4.1 engine delta " + ("checked" if args.check else "applied"))


if __name__ == "__main__":
    main()
