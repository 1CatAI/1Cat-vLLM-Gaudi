# SPDX-License-Identifier: Apache-2.0
"""Local serving installation: device ownership and persistent recipe cache."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

_leases = []


def prepare_serving_resources(settings, model, arguments):
    lock_dir = settings.get("device_lock_dir")
    if lock_dir and not _leases:
        for module in os.environ.get("HABANA_VISIBLE_MODULES", "").split(","):
            if not module.isdecimal():
                raise ValueError("Serving installation must select physical HPU module IDs")
            paths = set(Path(lock_dir).glob(f"*module{module}.lock"))
            paths.add(Path(lock_dir) / f"gaudi-module{module}.lock")
            for path in sorted(paths):
                handle = path.open("a")
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _leases.append(handle)
            devices = [
                p for p in Path("/sys/class/accel").glob("accel[0-9]*")
                if (p / "device/module_id").read_text().strip() == module
            ]
            if len(devices) != 1:
                raise RuntimeError(f"Cannot resolve HPU module {module}")
            result = subprocess.run(["fuser", f"/dev/accel/{devices[0].name}"], capture_output=True)
            if result.returncode != 1 or result.stdout or result.stderr:
                raise RuntimeError(f"HPU module {module} already has an owner")
    if cpus := settings.get("cpus"):
        os.sched_setaffinity(0, set(cpus))
    if cache_dir := settings.get("recipe_cache_dir"):
        package = Path(__file__).resolve().parents[1]
        engine = Path(importlib.util.find_spec("vllm").origin).resolve().parents[1]
        sources = {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob("*.py")
        }
        engine_patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=engine)
        identity = {
            "sources": sources,
            "arguments": arguments,
            "engine_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine).decode().strip(),
            "engine_diff": hashlib.sha256(engine_patch).hexdigest(),
            "runtime": os.environ.get("DSV41_SERVING_RUNTIME"),
            "model": hashlib.sha256((Path(model) / "manifest.json").read_bytes()).hexdigest()
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(cache_dir) / digest
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{cache / 'rank{rank}'},false,8192,false"
        (cache / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
