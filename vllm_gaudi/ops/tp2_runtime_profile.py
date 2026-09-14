# SPDX-License-Identifier: Apache-2.0
"""Validate explicitly pinned companion libraries in the current process."""

import hashlib
import json
import os
from pathlib import Path


def _mapped_libraries():
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped.add(Path(fields[5].removesuffix(" (deleted)")).resolve())
    return mapped


def verify_loaded_profile_libraries():
    configured = os.getenv("DSV41_RUNTIME_PROFILE")
    if not configured:
        return None
    profile_path = Path(configured)
    profile = json.loads(profile_path.read_text())
    mapped = _mapped_libraries()
    loaded, pending, configurations = [], [], []
    for item in profile.get("configuration_files", []):
        path = Path(item["path"]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Runtime configuration fingerprint changed: {path}")
        configurations.append({"path": str(path), "sha256": digest})
    for item in profile.get("additional_libraries", []):
        expected = Path(item["path"]).resolve()
        candidates = {path for path in mapped if path.name == expected.name}
        if not candidates:
            pending.append(str(expected))
            continue  # Some profiler/parser libraries load only at start/stop.
        if candidates != {expected}:
            raise RuntimeError(f"Runtime profile library {expected.name} loaded from {sorted(map(str, candidates))}")
        with expected.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Loaded runtime profile library fingerprint changed: {expected}")
        loaded.append({"path": str(expected), "sha256": digest})
    return {"profile": str(profile_path), "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "loaded": loaded, "not_loaded_at_sample": pending, "configurations": configurations}
