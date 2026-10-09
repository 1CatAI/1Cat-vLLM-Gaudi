# SPDX-License-Identifier: Apache-2.0
"""Reject stale registrations before a launcher acquires hardware leases."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).parents[3] / "tools/run_deepseek_v41.py"
    spec = importlib.util.spec_from_file_location("dspark_launcher_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hash_mismatch_rejected_before_hardware(launcher, tmp_path):
    binary = tmp_path / "addon.so"
    binary.write_bytes(b"prepared version")
    manifest = tmp_path / "deepseek_v41_unique_build.json"
    manifest.write_text(json.dumps({"binaries": {binary.name: hashlib.sha256(binary.read_bytes()).hexdigest()}}))
    env = {"VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR": str(tmp_path)}
    launcher.validate_native_database_path(env)
    binary.write_bytes(b"different capability build")
    with pytest.raises(ValueError, match="differs from its build manifest"):
        launcher.validate_native_database_path(env)


@pytest.mark.parametrize("name", ("missing.so", "../outside.so"))
def test_missing_or_external_binary_rejected(launcher, tmp_path, name):
    (tmp_path / "deepseek_v41_unique_build.json").write_text(json.dumps({"binaries": {name: "unknown"}}))
    with pytest.raises(ValueError, match="Invalid or missing"):
        launcher.validate_native_database_path({"VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR": str(tmp_path)})
