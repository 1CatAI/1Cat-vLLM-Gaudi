# SPDX-License-Identifier: Apache-2.0
"""Configured Engram native binaries must satisfy both hash and ABI contracts."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops import deepseek_v41_host as host


@pytest.mark.parametrize("failure", [None, "hash", "abi", "ambiguous"])
def test_configured_host_binary_validation(tmp_path, monkeypatch, failure):
    binary = tmp_path / "dsv41_host_gather.test.so"
    binary.write_bytes(b"test fixture")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    (tmp_path / "deepseek_v41_build.json").write_text(
        json.dumps({
            "binaries": {
                binary.name: digest if failure != "hash" else "wrong"
            },
            "host_gather_abi_version": 1,
            "host_c1_abi_version": 1,
        }))
    if failure == "ambiguous":
        (tmp_path / "dsv41_host_gather.other.so").write_bytes(b"other")
    module = SimpleNamespace(abi_version=1, c1_abi_version=2 if failure == "abi" else 1)
    loaded = []
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda value: loaded.append(value)))
    monkeypatch.setattr(host.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(host.importlib.util, "module_from_spec", lambda value: module)
    monkeypatch.setenv("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR", str(tmp_path))
    if failure:
        with pytest.raises(RuntimeError, match="ABI" if failure == "abi" else "binary"):
            host.host_native()
        assert len(loaded) == int(failure == "abi")
    else:
        assert host.host_native() is module
        assert host.host_native() is module
        assert loaded == [module]
    host._configured_host_native.cache_clear()
