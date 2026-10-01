# SPDX-License-Identifier: Apache-2.0
"""Serving installation ownership and runtime relocation contracts, CPU only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_relocation_keeps_dependency_hashes_and_path_boundaries():
    module = tool("install_deepseek_v41_runtime")
    original = {"eager_runtime": [{"path": "/old/python/lib/backend.so", "sha256": "abc"}],
                "native": "/old/native/libhcl.so", "description": "/old/native-other/not-an-artifact"}
    relocated = module.relocate(original, {"/old": "/installation", "/old/python": "/venv",
                                          "/old/native": "/runtime"})
    assert relocated["eager_runtime"] == [{"path": "/venv/lib/backend.so", "sha256": "abc"}]
    assert relocated["native"] == "/runtime/libhcl.so"
    assert relocated["description"] == "/installation/native-other/not-an-artifact"
    assert original["native"] == "/old/native/libhcl.so"


def test_cpu_supervisor_pins_new_threads_and_skips_unowned_roles(monkeypatch):
    module = tool("serve_deepseek_v41")
    monkeypatch.setattr(module, "service_processes", lambda pid: {
        100: "python", 101: "VLLM::EngineCore", 102: "VLLM::Worker_TP0", 103: "resource_tracker"})
    class Tasks:
        def __init__(self, path):
            self.pid = int(str(path).split("/")[2])
        def iterdir(self):
            return [SimpleNamespace(name=str(self.pid)), SimpleNamespace(name=str(self.pid + 1000))]
    monkeypatch.setattr(module, "Path", Tasks)
    current = {pid: {9} for pid in (100, 101, 102, 1100, 1101, 1102)}
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: current[pid])
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda pid, cpus: current.__setitem__(pid, cpus))
    settings = {"cpu_allocation": {"worker_main": [0], "worker_helpers": [[1, 2]],
                                  "engine_main": [3], "api_main": [4], "control_helpers": [5, 6]}}
    module.maintain_affinity(100, settings)
    assert current == {100: {4}, 101: {3}, 102: {0}, 1100: {5, 6}, 1101: {5, 6}, 1102: {1, 2}}
    current[1102] = {0}
    module.maintain_affinity(100, settings)
    assert current[1102] == {1, 2}
