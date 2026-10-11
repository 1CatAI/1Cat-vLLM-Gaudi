# SPDX-License-Identifier: Apache-2.0
"""Serving installation ownership and runtime relocation contracts, CPU only."""
import importlib.util
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialized_engine_cache_does_not_require_a_git_checkout(tmp_path, monkeypatch):
    from vllm_gaudi.entrypoints import serving_resources as module

    engine = tmp_path / "engine" / "vllm"
    engine.mkdir(parents=True)
    origin = engine / "__init__.py"
    origin.write_text("VERSION = 1\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text("{}")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(origin)))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=128))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: pytest.fail("Git checkout required"))
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "")
    settings = {"recipe_cache_dir": str(tmp_path / "recipes")}
    module.prepare_serving_resources(settings, model, [])
    first = module.os.environ["PT_HPU_RECIPE_CACHE_CONFIG"]
    inventory = Path(first.split(",")[0]).parent / "source-inventory.json"
    previous_inventory = inventory.read_text()
    module.prepare_serving_resources(settings, model, [])
    assert module.os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] == first
    origin.write_text("VERSION = 2\n")
    module.prepare_serving_resources(settings, model, [])
    assert module.os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] == first
    assert inventory.read_text() != previous_inventory


def test_runtime_relocation_keeps_dependency_hashes_and_path_boundaries():
    module = tool("install_deepseek_v41_runtime")
    original = {
        "eager_runtime": [{
            "path": "/old/python/lib/backend.so",
            "sha256": "abc"
        }],
        "native": "/old/native/libhcl.so",
        "description": "/old/native-other/not-an-artifact"
    }
    relocated = module.relocate(original, {"/old": "/installation", "/old/python": "/venv", "/old/native": "/runtime"})
    assert relocated["eager_runtime"] == [{"path": "/venv/lib/backend.so", "sha256": "abc"}]
    assert relocated["native"] == "/runtime/libhcl.so"
    assert relocated["description"] == "/installation/native-other/not-an-artifact"
    assert original["native"] == "/old/native/libhcl.so"


def test_compilation_proof_survives_source_retirement(tmp_path):
    module = tool("install_deepseek_v41_runtime")
    source = tmp_path / "build"
    source.mkdir()
    patch = source / "source.patch"
    patch.write_text("allocation patch")
    proof = source / "RESULT.json"
    proof.write_text(json.dumps(dict(sha256="runtime", source_patch_sha256=module.digest(patch))))
    profile = dict(additional_libraries=[dict(path="/build/libSynapse.so", compilation_proof=dict(
        path=str(proof), sha256=module.digest(proof)))])
    output = tmp_path / "installation"
    records = [dict(path=str(output / "lib/libSynapse.so"), sha256="runtime")]
    module.install_compilation_proofs(profile, output, records)
    installed = records[0]["compilation_proof"]
    proof.unlink()
    patch.unlink()
    assert module.digest(installed["path"]) == installed["sha256"]
    assert (Path(installed["path"]).parent / "source.patch").read_text() == "allocation patch"
    profile["additional_libraries"][0]["compilation_proof"] = installed
    (Path(installed["path"]).parent / "source.patch").write_text("corrupt")
    with pytest.raises(ValueError, match="does not describe"):
        module.install_compilation_proofs(profile, tmp_path / "another", records)


def test_native_loader_entry_does_not_redirect_selected_kernel_database(tmp_path):
    module = tool("install_deepseek_v41_runtime")
    native = tmp_path / "native-build"
    output = tmp_path / "installation"
    kernel = native / "libdeepseek_v41_unique_kernels.so"
    target = module.relocated_library_target(kernel, native, output)
    mapping = {str(native): str(output / "native"), str(kernel): str(target)}
    relocated = module.relocate(dict(GC_KERNEL_PATH=str(kernel), native=str(native)), mapping)
    assert relocated["GC_KERNEL_PATH"] == str(output / "native" / kernel.name)
    assert relocated["native"] == str(output / "native")
    external = tmp_path / "sdk" / "libSynapse.so"
    assert module.relocated_library_target(external, native, output) == output / "lib" / external.name


@pytest.mark.parametrize("error_code", (errno.ENOSPC, errno.EDQUOT))
def test_optional_status_write_survives_full_storage_and_retries(tmp_path, monkeypatch, error_code):
    module = tool("serve_deepseek_v41")
    target = tmp_path / "process.json"
    module.atomic_json(target, {"pid": 1})
    original = Path.write_text
    with monkeypatch.context() as patch:

        def fail_write(path, *args, **kwargs):
            if path == target.with_suffix(".tmp"):
                raise OSError(error_code, "Storage full")
            return original(path, *args, **kwargs)

        patch.setattr(Path, "write_text", fail_write)
        assert module.atomic_json(target, {"pid": 2}, required=False) is False
        assert target.read_text() == '{\n  "pid": 1\n}\n'
        with pytest.raises(OSError):
            module.atomic_json(target, {"pid": 2})
    assert module.atomic_json(target, {"pid": 2}, required=False) is True
    assert target.read_text() == '{\n  "pid": 2\n}\n'


def test_optional_status_write_does_not_hide_permission_errors(tmp_path, monkeypatch):
    module = tool("serve_deepseek_v41")
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs:
                        (_ for _ in ()).throw(OSError(errno.EACCES, "Permission denied")))
    with pytest.raises(OSError, match="Permission denied"):
        module.atomic_json(tmp_path / "process.json", {}, required=False)


def test_settings_override_creates_actual_compiler_scratch_before_launch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import json
    import sys

    module = tool("serve_deepseek_v41")
    installation = tmp_path / "installation"
    installation.mkdir()
    scratch = tmp_path / "scratch" / "compiler"
    override = tmp_path / "candidate-settings.json"
    allocation = {
        "worker_main": [0],
        "worker_helpers": [[1]],
        "engine_main": [2],
        "api_main": [3],
        "control_helpers": [4]
    }
    settings = {"cpus": list(range(5)), "cpu_allocation": allocation, "environment": {"TMPDIR": str(scratch)}}
    override.write_text(json.dumps(settings))
    (installation / "settings.json").write_text("{}")
    captured = {}

    def launch(command, **kwargs):
        assert Path(kwargs["env"]["TMPDIR"]).is_dir()
        captured.update(command=command, environment=kwargs["env"])
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["serve", str(installation), "--settings", str(override)])
    with pytest.raises(SystemExit) as finished:
        module.main()
    assert finished.value.code == 0
    assert captured["command"].count("--settings") == 1
    assert captured["command"][captured["command"].index("--settings") + 1] == str(override)
    from run_deepseek_v41 import ipc_scratch_path

    assert captured["environment"]["TMPDIR"] == str(ipc_scratch_path(str(scratch), installation))


def test_public_adapter_returns_unavailable_when_backend_has_stopped():
    import http.client
    import json
    import socket
    import threading
    from http.server import ThreadingHTTPServer
    module = tool("serve_deepseek_v41_api")
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        # Bound but not listening: this port cannot be reused during the test.
        proxy = ThreadingHTTPServer(("127.0.0.1", 0),
                                    module.handler(f"http://127.0.0.1:{unavailable.getsockname()[1]}", "test-key"))
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            client = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=2)
            client.request("GET", "/v1/models", headers={"Authorization": "Bearer test-key"})
            response = client.getresponse()
            assert response.status == 503
            assert response.getheader("Retry-After") == "5"
            assert response.getheader("Access-Control-Allow-Origin") == "*"
            assert json.loads(response.read())["error"]["code"] == "service_unavailable"
            client.close()
        finally:
            proxy.shutdown()
            proxy.server_close()


def test_cpu_supervisor_pins_new_threads_and_skips_unowned_roles(monkeypatch):
    module = tool("serve_deepseek_v41")
    monkeypatch.setattr(
        module, "service_processes", lambda pid: {
            100: "python",
            101: "VLLM::EngineCore",
            102: "VLLM::Worker_TP0",
            103: "resource_tracker"
        })

    class Tasks:

        def __init__(self, path):
            self.pid = int(str(path).split("/")[2])

        def iterdir(self):
            return [SimpleNamespace(name=str(self.pid)), SimpleNamespace(name=str(self.pid + 1000))]

    monkeypatch.setattr(module, "Path", Tasks)
    current = {pid: {9} for pid in (100, 101, 102, 1100, 1101, 1102)}
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: current[pid])
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda pid, cpus: current.__setitem__(pid, cpus))
    settings = {
        "cpu_allocation": {
            "worker_main": [0],
            "worker_helpers": [[1, 2]],
            "engine_main": [3],
            "api_main": [4],
            "control_helpers": [5, 6]
        }
    }
    module.maintain_affinity(100, settings)
    assert current == {100: {4}, 101: {3}, 102: {0}, 1100: {5, 6}, 1101: {5, 6}, 1102: {1, 2}}
    current[1102] = {0}
    module.maintain_affinity(100, settings)
    assert current[1102] == {1, 2}


def test_public_adapter_forwards_inference_and_hides_internal_routes():
    import http.client
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    module = tool("serve_deepseek_v41_api")

    class Upstream(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer test-key"
            body = b'{"data": [{"id": "model"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), module.handler(f"http://127.0.0.1:{upstream.server_port}",
                                                                 "test-key"))
    for server in (upstream, proxy):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for route, authorized, expected in (("/v1/models", False, 401), ("/start_profile", True, 404),
                                            ("/collective_rpc", True, 404), ("/v1/models", True, 200)):
            connection = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=2)
            connection.request("GET", route, headers={"Authorization": "Bearer test-key"} if authorized else {})
            response = connection.getresponse()
            body = response.read()
            assert response.status == expected
            if expected == 200:
                assert body == b'{"data": [{"id": "model"}]}'
            connection.close()
        for route, expected in (("/v1/chat/completions", 204), ("/collective_rpc", 404)):
            connection = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=2)
            connection.request("OPTIONS", route)
            response = connection.getresponse()
            assert response.status == expected
            assert response.getheader("Access-Control-Allow-Origin") == "*"
            response.read()
            connection.close()
    finally:
        for server in (proxy, upstream):
            server.shutdown()
            server.server_close()


def test_raw_trace_configuration_rejected_before_model_loading():
    module = tool("serve_deepseek_v41")
    with pytest.raises(ValueError, match="HABANA_PROFILE_WRITE_HLTV"):
        module.validate_raw_trace_profile(
            {"environment": {
                "VLLM_HPU_DSV41_RAW_TRACE": "1",
                "HABANA_PROF_CONFIG": "/unused"
            }})
    module.validate_raw_trace_profile({"environment": {"VLLM_HPU_DSV41_RAW_TRACE": "0"}})


@pytest.mark.parametrize("key", ["DUMP_PRE_GRAPHS", "DUMP_POST_GRAPHS"])
def test_zero_dump_directory_rejected_before_model_loading(key):
    module = tool("serve_deepseek_v41")
    with pytest.raises(ValueError, match="dump directory"):
        module.validate_raw_trace_profile({"environment": {key: "0"}})
