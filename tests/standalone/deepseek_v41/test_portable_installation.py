# SPDX-License-Identifier: Apache-2.0
"""Serving installation ownership and runtime relocation contracts, CPU only."""
import importlib.util
import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError(errno.EACCES, "Permission denied")))
    with pytest.raises(OSError, match="Permission denied"):
        module.atomic_json(tmp_path / "process.json", {}, required=False)


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
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), module.handler(
            f"http://127.0.0.1:{unavailable.getsockname()[1]}", "test-key"))
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
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), module.handler(
        f"http://127.0.0.1:{upstream.server_port}", "test-key"))
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
