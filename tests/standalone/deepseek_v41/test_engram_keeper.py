# SPDX-License-Identifier: Apache-2.0
"""Table owner readiness and stale-manifest lifetime contracts."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import pytest


@pytest.fixture
def keeper(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / "tools/keep_deepseek_v41_engram_tables.py"
    spec = importlib.util.spec_from_file_location("engram_keeper_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from vllm_gaudi.ops import deepseek_v41_residency as residency

    events = []

    class Event:
        def set(self):
            pass

        def is_set(self):
            return False

        def wait(self, timeout):
            return True

    class Tables:
        reports = [{"ready": True}]
        admission = {"validated": True}

        def __init__(self, owners, budget, device_layers):
            assert device_layers == (1, 14)

        def start(self, cancel):
            events.append("prepared")

        def worker_bindings(self):
            assert events == ["prepared"]
            return {"validated": True}

        def close(self):
            events.append("closed")

    monkeypatch.setattr(module.threading, "Event", Event)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(residency, "EngramResidency", Tables)
    monkeypatch.setattr(residency, "table_regions", lambda *args: [{"layer": 1}, {"layer": 14}])
    monkeypatch.setattr(sys, "argv", [str(source), "--prepared", str(tmp_path), "--output", str(tmp_path)])
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    return module, events


def test_readiness_after_preparation_and_clean_retirement(keeper, tmp_path, monkeypatch):
    module, events = keeper
    address = str(tmp_path / "notify")
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener:
        listener.bind(address)
        listener.settimeout(1)
        monkeypatch.setenv("NOTIFY_SOCKET", address)
        module.main()
        assert listener.recv(64) == b"READY=1"
    assert events == ["prepared", "closed"]
    assert not (tmp_path / "ready.json").exists()
    assert json.loads((tmp_path / "preparation.json").read_text())["admission"]["validated"]


def test_live_legacy_owner_is_not_overwritten(keeper, tmp_path):
    module, events = keeper
    record = dict(pid=os.getpid(), bindings={"already": "owned"})
    (tmp_path / "ready.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="live backing"):
        module.main()
    assert not events
    assert json.loads((tmp_path / "ready.json").read_text()) == record


def test_previous_boot_rebuilds_fresh_backing(keeper, tmp_path):
    module, events = keeper
    (tmp_path / "ready.json").write_text(json.dumps(dict(pid=os.getpid(), boot_id="previous-boot")))
    module.main()
    assert events == ["prepared", "closed"]
    assert not (tmp_path / "ready.json").exists()
