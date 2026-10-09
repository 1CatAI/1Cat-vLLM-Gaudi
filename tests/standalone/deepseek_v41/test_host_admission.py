# SPDX-License-Identifier: Apache-2.0
"""A waiting model cannot reserve cards without its loading memory budget."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock


def launcher():
    path = Path(__file__).resolve().parents[3] / 'tools/run_deepseek_v41.py'
    spec = importlib.util.spec_from_file_location('leased_launcher_admission', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_insufficient_memory_never_acquires_cards(tmp_path, monkeypatch):
    module = launcher()
    acquire = Mock()
    monkeypatch.setattr(module, 'host_available_gib', lambda: 90.)
    monkeypatch.setattr(module, 'acquire', acquire)
    assert module.acquire_admitted(tmp_path, 4, (2, 6, 7, 3), (), 350., tmp_path) == (None, [])
    acquire.assert_not_called()
    assert not json.loads((tmp_path / 'host-admission.json').read_text())['owns_cards']


def test_racing_memory_drop_releases_unused_leases(tmp_path, monkeypatch):
    module = launcher()
    memory = iter((360., 100.))
    held = [Mock(), Mock()]
    monkeypatch.setattr(module, 'host_available_gib', lambda: next(memory))
    monkeypatch.setattr(module, 'acquire', lambda *args: ([dict(module=2)], held))
    assert module.acquire_admitted(tmp_path, 1, (2,), (), 350., tmp_path) == (None, [])
    for stream in held:
        stream.close.assert_called_once()


def test_ready_budget_retains_the_same_rank_order(tmp_path, monkeypatch):
    module = launcher()
    selected = [dict(module=i) for i in (2, 6, 7, 3)]
    held = [Mock()]
    monkeypatch.setattr(module, 'host_available_gib', lambda: 360.)
    monkeypatch.setattr(module, 'acquire', lambda *args: (selected, held))
    assert module.acquire_admitted(tmp_path, 4, (2, 6, 7, 3), (), 350., tmp_path) == (selected, held)
    held[0].close.assert_not_called()


def test_long_scratch_path_is_shortened_before_socket_binding():
    module = launcher()
    requested = Path('/opt/optane') / ('long-directory-' * 8)
    actual = module.ipc_scratch_path(requested, '/a/unique/measurement')
    assert actual != requested
    assert len(str(actual).encode()) + 37 <= 107
    assert actual != module.ipc_scratch_path(requested, '/a/different/measurement')


def test_existing_short_scratch_is_preserved():
    module = launcher()
    assert module.ipc_scratch_path('/opt/optane/tmp/12345678', '/identity') == Path('/opt/optane/tmp/12345678')
