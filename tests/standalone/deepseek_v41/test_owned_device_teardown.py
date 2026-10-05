# SPDX-License-Identifier: Apache-2.0
"""A completed launcher need not imply immediately released HPU allocations."""
import importlib.util
import json
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[3] / 'tools/deepseek_v41_owned_devices.py'
_spec = importlib.util.spec_from_file_location('owned_devices', _path)
devices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(devices)


def test_waits_for_owned_allocations_without_waiting_for_foreign_cards(monkeypatch, tmp_path):
    samples = iter(['0, 89000 MiB, 0 %\n2, 90000 MiB, 99 %\n',
                    '0, 768 MiB, 0 %\n2, 90000 MiB, 99 %\n'])
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: next(samples))
    monkeypatch.setattr(devices.time, 'sleep', lambda _: None)
    record = tmp_path/'release.json'
    result = devices.wait_for_released_modules((0,), record)
    assert '0, 768 MiB' in result
    assert [row['pending'] for row in json.loads(record.read_text())] == [[0], []]


def test_timeout_leaves_busy_allocations_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: '0, 89000 MiB, 0 %\n')
    with pytest.raises(RuntimeError, match='ownership must be checked'):
        devices.wait_for_released_modules((0,), tmp_path/'release.json', timeout_s=0)


def test_release_monitor_keeps_unknown_owned_module_pending(monkeypatch, tmp_path):
    samples = iter(['N/A, N/A MiB, N/A %\n', '2, N/A MiB, N/A %\n', '2, 768 MiB, 0 %\n'])
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: next(samples))
    monkeypatch.setattr(devices.time, 'sleep', lambda _: None)
    record = tmp_path/'release.json'
    devices.wait_for_released_modules((2,), record)
    assert [row['pending'] for row in json.loads(record.read_text())] == [[2], [2], []]


def test_selects_other_free_cards_without_a_lease(monkeypatch, tmp_path):
    load = ''.join(f'{module}, {90000 if module in (0, 1, 4, 5) else 768} MiB, 0 %\n' for module in range(8))
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: load)
    selected, actual = devices.wait_for_free_modules(tmp_path/'availability.json', owners=lambda _: [])
    assert selected == (2, 3, 6, 7)
    assert actual == load


def test_open_worker_is_not_free_even_before_hbm_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: '0, 768 MiB, 0 %\n')
    samples = iter([[99], []])
    monkeypatch.setattr(devices.time, 'sleep', lambda _: None)
    record = tmp_path/'availability.json'
    selected, _ = devices.wait_for_free_modules(record, count=1, owners=lambda _: next(samples))
    assert selected == (0,)
    assert [row['free'] for row in json.loads(record.read_text())] == [[], [0]]


def test_explicit_mapping_does_not_use_busy_or_duplicate_cards(monkeypatch, tmp_path):
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: '0, 768 MiB, 0 %\n1, 89000 MiB, 0 %\n')
    with pytest.raises(RuntimeError, match='Not enough free modules'):
        devices.wait_for_free_modules(tmp_path/'availability.json', count=1, modules=(1,), timeout_s=0,
                                      owners=lambda _: [])
    with pytest.raises(ValueError, match='distinct'):
        devices.wait_for_free_modules(tmp_path/'availability.json', count=2, modules=(0, 0), owners=lambda _: [])


def test_maps_compute_handle_to_physical_module(monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = tmp_path/'sysfs/accel0/device/module_id'
    module.parent.mkdir(parents=True)
    module.write_text('2\n')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='123 456', stderr='')

    monkeypatch.setattr(devices.subprocess, 'run', run)
    assert devices.device_owners(2, sysfs=tmp_path/'sysfs', device_root=tmp_path/'devices') == [123, 456]
    assert calls == [['fuser', str(tmp_path/'devices/accel0')]]


def test_lease_respects_all_existing_aliases_and_releases(monkeypatch, tmp_path):
    import fcntl
    load = '0, 768 MiB, 0 %\n1, 768 MiB, 0 %\n'
    monkeypatch.setattr(devices.subprocess, 'check_output', lambda *a, **k: load)
    monkeypatch.setattr(devices, 'device_owners', lambda _: [])
    # The original default function in wait_for_free_modules is bound at import.
    select = devices.wait_for_free_modules
    monkeypatch.setattr(devices, 'wait_for_free_modules',
                        lambda *a, **k: select(*a, **k, owners=lambda _: []))
    lock_dir = tmp_path / 'locks'
    lock_dir.mkdir()
    with (lock_dir / 'gaudi-module0.lock').open('a+') as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with devices.lease_free_modules(tmp_path / 'lease.json', lock_dir=lock_dir,
                                         count=1, preferred=(0,)) as (selected, _):
            assert selected == (1,)
            for name in ('module-1.lock', 'module1.lock', 'gaudi-module1.lock'):
                with (lock_dir / name).open('a+') as probe:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (lock_dir / 'module1.lock').open('a+') as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with devices.lease_free_modules(tmp_path / 'second.json', lock_dir=lock_dir,
                                     count=1, preferred=(0,)) as (selected, _):
        assert selected == (0,)


def test_retirement_checks_identity_and_separate_sessions(monkeypatch):
    from types import SimpleNamespace
    rows = {10: ('S', 1, 10, 100), 11: ('S', 1, 11, 110),
            12: ('S', 1, 12, 999)}  # PID reused by foreign task.
    signals = []
    def kill(pid, sig):
        signals.append((pid, sig))
        rows.pop(pid)
    monkeypatch.setattr(devices, '_process_identity', rows.get)
    monkeypatch.setattr(devices.os, 'kill', kill)
    process = SimpleNamespace(pid=10, poll=lambda: 0, wait=lambda: 0)
    devices.retire_process_group(process, owned={10:100, 11:110, 12:120})
    assert [pid for pid, sig in signals] == [10, 11]
    assert 12 in rows


def test_retirement_does_not_silently_release_live_worker(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(devices, '_process_identity',
                        lambda pid: ('D', 1, pid, pid*10))
    monkeypatch.setattr(devices.os, 'kill', lambda *args: None)
    process = SimpleNamespace(pid=10, poll=lambda: 0, wait=lambda: 0)
    with pytest.raises(RuntimeError, match='still retiring'):
        devices.retire_process_group(process, owned={10:100, 11:110}, grace_s=0)
