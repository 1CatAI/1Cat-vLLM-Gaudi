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
