# SPDX-License-Identifier: Apache-2.0
"""Concurrent multi-card selectors must not split each other's free set."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from tools import run_deepseek_v41 as launcher


def test_project_namespaces_share_acquisition_guard(monkeypatch, tmp_path):
    locks, evidence = tmp_path / "locks", tmp_path / "evidence"
    locks.mkdir()
    evidence.mkdir()
    entered, finish = Event(), Event()
    calls = []

    def probe(directory, count, requested, secondary):
        calls.append(directory)
        entered.set()
        assert finish.wait(5)
        return ["four reserved modules"], ["owned lease handles"]

    monkeypatch.setattr(launcher, "_acquire_modules", probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(launcher.acquire, locks, 4)
        try:
            assert entered.wait(5)
            assert launcher.acquire(evidence, 4) == (None, [])
            assert calls == [locks]
        finally:
            finish.set()
        assert pending.result() == (["four reserved modules"], ["owned lease handles"])
    # Returning module leases releases the acquisition guard immediately.
    # Active job lifetime remains protected by the individual module leases.
    assert launcher.acquire(evidence, 4) == (["four reserved modules"], ["owned lease handles"])
    assert calls == [locks, evidence]
