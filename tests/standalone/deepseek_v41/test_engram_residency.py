# SPDX-License-Identifier: Apache-2.0
import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from vllm_gaudi.ops.deepseek_v41_residency import (
    EngramDeviceGate,
    EngramResidency,
    LockedMapping,
    locked_shared_bytes,
    table_regions,
    wait_for_engram_residency,
)


def owner(path):
    return dict(tp=0, layer=1, regions=[dict(file=str(path), offset=7, length=8192)])


def test_locked_pages_and_mapping_bounds(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(bytes(range(256)) * 64)
    mapping = LockedMapping(owner(path)["regions"][0])
    try:
        assert mapping.resident_pages() * os.sysconf("SC_PAGESIZE") == mapping.length
    finally:
        mapping.close()
    with pytest.raises(ValueError, match="exceeds"):
        LockedMapping(dict(file=str(path), offset=16000, length=8192))


def test_residency_owners_release_and_failure_is_visible(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(b"a" * 16384)
    lease = EngramResidency([owner(path)], budget_bytes=1 << 20).start(check_available=False)
    processes = list(lease.processes)
    try:
        assert lease.reports[0]["ready"]
        assert lease.reports[0]["locked_bytes"] == 12288
        lease.check()
        processes[0].terminate()
        processes[0].join(timeout=5)
        with pytest.raises(RuntimeError, match="lost"):
            lease.check()
    finally:
        lease.close()
    assert all(not process.is_alive() for process in processes)


def test_budget_failure_does_not_start_owners(tmp_path):
    lease = EngramResidency([owner(tmp_path / "unused")], budget_bytes=4096)
    with pytest.raises(RuntimeError, match="budget"):
        lease.start(check_available=False)
    assert not lease.processes


def test_regions_use_original_offsets_and_validate_manifest(tmp_path):
    table = dict(file="/frozen/source", row_start=4, row_stop=8, row_bytes=256, shard_offset=321, shard_bytes=1024)
    shard = dict(tp_rank=0,
                 pp_owner=0,
                 shared_read_only=True,
                 sharding="complete_hash_heads",
                 model_revision="frozen",
                 tables={
                     "layers.1.engram.embed.weight": table,
                     "layers.1.engram.embed.scale": dict(table, row_bytes=8, shard_bytes=32, shard_offset=99)
                 })
    path = tmp_path / "tp0.json"
    path.write_text(json.dumps(shard))
    manifest = dict(
        model_revision="frozen",
        engram_host_shards={"0": dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = table_regions(tmp_path)
    assert len(result) == 1 and result[0]["regions"][0]["offset"] == 321
    assert result[0]["regions"][1]["length"] == 32
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="manifest changed"):
        table_regions(tmp_path)


def test_shared_locked_extents_are_counted_once(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(b"a" * 32768)
    stat = path.stat()
    proc = tmp_path / "proc"
    proc.mkdir()

    def mapping(pid, offset, size, *, shared=True, resident=True, locked=True):
        process = proc / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(b"python\0multiprocessing.spawn\0")
        header = (f"100000-{0x100000 + size:x} r--{'s' if shared else 'p'} {offset:08x} "
                  f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x} {stat.st_ino} {path}\n")
        (process / "smaps").write_text(header + f"Rss: {size // 1024 if resident else 0} kB\n" +
                                       f"Locked: {size // 2048 if locked else 0} kB\n" +
                                       f"VmFlags: rd mr me ms {'lo' if locked else ''}\n")

    mapping(100, 0, 8192)
    mapping(101, 4096, 8192)
    mapping(102, 8192, 8192, locked=False)
    mapping(103, 8192, 8192, shared=False)
    mapping(104, 8192, 8192, resident=False)
    # Requested offset is not page aligned; partial overlaps and duplicate
    # proportional Locked accounting must still charge each physical page once.
    owners = [dict(regions=[dict(file=str(path), offset=4097, length=12287)])]
    assert locked_shared_bytes(owners, proc) == 8192


def test_disappeared_or_unlocked_owner_gets_no_credit(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(b"a" * 8192)
    proc = tmp_path / "proc"
    proc.mkdir()
    process = proc / "100"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0multiprocessing.spawn\0")
    assert locked_shared_bytes([dict(regions=[dict(file=str(path), offset=0, length=8192)])], proc) == 0


class _Residency:
    budget_bytes = 123
    reports = []

    def __init__(self, fail=False):
        self.started = threading.Event()
        self.closed = False
        self.fail = fail

    def start(self, **kwargs):
        self.started.set()
        if self.fail:
            raise RuntimeError("test lock budget unavailable")
        return self

    def check(self):
        assert not self.closed

    def close(self):
        self.closed = True


def test_device_gate_waits_for_all_workers_before_locking(monkeypatch):
    monkeypatch.delenv("DSV41_ENGRAM_DEVICE_GATE", raising=False)
    lease = _Residency()
    gate = EngramDeviceGate(lease, world_size=2, timeout=5).start()
    try:
        with ThreadPoolExecutor(2) as workers:
            first = workers.submit(wait_for_engram_residency, 0, 2, timeout=5)
            assert not lease.started.wait(.15)
            assert not first.done()
            second = workers.submit(wait_for_engram_residency, 1, 2, timeout=5)
            first.result(timeout=5)
            second.result(timeout=5)
        assert lease.started.is_set()
        assert len(json.loads((gate.path / "ready.json").read_text())["initialized_workers"]) == 2
    finally:
        gate.close()
    assert lease.closed and not gate.path.exists()
    assert "DSV41_ENGRAM_DEVICE_GATE" not in os.environ


def test_device_gate_propagates_lock_failure_and_preserves_world_contract(monkeypatch):
    monkeypatch.delenv("DSV41_ENGRAM_DEVICE_GATE", raising=False)
    lease = _Residency(fail=True)
    gate = EngramDeviceGate(lease, world_size=1, timeout=5).start()
    try:
        with pytest.raises(RuntimeError, match="world mismatch"):
            wait_for_engram_residency(0, 2)
        with pytest.raises(RuntimeError, match="lock budget"):
            wait_for_engram_residency(0, 1, timeout=5)
        assert not (gate.path / "ready.json").exists()
    finally:
        gate.close()
    assert lease.closed


def test_cancel_before_devices_does_not_pin_tables(monkeypatch):
    monkeypatch.delenv("DSV41_ENGRAM_DEVICE_GATE", raising=False)
    lease = _Residency()
    gate = EngramDeviceGate(lease, timeout=5).start()
    gate.close()
    assert not lease.started.is_set() and lease.closed
