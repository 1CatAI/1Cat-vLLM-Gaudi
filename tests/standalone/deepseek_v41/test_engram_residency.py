# SPDX-License-Identifier: Apache-2.0
import json
import os
import hashlib
import fcntl
import threading

import pytest

from vllm_gaudi.ops.deepseek_v41_host import resident_table_source
from vllm_gaudi.ops.deepseek_v41_residency import (EngramResidency, EngramStartup, LockedMapping, locked_file_reuse,
                                                   shared_host_region, table_regions, wait_for_resident_tables)


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


def test_reuse_credits_real_locked_pages_once_and_excludes_device_copies(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(b"a" * 16384)
    region = owner(path)["regions"][0]
    first, second = LockedMapping(region), LockedMapping(region)
    try:
        audit = locked_file_reuse([owner(path), owner(path)])
        assert audit["reused_bytes"] == 12288
        assert len([item for item in audit["mappings"] if item["pid"] == os.getpid()]) == 2
        assert locked_file_reuse([dict(owner(path), shared_memory=True)])["reused_bytes"] == 0
    finally:
        first.close()
        second.close()
    assert locked_file_reuse([owner(path)])["reused_bytes"] == 0


@pytest.mark.parametrize("permissions,rss_kib,flags,expected", [
    ("r--s", 16, "rd sh lo", 12288),
    ("r--s", 16, "rd sh", 0),
    ("r--s", 12, "rd sh lo", 0),
    ("r--p", 16, "rd lo", 0),
    ("rw-s", 16, "rd wr sh lo", 0),
])
def test_reuse_requires_shared_readonly_fully_resident_locked_extent(tmp_path, permissions, rss_kib, flags, expected):
    path = tmp_path / "table.bin"
    path.write_bytes(b"a" * 16384)
    stat = path.stat()
    proc = tmp_path / "proc"
    entry = proc / "123"
    entry.mkdir(parents=True)
    header = (f"10000-14000 {permissions} 00000000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} "
              f"{stat.st_ino} {path}\n")
    (entry / "maps").write_text(header)
    (entry / "smaps").write_text(header + f"Rss: {rss_kib} kB\nVmFlags: {flags}\n")
    audit = locked_file_reuse([owner(path)], proc_root=proc)
    assert audit["reused_bytes"] == expected
    if expected:
        assert audit["mappings"] == [
            dict(pid=123,
                 device_inode=(os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino),
                 offset=0,
                 length=12288)
        ]


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


def test_device_backing_preserves_extent_and_checkpoint(tmp_path):
    path = tmp_path / "table.bin"
    contents = bytes(range(256)) * 70
    path.write_bytes(contents)
    original = path.stat()
    region = dict(file=str(path), offset=13, length=16387)
    descriptor, binding = shared_host_region(region)
    try:
        assert os.pread(descriptor, region["length"], 0) == contents[13:16400]
        assert binding["sha256"] == hashlib.sha256(contents[13:16400]).hexdigest()
        item = dict(file=str(path), shard_offset=13, shard_bytes=16387, row_bytes=1)
        replacement = resident_table_source(item, binding)
        assert replacement["file"] == binding["file"]
        assert replacement["shard_offset"] == 0
        assert replacement["shared_memfd"]
        assert replacement["row_bytes"] == 1
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & fcntl.F_SEAL_SHRINK
        with pytest.raises(OSError):
            os.ftruncate(descriptor, region["length"] - 1)
        with pytest.raises(RuntimeError, match="identity or extent"):
            resident_table_source(item, dict(binding, inode=binding["inode"] + 1))
        with pytest.raises(RuntimeError, match="checkpoint extent"):
            resident_table_source(dict(item, shard_offset=14), binding)
        assert path.read_bytes() == contents
        assert path.stat().st_mtime_ns == original.st_mtime_ns
        path.write_bytes(contents[:-1])
        with pytest.raises(RuntimeError, match="source changed"):
            resident_table_source(item, binding)
    finally:
        os.close(descriptor)


def test_device_residency_shares_one_backing_and_releases_it(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(bytes(range(256)) * 64)
    lease = EngramResidency([owner(path)], budget_bytes=1 << 20, device_layers=(1, ))
    lease.start(check_available=False)
    try:
        binding = lease.worker_bindings()["0"]["1"][0]
        assert lease.reports[0]["locked_bytes"] == 8192
        with open(binding["file"], "rb") as table:
            assert table.read() == path.read_bytes()[7:8199]
        assert lease.reports[0]["files"][0]["inode"] == binding["inode"]
    finally:
        lease.close()
    assert not os.path.exists(binding["file"])
    assert not any(process.is_alive() for process in lease.processes)


def test_residency_starts_only_after_all_devices_are_ready(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(b"a" * 16384)
    lease = EngramResidency([owner(path)], budget_bytes=1 << 20, device_layers=(1, ))
    failures, results = [], []
    startup = EngramStartup(lease, [6, 3], timeout=10).start(lambda: failures.append("lost"))
    waiter = threading.Thread(target=lambda: results.append(wait_for_resident_tables(startup.directory, 0, 6)))
    try:
        waiter.start()
        assert not lease.processes
        with pytest.raises(RuntimeError, match="ownership mismatch"):
            wait_for_resident_tables(startup.directory, 1, 7)
        ready = wait_for_resident_tables(startup.directory, 1, 3, timeout=10)
        waiter.join(timeout=5)
        assert len(results) == 1 and results[0] == ready
        assert ready["0"]["1"][0]["backing"] == "shared_memfd"
        assert lease.reports[0]["ready"]
        assert not startup.stopped.wait(.5)
        lease.check()
        assert startup.thread.is_alive()
    finally:
        startup.close()
        waiter.join(timeout=5)
    assert not failures
    assert not startup.directory.exists()
    assert not any(process.is_alive() for process in lease.processes)


def test_residency_startup_cancellation_before_device_registration(tmp_path):
    lease = EngramResidency([owner(tmp_path / "unused")], budget_bytes=1 << 20)
    startup = EngramStartup(lease, [6]).start(lambda: None)
    startup.close()
    assert not startup.thread.is_alive()
    assert not lease.processes
