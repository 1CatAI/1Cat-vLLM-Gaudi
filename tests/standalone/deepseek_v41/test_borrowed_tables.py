# SPDX-License-Identifier: Apache-2.0
"""Shared immutable backing retains independent ownership and fails closed."""
import fcntl
import json
import os

import pytest

from vllm_gaudi.ops.deepseek_v41_borrowed_tables import BorrowedEngramTables


def fixture_table(tmp_path, monkeypatch, *, sealed=True):
    from vllm_gaudi.ops import deepseek_v41_residency

    source = tmp_path / "checkpoint"
    source.write_bytes(b"immutable payload")
    region = dict(file=str(source), offset=0, length=source.stat().st_size)
    monkeypatch.setattr(deepseek_v41_residency, "table_regions", lambda model: [dict(tp=0, layer=1, regions=[region])])
    fd = os.memfd_create("engram-test", os.MFD_ALLOW_SEALING)
    os.write(fd, source.read_bytes())
    if sealed:
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
    stat, target = source.stat(), os.fstat(fd)
    record = dict(file=f"/proc/{os.getpid()}/fd/{fd}",
                  source=region,
                  offset=0,
                  length=region["length"],
                  backing="shared_memfd",
                  device=target.st_dev,
                  inode=target.st_ino,
                  source_identity=dict(device=stat.st_dev,
                                       inode=stat.st_ino,
                                       size=stat.st_size,
                                       mtime_ns=stat.st_mtime_ns))
    manifest = tmp_path / "ready.json"
    manifest.write_text(json.dumps(dict(bindings={"0": {"1": [record]}})))
    return source, fd, manifest


def test_retained_descriptor_survives_keeper_retirement(tmp_path, monkeypatch):
    source, fd, manifest = fixture_table(tmp_path, monkeypatch)
    lease = BorrowedEngramTables(tmp_path, manifest)
    os.close(fd)
    assert lease.reused_bytes == source.stat().st_size
    with open(lease.bindings["0"]["1"][0]["file"], "rb") as stream:
        assert stream.read() == source.read_bytes()
    lease.close()
    assert not lease.descriptors


def test_unsealed_backing_is_rejected(tmp_path, monkeypatch):
    _, fd, manifest = fixture_table(tmp_path, monkeypatch, sealed=False)
    try:
        with pytest.raises(RuntimeError, match="identity"):
            BorrowedEngramTables(tmp_path, manifest)
    finally:
        os.close(fd)


def test_changed_source_is_rejected(tmp_path, monkeypatch):
    source, fd, manifest = fixture_table(tmp_path, monkeypatch)
    source.write_bytes(b"changed checkpoint")
    try:
        with pytest.raises(RuntimeError, match="source changed"):
            BorrowedEngramTables(tmp_path, manifest)
    finally:
        os.close(fd)
