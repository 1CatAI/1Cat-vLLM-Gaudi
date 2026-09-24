# SPDX-License-Identifier: Apache-2.0
"""Rank-local loading must not synchronously evict each copied file chunk."""

import os
from types import SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def _forbid_fadvise(*_args):
    raise AssertionError("serving loader synchronously evicted its file cache")


def test_prepared_tensor_keeps_cache_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "posix_fadvise", _forbid_fadvise)
    path = tmp_path / "rank"
    path.write_bytes(bytes(range(16)))
    shard = PreparedV41Shard.__new__(PreparedV41Shard)
    shard.path = path
    shard.catalog = {"small": SimpleNamespace(dtype="U8", shape=(16,), offset=0)}
    shard.check_identity = lambda: None
    shard.max_host_chunk_bytes = 0

    assert shard.tensor("small", "cpu").tolist() == list(range(16))
    assert shard.max_host_chunk_bytes == 16


def test_dense_and_engram_sidecar_do_not_evict_each_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "posix_fadvise", _forbid_fadvise)
    path = tmp_path / "rank"
    path.write_bytes(bytes([0x38]) * (32 * 32) + bytes([127]))
    shard = PreparedV41Shard.__new__(PreparedV41Shard)
    shard.path = path
    shard.catalog = {
        "dense.weight": SimpleNamespace(dtype="F8_E4M3", shape=(32, 32), offset=0),
        "dense.scale": SimpleNamespace(dtype="U8", shape=(1, 1), offset=32 * 32),
    }
    shard.check_identity = lambda: None
    shard.max_host_chunk_bytes = 0
    assert torch.equal(shard.dense("dense.weight", "cpu"), torch.ones((32, 32), dtype=torch.bfloat16))

    sidecar = EngramFP8Sidecar.__new__(EngramFP8Sidecar)
    sidecar.path = path
    sidecar.catalog = {"scale": SimpleNamespace(dtype="F32", shape=(1,), offset=0)}
    sidecar._check_identity = lambda: None
    sidecar.max_host_chunk_bytes = 0
    assert sidecar.tensor("scale", "cpu").numel() == 1
