# SPDX-License-Identifier: Apache-2.0
"""Rank-local loading must preserve storage bytes and reject stale bindings."""

import json

import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, canonical_hash, file_hash


def make_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    specs = {"weight": {"dtype": "BF16", "shape": [4, 8]}, "scale": {"dtype": "U8", "shape": [4, 8]}}
    fields = {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "pp_layer_ranges": [[0, 20], [20, 40]],
        "prepared_layout_version": 2,
        "model_revision": "test",
        "quantization_fingerprint": "quant",
        "upstream_lock_sha256": "upstream",
        "metadata_sha256": {
            "config.json": file_hash(tmp_path / "config.json")
        },
        "encoding_sha256": {}
    }
    plan = {**fields, "ranks": {"pp0-tp0": specs}}
    fingerprint = canonical_hash(plan)
    metadata = {
        "model_revision": "test",
        "plan_fingerprint": fingerprint,
        "prepared_layout_version": "2",
        "pp_rank": "0",
        "tp_rank": "0"
    }
    path = tmp_path / "pp0-tp0.safetensors"
    writer = RankWriter(path, specs, metadata)
    bits = np.array([0, 0x8000, 0x3f80, 0x7fc0, 0x7f80, 0xff80, 1, 0x8001] * 4, dtype=np.uint16).reshape(4, 8)
    codes = np.arange(32, dtype=np.uint8).reshape(4, 8)
    writer.write("weight", 0, bits)
    writer.write("scale", 0, codes)
    writer.sync()
    writer.close()
    stat = path.stat()
    record = {
        "file": path.name,
        "bytes": stat.st_size,
        "sha256": file_hash(path),
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns
    }
    (tmp_path / "preparation-plan.json").write_text(json.dumps(plan))
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            **fields, "plan_fingerprint": fingerprint,
            "rank_files": {
                "pp0-tp0": record
            }
        }))
    return bits, codes


def test_direct_load_binds_prepared_bytes_without_requantization(tmp_path):
    bits, codes = make_checkpoint(tmp_path)
    shard = PreparedV41Shard(tmp_path, 0, 0)
    model = torch.nn.Module()
    model.register_parameter("w", torch.nn.Parameter(torch.empty(4, 8, dtype=torch.bfloat16, device="meta")))
    model.register_buffer("s", torch.empty(4, 8, dtype=torch.uint8, device="meta"))
    shard.bind(model, {"weight": "w", "scale": "s"}, "cpu")
    assert np.array_equal(model.w.detach().view(torch.int16).numpy().view(np.uint16), bits)
    assert np.array_equal(model.s.numpy(), codes)
    assert shard.max_host_chunk_bytes == 64


def test_prefetched_chunks_preserve_order_bytes_and_metrics(tmp_path, monkeypatch):
    import vllm_gaudi.ops.deepseek_v41_shard_loader as loader

    bits, _ = make_checkpoint(tmp_path)
    monkeypatch.setattr(loader, "COPY_BYTES", 16)
    shard = PreparedV41Shard(tmp_path, 0, 0)
    actual = shard.tensor("weight", "cpu")
    assert np.array_equal(actual.view(torch.int16).numpy().view(np.uint16), bits)
    assert shard.max_host_chunk_bytes == 16
    assert shard.startup_timings["source_bytes"] == bits.nbytes
    assert shard.startup_timings["upload_bytes"] == bits.nbytes


def test_loader_rejects_weight_changes_and_wrong_manifests(tmp_path):
    make_checkpoint(tmp_path)
    shard = PreparedV41Shard(tmp_path, 0, 0)
    path = tmp_path / "pp0-tp0.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(RuntimeError, match="changed during loading"):
        shard.tensor("weight", "cpu")
    with pytest.raises(ValueError, match="hash differs"):
        PreparedV41Shard(tmp_path, 0, 0, verify_hash=True)
    with pytest.raises(ValueError, match="TP2"):
        PreparedV41Shard(tmp_path, 0, 2)


@pytest.mark.parametrize("tp,pp,ranges", ((4, 2, [[0, 20], [20, 40]]), (2, 1, [[0, 40]])))
def test_loader_rejects_unimplemented_topologies(tmp_path, tp, pp, ranges):
    make_checkpoint(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(tensor_parallel_size=tp, pipeline_parallel_size=pp, pp_layer_ranges=ranges)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="topology"):
        PreparedV41Shard(tmp_path, 0, 0)


def test_loader_rejects_tp4_manifest_bound_to_tp2_plan(tmp_path):
    make_checkpoint(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(tensor_parallel_size=4, pipeline_parallel_size=1, pp_layer_ranges=[[0, 40]])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor_parallel_size mismatch"):
        PreparedV41Shard(tmp_path, 0, 0)
