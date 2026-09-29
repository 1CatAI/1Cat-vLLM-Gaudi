# SPDX-License-Identifier: Apache-2.0
"""Prepared runtime files must preserve weights and reject stale bindings."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.prepare_deepseek_v41_n256 import prepare_rank
from vllm_gaudi.ops.deepseek_v41_expert_n256 import (
    COMPACT_FINGERPRINT, COMPACT_LAYOUT, FINGERPRINT, LAYOUT, load_projection, prepare_expert, restore_expert)
from vllm_gaudi.ops.deepseek_v41_fp8 import FINGERPRINT as QUANTIZATION_FINGERPRINT
from vllm_gaudi.ops.deepseek_v41_n256_shards import N256PreparedShard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, file_hash, publish_json, read_header


@pytest.fixture(params=[(2, 2, False), (4, 1, False), (4, 1, True)])
def prepared(tmp_path, request):
    tp, pp, compact = request.param
    prefix = "layers.0.ffn.experts.w13"
    rng = np.random.default_rng(31)
    q = rng.integers(-32768, 32768, (3, 4, 4096), dtype=np.int16)
    s = np.full((3, 4, 512), 127 << 7, dtype=np.uint16)
    base = tmp_path / "base"
    base.mkdir()
    source = base / "pp0-tp0.safetensors"
    arrays = {prefix + "_q16": q, prefix + "_s16": s}
    specs = {name: {"dtype": "I16" if name.endswith("_q16") else "BF16", "shape": list(value.shape)}
             for name, value in arrays.items()}
    writer = RankWriter(source, specs, {})
    try:
        for name, value in arrays.items():
            writer.write(name, 0, value)
        writer.sync()
    finally:
        writer.close()
    manifest = {"tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                "rank_files": {"pp0-tp0": {"sha256": file_hash(source)}}}
    publish_json(base / "manifest.json", manifest)
    shard = SimpleNamespace(directory=base, pp_rank=0, tp_rank=0, manifest=manifest,
                            catalog=read_header(source), check_identity=lambda: None)
    output = tmp_path / "runtime"
    output.mkdir()
    rank, record = prepare_rank(shard, output, compact_scales=compact)
    publish_json(output / "manifest.json", {
        "schema_version": 1, "layout": COMPACT_LAYOUT if compact else LAYOUT,
        "layout_fingerprint": COMPACT_FINGERPRINT if compact else FINGERPRINT,
        "quantization_fingerprint": QUANTIZATION_FINGERPRINT,
        "source_manifest_sha256": file_hash(base / "manifest.json"),
        "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
        "rank_files": {rank: record}})
    return output, shard, prefix, q, s


def test_runtime_loader_is_exact_and_skips_preparation(prepared, monkeypatch):
    output, shard, prefix, q, s = prepared
    compact = json.loads((output / "manifest.json").read_text())["layout"] == COMPACT_LAYOUT
    expected = [prepare_expert(a, b, compact_scales=compact)[:3] for a, b in zip(q, s)]
    monkeypatch.setenv("VLLM_HPU_DSV41_N256_PREPARED_DIR", str(output))

    def forbidden(*args):
        raise AssertionError("runtime must not recompute prepared weights")

    monkeypatch.setattr("vllm_gaudi.ops.deepseek_v41_expert_n256.prepare_expert", forbidden)
    actual = load_projection(shard, prefix, "cpu")
    for index, tensor in enumerate(actual):
        bits = tensor.view(torch.int16).numpy().view(np.uint16)
        wanted = np.stack([row[index] for row in expected]).view(np.uint16)
        assert np.array_equal(bits, wanted)
    for expert in range(q.shape[0]):
        restored_q, restored_s = restore_expert(actual[0][expert].numpy(), actual[1][expert].numpy())
        assert np.array_equal(restored_q, q[expert])
        assert np.array_equal(restored_s, s[expert])


@pytest.mark.parametrize("fault", ["layout", "quantization", "source", "rank", "truncated", "modified", "topology"])
def test_invalid_runtime_files_fail_explicitly(prepared, fault):
    output, shard, _, _, _ = prepared
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    rank = manifest["rank_files"]["pp0-tp0"]
    if fault in ("layout", "quantization"):
        manifest[fault + "_fingerprint"] = "wrong"
    elif fault == "topology":
        manifest["tensor_parallel_size"] = 8
    elif fault == "source":
        manifest["source_manifest_sha256"] = "wrong"
    elif fault == "rank":
        rank["source_sha256"] = "wrong"
    else:
        with (output / rank["file"]).open("r+b") as stream:
            if fault == "truncated":
                stream.truncate(100)
            else:
                stream.seek(-1, 2)
                old = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes([old ^ 1]))
    publish_json(path, manifest)
    with pytest.raises(ValueError):
        N256PreparedShard(output, shard)


def test_live_file_replacement_invalidates_binding(prepared):
    output, shard, _, _, _ = prepared
    loaded = N256PreparedShard(output, shard)
    path = loaded.path
    copy = path.with_suffix(".replacement")
    copy.write_bytes(path.read_bytes())
    copy.replace(path)
    with pytest.raises(RuntimeError, match="changed"):
        loaded.check_identity()
