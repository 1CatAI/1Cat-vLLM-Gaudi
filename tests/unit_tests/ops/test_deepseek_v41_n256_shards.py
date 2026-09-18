# SPDX-License-Identifier: Apache-2.0
"""Prepared runtime files must preserve weights and reject stale bindings."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.prepare_deepseek_v41_n256 import prepare_rank
from vllm_gaudi.ops.deepseek_v41_expert_n256 import FINGERPRINT, LAYOUT, load_projection, prepare_expert
from vllm_gaudi.ops.deepseek_v41_fp8 import FINGERPRINT as QUANTIZATION_FINGERPRINT
from vllm_gaudi.ops.deepseek_v41_n256_shards import N256PreparedShard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, file_hash, publish_json, read_header


@pytest.fixture
def prepared(tmp_path):
    prefix = "layers.0.ffn.experts.w13"
    rng = np.random.default_rng(31)
    q = rng.integers(-32768, 32768, (3, 4, 4096), dtype=np.int16)
    s = np.full((3, 4, 512), 127 << 7, dtype=np.uint16)
    base = tmp_path / "base"
    base.mkdir()
    source = base / "pp0-tp0.safetensors"
    arrays = {prefix + "_q16": q, prefix + "_s16": s}
    specs = {
        name: {
            "dtype": "I16" if name.endswith("_q16") else "BF16",
            "shape": list(value.shape)
        }
        for name, value in arrays.items()
    }
    writer = RankWriter(source, specs, {})
    try:
        for name, value in arrays.items():
            writer.write(name, 0, value)
        writer.sync()
    finally:
        writer.close()
    manifest = {"rank_files": {"pp0-tp0": {"sha256": file_hash(source)}}}
    publish_json(base / "manifest.json", manifest)
    shard = SimpleNamespace(directory=base,
                            pp_rank=0,
                            tp_rank=0,
                            manifest=manifest,
                            catalog=read_header(source),
                            check_identity=lambda: None)
    output = tmp_path / "runtime"
    output.mkdir()
    rank, record = prepare_rank(shard, output)
    publish_json(
        output / "manifest.json", {
            "schema_version": 1,
            "layout": LAYOUT,
            "layout_fingerprint": FINGERPRINT,
            "quantization_fingerprint": QUANTIZATION_FINGERPRINT,
            "source_manifest_sha256": file_hash(base / "manifest.json"),
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "rank_files": {
                rank: record
            }
        })
    return output, shard, prefix, q, s


def test_runtime_loader_is_exact_and_skips_preparation(prepared, monkeypatch):
    output, shard, prefix, q, s = prepared
    expected = [prepare_expert(a, b)[:3] for a, b in zip(q, s)]
    monkeypatch.setenv("VLLM_HPU_DSV41_N256_PREPARED_DIR", str(output))

    def forbidden(*args):
        raise AssertionError("runtime must not recompute prepared weights")

    monkeypatch.setattr("vllm_gaudi.ops.deepseek_v41_expert_n256.prepare_expert", forbidden)
    actual = load_projection(shard, prefix, "cpu")
    for index, tensor in enumerate(actual):
        bits = tensor.view(torch.int16).numpy().view(np.uint16)
        wanted = np.stack([row[index] for row in expected]).view(np.uint16)
        assert np.array_equal(bits, wanted)


@pytest.mark.parametrize("fault", ["layout", "quantization", "source", "rank", "truncated", "modified"])
def test_invalid_runtime_files_fail_explicitly(prepared, fault):
    output, shard, _, _, _ = prepared
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    rank = manifest["rank_files"]["pp0-tp0"]
    if fault in ("layout", "quantization"):
        manifest[fault + "_fingerprint"] = "wrong"
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
