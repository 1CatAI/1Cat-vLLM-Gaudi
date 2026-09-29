# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; no Bridge, device allocation or model download required."""

import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vllm_gaudi.ops.deepseek_v41_weights import (
    RankWriter, build_plan, copy_experts, copy_plain, host_manifest, prepare_q16, prepare_s16,
    read_header, restore_q16, restore_s16, stage_for, tp_axis,
)


def write_source(path, tensors):
    header, chunks, position = {}, [], 0
    for name, (dtype, value) in tensors.items():
        data = value.tobytes()
        header[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [position, position + len(data)]}
        position += len(data)
        chunks.append(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))
    return read_header(path)


@pytest.mark.parametrize("n,k", [(128, 128), (1152, 5120), (5120, 1152), (130, 130), (129, 129)])
def test_q16_exact_roundtrip_and_padding(n, k):
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (n, (k + 1) // 2), dtype=np.uint8)
    if k % 2:
        packed[:, -1] &= 15
    prepared, logical = prepare_q16(packed, original_k=k)
    assert np.array_equal(restore_q16(prepared, logical), packed)
    padded = restore_q16(prepared, (prepared.shape[0] * 128, prepared.shape[1] // 32))
    assert not padded[n:].any()
    assert not padded[:n, (k + 1) // 2:].any()


def test_q16_lane_contract_independent_of_roundtrip():
    packed = np.arange(128 * 64, dtype=np.uint16).reshape(128, 64).astype(np.uint8)
    q16, _ = prepare_q16(packed)
    words = q16.view(np.uint16).reshape(1, 64, 64)
    for row in range(0, 128, 2):
        assert np.array_equal(words[0, :, row // 2], packed[row].astype(np.uint16) | (packed[row + 1].astype(np.uint16) << 8))


def test_every_scale_encoding_is_reversible():
    codes = np.arange(256, dtype=np.uint8).reshape(128, 2)
    bits, normal = prepare_s16(codes, (128, 64))
    assert not normal
    assert np.array_equal(restore_s16(bits, (128, 64)), codes)
    assert np.all(bits.reshape(1, 4, 128)[:, 2:] == 127 << 7)
    codes[:] = 2
    assert prepare_s16(codes, (128, 64))[1]
    codes[:] = 254
    assert prepare_s16(codes, (128, 64))[1]


def test_tp_rules_preserve_pp_and_draft_embedding():
    assert stage_for("layers.19.ffn.gate.weight") == 0
    assert stage_for("layers.20.ffn.gate.weight") == 1
    assert stage_for("mtp.0.main_proj.weight") == 1
    assert stage_for("vision.blocks.1.norm1.weight") == 0
    assert tp_axis("layers.2.attn.indexer.wq_b.scale") == 0
    assert tp_axis("layers.2.attn.wo_b.scale") == 1
    assert tp_axis("layers.2.attn.wkv.scale") is None
    assert tp_axis("layers.2.ffn.shared_experts.w2.scale") == 1
    with pytest.raises(ValueError):
        stage_for("layers.40.attn.wq_a.weight")


@pytest.mark.parametrize("tp_size,pp_size,intermediate", [(2, 2, 256), (4, 1, 256), (4, 1, 2304)])
def test_streamed_tp_experts_reconstruct_all_source_bytes(tmp_path, tp_size, pp_size, intermediate):
    rng = np.random.default_rng(0)
    tensors = {}
    prefix = "layers.0.ffn.experts"
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            n, k = (256, intermediate) if projection == "w2" else (intermediate, 256)
            for kind, columns in (("weight", k // 2), ("scale", k // 32)):
                tensors[f"{prefix}.{expert}.{projection}.{kind}"] = (
                    "I8" if kind == "weight" else "F8_E8M0",
                    rng.integers(0, 256, (n, columns), dtype=np.uint8))
    tensors["embed.weight"] = ("BF16", np.arange(64, dtype=np.uint16).reshape(8, 8))
    tensors["layers.0.attn.wo_b.scale"] = ("F8_E8M0", np.arange(32, dtype=np.uint8).reshape(4, 8))
    catalog = write_source(tmp_path / "source.safetensors", tensors)
    plans, groups, tables = build_plan(catalog, tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    assert not tables
    assert len(plans) == 4
    assert plans[pp_size - 1, 0]["mtp.embed.weight"]["source"] == "embed.weight"
    writers = [RankWriter(tmp_path / f"tp{tp}.safetensors", plans[0, tp], {}) for tp in range(tp_size)]
    try:
        copy_experts(prefix, groups[prefix], writers, tp_size)
        for tp, writer in enumerate(writers):
            for name, spec in writer.specs.items():
                if "source" in spec:
                    copy_plain(catalog[spec["source"]], writer, name, tp, tp_size)
            writer.sync()
    finally:
        for writer in writers:
            writer.close()
    local = intermediate // tp_size
    padded = (local + 127) // 128 * 128
    for tp in range(tp_size):
        result = read_header(tmp_path / f"tp{tp}.safetensors")
        for expert in range(2):
            for projection in ("w1", "w3", "w2"):
                prepared_projection = "w2" if projection == "w2" else "w13"
                logical = (256, padded) if projection == "w2" else (padded * 2, 256)
                for kind, prepared_kind, restore, dtype in (("weight", "q16", restore_q16, np.int16),
                                                           ("scale", "s16", restore_s16, np.uint16)):
                    name = f"{prefix}.{prepared_projection}_{prepared_kind}"
                    spec = result[name]
                    value = spec.raw_rows(expert, expert + 1).view(dtype).reshape(spec.shape[1:])
                    restored = restore(value, logical)
                    expected = tensors[f"{prefix}.{expert}.{projection}.{kind}"][1]
                    if projection == "w2":
                        width = expected.shape[1] // tp_size
                        expected = expected[:, tp * width:(tp + 1) * width]
                        padding = restored[:, width:]
                        assert np.all(padding == (0 if kind == "weight" else 127))
                        restored = restored[:, :width]
                    else:
                        restored = restored[:padded] if projection == "w1" else restored[padded:]
                        assert np.all(restored[local:] == (0 if kind == "weight" else 127))
                        restored = restored[:local]
                        expected = expected[tp * local:(tp + 1) * local]
                    assert np.array_equal(restored, expected)
        assert result["layers.0.attn.wo_b.scale"].dtype == "U8"
        assert np.array_equal(result["layers.0.attn.wo_b.scale"].raw_rows(),
                              tensors["layers.0.attn.wo_b.scale"][1][:, tp * (8 // tp_size):(tp + 1) * (8 // tp_size)])


def test_tp4_single_stage_owns_all_layers_and_output(tmp_path):
    names = ("embed.weight", "head.weight", "norm.weight", "layers.0.attn.wq_a.weight",
             "layers.20.attn.wq_a.weight", "layers.39.attn.wq_a.weight", "mtp.0.main_proj.weight")
    catalog = write_source(tmp_path / "source.safetensors",
                           {name: ("BF16", np.zeros((8, 8), dtype=np.uint16)) for name in names})
    plans, _, _ = build_plan(catalog, tensor_parallel_size=4, pipeline_parallel_size=1)
    assert set(plans) == {(0, rank) for rank in range(4)}
    for specs in plans.values():
        assert set(names).issubset(specs)
        assert specs["embed.weight"]["shape"] == [2, 8]
        assert specs["head.weight"]["shape"] == [2, 8]
    with pytest.raises(ValueError, match="Layer is outside"):
        stage_for("layers.40.attn.wq_a.weight", pipeline_parallel_size=1)


def test_host_shards_reference_disjoint_source_rows(tmp_path):
    tensors = {"layers.1.engram.embed.weight": ("F8_E4M3", np.arange(15, dtype=np.uint8).reshape(5, 3)),
               "layers.1.engram.embed.scale": ("F8_E8M0", np.arange(5, dtype=np.uint8).reshape(5, 1))}
    path = tmp_path / "source.safetensors"
    catalog = write_source(path, tensors)
    manifests = [host_manifest(catalog, tp, "revision", {path.name: "sha256"}) for tp in range(2)]
    for name, source in catalog.items():
        first, second = [manifest["tables"][name] for manifest in manifests]
        assert first["row_start"] == 0 and first["row_stop"] == second["row_start"] == 3
        assert second["row_stop"] == 5 and second["padded_rows"] == 3
        assert first["shard_offset"] + first["shard_bytes"] == second["shard_offset"]
        assert second["shard_offset"] + second["shard_bytes"] == source.offset + source.nbytes


def test_truncation_and_overwrite_are_rejected(tmp_path):
    path = tmp_path / "source.safetensors"
    write_source(path, {"norm.weight": ("BF16", np.arange(8, dtype=np.uint16))})
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match="incomplete"):
        read_header(path)
    spec = {"x": {"dtype": "U8", "shape": [4]}}
    destination = tmp_path / "prepared.safetensors"
    writer = RankWriter(destination, spec, {"version": "2"})
    with pytest.raises(ValueError, match="outside"):
        writer.write("x", 1, np.zeros(4, dtype=np.uint8))
    writer.close()
    with pytest.raises(FileExistsError):
        RankWriter(destination, spec, {})
    with pytest.raises(ValueError, match="frozen"):
        RankWriter(destination, spec, {"version": "1"}, resume=True)
