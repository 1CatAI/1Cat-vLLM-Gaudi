# SPDX-License-Identifier: Apache-2.0
"""Immutable channel-scaled Attention projection weights for Gaudi2 MME."""
import json
from pathlib import Path

from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash, file_hash, read_header
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import QUANTIZATION as WOA_QUANTIZATION

SHAPES = {"wq_b": (16384, 1280), "wo_b": (5120, 4096)}
QUANTIZATION = {
    **WOA_QUANTIZATION,
    "layout": "N-K-contiguous",
    "activation_scale": "block32-roundtrip-BF16-then-per-token-power-of-two-f32",
    "projections": SHAPES,
}
FINGERPRINT = canonical_hash(QUANTIZATION)
INPUT_PROJECTIONS = ("wq_a", "wkv", "shared_w1", "shared_w3", "shared_w2")


def projection_prefix(layer, projection):
    if projection.startswith("shared_"):
        return f"layers.{layer}.ffn.shared_experts.{projection.removeprefix('shared_')}."
    return f"layers.{layer}.attn.{projection}."



def shapes_for_tp(tp_size, projections=None):
    if tp_size not in (2, 4):
        raise ValueError("Dense FP8 preparation requires TP2 or TP4")
    shapes = {"wq_b": (32768 // tp_size, 1280), "wo_b": (5120, 8192 // tp_size),
              "wq_a": (1280, 5120), "wkv": (512, 5120),
              "shared_w1": (2304 // tp_size, 5120), "shared_w3": (2304 // tp_size, 5120),
              "shared_w2": (5120, 2304 // tp_size)}
    selected = tuple(SHAPES) if projections is None else tuple(projections)
    if not selected or len(set(selected)) != len(selected) or any(p not in shapes for p in selected):
        raise ValueError("Invalid dense FP8 projections")
    return {p: shapes[p] for p in selected}


def quantization_for_tp(tp_size, projections=None):
    shapes = shapes_for_tp(tp_size, projections)
    result = {**QUANTIZATION, "projections": shapes}
    if any(p in shapes for p in INPUT_PROJECTIONS):
        result["input_activation_scale"] = "per-token-power-of-two-f32-rne"
        result["shared_activation"] = "BF16-projection-clamp-SiLU-BF16-power-of-two-rne"
    return result


def precision_config(path):
    data = {"version": 1, **{p: list(range(40)) for p in SHAPES}} if not path else json.loads(Path(path).read_text())
    version = data.get("version")
    allowed = set(SHAPES) | (set(INPUT_PROJECTIONS) if version == 2 else set())
    if version not in (1, 2) or set(data) != {"version", *allowed}:
        raise ValueError("Invalid dense FP8 precision configuration")
    for projection in allowed:
        layers = data[projection]
        if (not isinstance(layers, list) or any(type(x) is not int or not 0 <= x < 40 for x in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("Dense FP8 layers must be unique backbone layer indices")
        data[projection] = sorted(layers)
    if version == 2:
        for pair in (("wq_a", "wkv"), ("shared_w1", "shared_w3", "shared_w2")):
            if any(data[p] != data[pair[0]] for p in pair):
                raise ValueError("Fused dense FP8 projections must select identical layers")
    return data


class DenseFP8Sidecar:

    def __init__(self, directory, shard):
        directory = Path(directory)
        data = json.loads((directory / "manifest.json").read_text())
        projections = tuple(data["quantization"]["projections"])
        fingerprint = canonical_hash(quantization_for_tp(shard.tensor_parallel_size, projections))
        if (canonical_hash(data["quantization"]) != fingerprint or data["quantization_fingerprint"] != fingerprint
                or data["source_manifest_sha256"] != file_hash(shard.directory / "manifest.json")):
            raise ValueError("Dense FP8 sidecar source/quantization mismatch")
        record = data["rank_files"][f"pp{shard.pp_rank}-tp{shard.tp_rank}"]
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe dense FP8 sidecar path")
        self.path = directory / relative
        if file_hash(self.path) != record["sha256"]:
            raise ValueError("Dense FP8 sidecar hash mismatch")
        rank = shard.manifest["rank_files"][f"pp{shard.pp_rank}-tp{shard.tp_rank}"]
        if record["source_sha256"] != rank["sha256"]:
            raise ValueError("Dense FP8 sidecar rank ownership mismatch")
        self.catalog = read_header(self.path)
        expected = {}
        for layer in range(*shard.manifest["pp_layer_ranges"][shard.pp_rank]):
            for projection, shape in shapes_for_tp(shard.tensor_parallel_size, projections).items():
                prefix = projection_prefix(layer, projection)
                expected[prefix + "weight"] = ("U8", shape)
                expected[prefix + "channel_scale"] = ("F32", (1, shape[0]))
        if set(expected) != set(self.catalog) or any(
            (self.catalog[k].dtype, self.catalog[k].shape) != v for k, v in expected.items()):
            raise ValueError("Dense FP8 sidecar layout/ownership mismatch")
        st = self.path.stat()
        self.identity = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns
        self.fingerprint = canonical_hash({"quantization": fingerprint, "rank_sha256": record["sha256"]})

    def tensor(self, name, device):
        import torch
        source = self.catalog[name]
        if source.nbytes > 20 * 2**20:
            raise ValueError("Dense FP8 tensor exceeds bounded transfer")
        st = self.path.stat()
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != self.identity:
            raise RuntimeError("Dense FP8 sidecar changed; invalidate weights and recipes")
        with self.path.open("rb") as stream:
            stream.seek(source.offset)
            raw = bytearray(stream.read(source.nbytes))
        if len(raw) != source.nbytes:
            raise ValueError("Truncated dense FP8 tensor")
        value = torch.frombuffer(raw, dtype=torch.uint8 if source.dtype == "U8" else torch.float32)
        if source.dtype == "U8":
            # The sidecar holds freshly encoded Gaudi2 values, never source bytes.
            value = value.view(torch.float8_e4m3fn)
        return value.reshape(source.shape).to(device)
