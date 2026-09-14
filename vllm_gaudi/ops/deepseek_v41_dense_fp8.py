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


def precision_config(path):
    data = {"version": 1, **{p: list(range(40)) for p in SHAPES}} if not path else json.loads(Path(path).read_text())
    if set(data) != {"version", *SHAPES} or data["version"] != 1:
        raise ValueError("Invalid dense FP8 precision configuration")
    for projection in SHAPES:
        layers = data[projection]
        if (not isinstance(layers, list) or any(type(x) is not int or not 0 <= x < 40 for x in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("Dense FP8 layers must be unique backbone layer indices")
        data[projection] = sorted(layers)
    return data


class DenseFP8Sidecar:

    def __init__(self, directory, shard):
        directory = Path(directory)
        data = json.loads((directory / "manifest.json").read_text())
        if (canonical_hash(data["quantization"]) != FINGERPRINT or data["quantization_fingerprint"] != FINGERPRINT
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
        for layer in range(shard.pp_rank * 20, shard.pp_rank * 20 + 20):
            for projection, shape in SHAPES.items():
                prefix = f"layers.{layer}.attn.{projection}."
                expected[prefix + "weight"] = ("U8", shape)
                expected[prefix + "channel_scale"] = ("F32", (1, shape[0]))
        if set(expected) != set(self.catalog) or any(
                (self.catalog[k].dtype, self.catalog[k].shape) != v for k, v in expected.items()):
            raise ValueError("Dense FP8 sidecar layout/ownership mismatch")
        st = self.path.stat()
        self.identity = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns
        self.fingerprint = canonical_hash({"quantization": FINGERPRINT, "rank_sha256": record["sha256"]})

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
