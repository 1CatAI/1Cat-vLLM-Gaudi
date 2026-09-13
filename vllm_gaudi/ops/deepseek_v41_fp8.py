# SPDX-License-Identifier: Apache-2.0
"""Bounded FP8 channel preparation from the immutable V4.1 Q16/S16 files.

The row-range algorithm comes from the maintained V4 prepared FP8 path.
Integer thresholds replace log2 so all power-of-two choices are exact. This
first fast decoder requires every nonzero group to remain normal in E4M3;
unsupported source ranges fail preparation before publishing a sidecar.
"""

import json
import os
from pathlib import Path

import numpy as np

from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash, file_hash, read_header

QUANTIZATION = {
    "version": 1,
    "source_layout": "q16-v2-k128-s16-group32",
    "weights": "per-output-channel-minimum-covering-power-of-two",
    "format": "gaudi2-e4m3-bias7",
    "maximum": 240,
    "weight_decoder": "normal-results-base-lut",
    "activation": "per-row-max-bf16-scale-epsilon-bf16-reciprocal-rne-canonical-zero-v1",
    "w13_rows": 1,
    "w2_rows": 6,
    "output_tile": 128,
}
FINGERPRINT = canonical_hash(QUANTIZATION)


def precision_config(path):
    config = {"version": 1, "routed_experts": list(range(40)), "attention": [], "shared_experts": [], "mhc": []}
    if path:
        config = json.loads(Path(path).read_text())
    if set(config) != {"version", "routed_experts", "attention", "shared_experts", "mhc"} or config["version"] != 1:
        raise ValueError("Unknown V4.1 FP8 precision configuration")
    if any(config[name] for name in ("attention", "shared_experts", "mhc")):
        raise ValueError("This V4.1 FP8 candidate implements routed experts only")
    layers = config["routed_experts"]
    if not isinstance(layers, list) or any(type(layer) is not int or not 0 <= layer < 40 for layer in layers):
        raise ValueError("Invalid V4.1 FP8 expert layer selection")
    if len(set(layers)) != len(layers):
        raise ValueError("Repeated V4.1 FP8 expert layer selection")
    return dict(config, routed_experts=sorted(layers))


def channel_scales(q16, s16):
    """Return [N/128,128] raw BF16 powers, without unpacking full weights."""
    if (q16.dtype != np.dtype("<i2") or s16.dtype != np.dtype("<u2") or q16.ndim != 2 or s16.ndim != 2
            or q16.shape[0] != s16.shape[0] or q16.shape[1] % 4096 or q16.shape[1] != s16.shape[1] * 8):
        raise ValueError("FP8 channel preparation requires one K128 Q16/S16 expert")
    blocks, stream = q16.shape
    groups = stream // 1024
    codes = (s16.reshape(blocks, groups, 128) >> 7).astype(np.int16)
    if not ((codes >= 2) & (codes <= 254)).all():
        raise ValueError("FP8 preparation requires finite normal E8M0 source encodings")
    raw = q16.view(np.uint8).reshape(blocks, groups, 16, 128)
    maximum = np.maximum((raw & 7).max(2), ((raw >> 4) & 7).max(2))
    # ceil(log2(E2M1[code] / 240)), exactly. Zero groups never set a row range.
    offsets = np.array([-32000, -8, -7, -7, -6, -6, -5, -5], dtype=np.int16)
    exponent = (codes - 127 + offsets[maximum]).max(1)
    nonzero = maximum != 0
    exponent[np.all(~nonzero, axis=1)] = 0
    if not ((exponent >= -126) & (exponent <= 127)).all():
        raise ValueError("FP8 channel scale exceeds normal BF16 power-of-two storage")
    delta = codes - 127 - exponent[:, None, :]
    if ((delta < -5) & nonzero).any():
        raise ValueError("FP8 base-LUT decoder cannot represent this row's subnormal range")
    bits = np.ascontiguousarray(((exponent + 127) << 7).astype("<u2"))
    # Conservative bound includes source storage and all simultaneously live
    # NumPy temporaries, allowing a second packed-byte expression allocation.
    temporary = q16.nbytes * 4 + s16.nbytes * 8 + bits.nbytes * 8
    return bits, {
        "minimum_channel_exponent": int(exponent.min()),
        "maximum_channel_exponent": int(exponent.max()),
        "minimum_nonzero_scale_delta": int(delta[nonzero].min()) if nonzero.any() else None,
        "zero_rows": int(np.all(~nonzero, axis=1).sum()),
        "temporary_upper_bound_bytes": temporary
    }


def read_expert(source, expert):
    """Keep only this bounded read in DRAM; do not populate a full weight mmap."""
    count = source.nbytes // source.shape[0]
    if count > 64 * 2**20 or not 0 <= expert < source.shape[0]:
        raise ValueError("FP8 preparation source read exceeds its bounded expert contract")
    with source.file.open("rb") as stream:
        offset = source.offset + expert * count
        stream.seek(offset)
        raw = bytearray(stream.read(count))
        if len(raw) != count:
            raise ValueError("Truncated prepared FP8 source")
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), offset, count, os.POSIX_FADV_DONTNEED)
    dtype = "<i2" if source.dtype == "I16" else "<u2"
    return np.frombuffer(raw, dtype=dtype).reshape(source.shape[1:])


class FP8Sidecar:

    def __init__(self, directory, shard):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["quantization"] != QUANTIZATION or manifest["quantization_fingerprint"] != FINGERPRINT
                or manifest["source_manifest_sha256"] != file_hash(shard.directory / "manifest.json")):
            raise ValueError("V4.1 FP8 sidecar is bound to different source weights or quantization")
        rank = f"pp{shard.pp_rank}-tp{shard.tp_rank}"
        record = manifest["rank_files"][rank]
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe FP8 sidecar path")
        self.path = directory / relative
        if file_hash(self.path) != record["sha256"]:
            raise ValueError("V4.1 FP8 channel scales changed after qualification")
        if record["source_sha256"] != shard.manifest["rank_files"][rank]["sha256"]:
            raise ValueError("FP8 sidecar belongs to another PP/TP shard")
        self.catalog = read_header(self.path)
        expected = {}
        for name, spec in shard.specs.items():
            if name.startswith("layers.") and name.endswith("_q16"):
                expected[name.removesuffix("_q16") + "_fp8_channel_scale"] = (spec["shape"][0], spec["shape"][1], 128)
        if set(self.catalog) != set(expected) or any(source.dtype != "BF16" or source.shape != expected[name]
                                                     for name, source in self.catalog.items()):
            raise ValueError("V4.1 FP8 channel ownership or dimensions changed")
        self.fingerprint = canonical_hash({"quantization": FINGERPRINT, "rank_sha256": record["sha256"]})

    def tensor(self, name, device):
        import torch
        source = self.catalog[name]
        if source.nbytes > 8 * 2**20:
            raise ValueError("FP8 channel tensor exceeds its bounded transfer")
        with self.path.open("rb") as stream:
            stream.seek(source.offset)
            raw = bytearray(stream.read(source.nbytes))
        if len(raw) != source.nbytes:
            raise ValueError("FP8 sidecar truncated during loading")
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(source.shape).to(device)
