# SPDX-License-Identifier: Apache-2.0
"""Gaudi2 wo_a channel preparation; checkpoint bytes are never reinterpreted."""

import json
from pathlib import Path

import numpy as np

from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash, file_hash, read_header

QUANTIZATION = {
    "version": 1,
    "source": "e4m3fn-ue8m0-block32",
    "format": "gaudi2-e4m3-bias7",
    "maximum": 240,
    "rounding": "nearest-even-then-flush-fp8-subnormal-canonical-zero",
    "weight_scale": "per-output-channel-minimum-covering-power-of-two-f32",
    "activation_scale": "per-token-group-minimum-covering-power-of-two-f32",
    "layout": "G4-K4096-N1024",
    "epilogue": "f32-product-times-weight-scale-times-activation-scale-to-bf16",
}
FINGERPRINT = canonical_hash(QUANTIZATION)


def layer_selection(path):
    data = {"version": 1, "layers": list(range(40))} if not path else json.loads(Path(path).read_text())
    if set(data) != {"version", "layers"} or data["version"] != 1 or not isinstance(data["layers"], list):
        raise ValueError("Invalid wo_a precision configuration")
    layers = data["layers"]
    if any(type(x) is not int or not 0 <= x < 40 for x in layers) or len(set(layers)) != len(layers):
        raise ValueError("wo_a layers must be unique backbone layer indices")
    return {"version": 1, "layers": sorted(layers)}


def decode_e4m3fn(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    exponent = ((codes >> 3) & 15).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    if np.any((exponent == 15) & (mantissa == 7)):
        raise ValueError("Nonfinite wo_a checkpoint encoding")
    value = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    return np.where(codes & 128, -value, value).astype(np.float32)


def covering_scale(values):
    """Exact power-of-two threshold choice, without rounded log2 decisions."""
    values = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Scale maxima must be finite and nonnegative")
    fraction, exponent = np.frexp(values)
    power = exponent - 8 + (fraction > np.float32(240 / 256))
    power = np.where(values == 0, 0, power)
    if np.any(power < -126) or np.any(power > 127):
        raise ValueError("wo_a scale must be a normal FP32 power of two")
    return np.ldexp(np.ones_like(values), power).astype(np.float32)


def encode_gaudi2(values):
    """RNE to bias-7 E4M3, then flush exponent-zero results explicitly."""
    values = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(np.abs(values) > 240):
        raise ValueError("FP8 encoding input exceeds finite Gaudi2 range")
    absolute = np.abs(values)
    _, exponent = np.frexp(absolute)
    step = np.ldexp(np.ones_like(absolute), np.maximum(exponent - 4, -9))
    rounded = np.rint(absolute / step) * step
    fraction, exponent = np.frexp(rounded)
    mantissa = np.rint((fraction * 2 - 1) * 8).astype(np.int32)
    code = ((exponent + 6) << 3) + mantissa
    code = np.where(rounded < np.float32(2**-6), 0, code)
    code = np.where((values < 0) & (code != 0), code | 128, code)
    return code.astype(np.uint8)


def decode_gaudi2(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any((codes & 0x78) == 0x78) or np.any(((codes & 0x78) == 0) & ((codes & 0x7f) != 0)):
        raise ValueError("Expected normal/zero finite Gaudi2 FP8 bytes")
    return decode_e4m3fn(codes)


def prepare_rows(codes, scales):
    """Prepare aligned N rows; source and expanded temporaries stay bounded."""
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != 4096 or codes.shape[0] % 32:
        raise ValueError("wo_a row blocks require uint8 [multiple-of-32,4096]")
    return prepare_block32_rows(codes, scales)


def prepare_block32_rows(codes, scales):
    """Channel-requantize a bounded, aligned block without discarding K scales."""
    if (codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[0] % 32 or codes.shape[1] % 32
            or not codes.size):
        raise ValueError("Channel preparation requires nonempty block32-aligned uint8 rows")
    if (scales.dtype != np.uint8 or scales.shape != (codes.shape[0] // 32, codes.shape[1] // 32)
            or np.any(scales == 255)):
        raise ValueError("Channel preparation requires finite UE8M0 block32 scales")
    powers = np.repeat(np.repeat(scales.astype(np.int32) - 127, 32, 0), 32, 1)
    values = np.ldexp(decode_e4m3fn(codes), powers)
    channel = covering_scale(np.max(np.abs(values), axis=1, keepdims=True))
    packed = encode_gaudi2(values / channel)
    restored = decode_gaudi2(packed) * channel
    error = restored.astype(np.float64) - values
    record = {
        "values": int(values.size),
        "changed": int(np.count_nonzero(restored != values)),
        "new_zero": int(np.count_nonzero((restored == 0) & (values != 0))),
        "source_energy": float(np.sum(values.astype(np.float64)**2)),
        "error_energy": float(np.sum(error**2)),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "minimum_scale": float(channel.min()),
        "maximum_scale": float(channel.max()),
        "temporary_upper_bound_bytes": int(codes.nbytes * 128),
    }
    return packed, channel, record


class WoaFP8Sidecar:

    def __init__(self, directory, shard):
        directory = Path(directory)
        data = json.loads((directory / "manifest.json").read_text())
        if (data["quantization"] != QUANTIZATION or data["quantization_fingerprint"] != FINGERPRINT
                or data["source_manifest_sha256"] != file_hash(shard.directory / "manifest.json")):
            raise ValueError("wo_a sidecar source/quantization mismatch")
        record = data["rank_files"][f"pp{shard.pp_rank}-tp{shard.tp_rank}"]
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe wo_a sidecar path")
        self.path = directory / relative
        if file_hash(self.path) != record["sha256"]:
            raise ValueError("wo_a sidecar hash mismatch")
        rank = shard.manifest["rank_files"][f"pp{shard.pp_rank}-tp{shard.tp_rank}"]
        if record["source_sha256"] != rank["sha256"]:
            raise ValueError("wo_a sidecar rank ownership mismatch")
        self.catalog = read_header(self.path)
        expected = {}
        for layer in range(shard.pp_rank * 20, shard.pp_rank * 20 + 20):
            expected[f"layers.{layer}.attn.wo_a.weight"] = ("U8", (4, 4096, 1024))
            expected[f"layers.{layer}.attn.wo_a.channel_scale"] = ("F32", (4, 1, 1024))
        if set(expected) != set(self.catalog) or any(
            (self.catalog[k].dtype, self.catalog[k].shape) != v for k, v in expected.items()):
            raise ValueError("wo_a sidecar layout/ownership mismatch")
        st = self.path.stat()
        self.identity = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns
        self.fingerprint = canonical_hash({"quantization": FINGERPRINT, "rank_sha256": record["sha256"]})

    def tensor(self, name, device):
        import torch
        source = self.catalog[name]
        if source.nbytes > 16 * 2**20:
            raise ValueError("wo_a tensor exceeds bounded transfer")
        st = self.path.stat()
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != self.identity:
            raise RuntimeError("wo_a sidecar changed; invalidate weights and recipes")
        with self.path.open("rb") as stream:
            stream.seek(source.offset)
            raw = bytearray(stream.read(source.nbytes))
        if len(raw) != source.nbytes:
            raise ValueError("Truncated wo_a tensor")
        value = torch.frombuffer(raw, dtype=torch.uint8 if source.dtype == "U8" else torch.float32)
        # These bytes were explicitly re-encoded, not copied from the source FP8 format.
        if source.dtype == "U8":
            value = value.view(torch.float8_e4m3fn)
        return value.reshape(source.shape).to(device)
