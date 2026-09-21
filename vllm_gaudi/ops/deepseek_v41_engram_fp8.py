# SPDX-License-Identifier: Apache-2.0
"""Immutable channel-scaled Engram projection weights for Gaudi2 MME."""
import json
import math
import os
from pathlib import Path

from vllm_gaudi.ops.deepseek_v41_weights import ITEM_BYTES, canonical_hash, file_hash, read_header
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import QUANTIZATION as WOA_QUANTIZATION

LAYERS = (1, 14)
SHAPE = (25600, 6144)
COPY_BYTES = 16 * 2**20
QUANTIZATION = {
    **WOA_QUANTIZATION,
    "layout": "N-K-contiguous",
    "activation_scale": "block32-roundtrip-BF16-then-per-token-power-of-two-f32",
    "projection": "engram.wkv",
    "layers": LAYERS,
    "shape": SHAPE,
}
FINGERPRINT = canonical_hash(QUANTIZATION)


class EngramFP8Sidecar:
    """Load PP0-only Engram weights with a bounded host staging allocation."""

    def __init__(self, directory, shard):
        if shard.pp_rank != 0:
            raise ValueError("Engram FP8 sidecars belong only to PP0")
        directory = Path(directory)
        data = json.loads((directory / "manifest.json").read_text())
        if (canonical_hash(data["quantization"]) != FINGERPRINT
                or data["quantization_fingerprint"] != FINGERPRINT
                or data["source_manifest_sha256"] != file_hash(shard.directory / "manifest.json")):
            raise ValueError("Engram FP8 sidecar source/quantization mismatch")
        rank_name = f"pp0-tp{shard.tp_rank}"
        record = data["rank_files"][rank_name]
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe Engram FP8 sidecar path")
        self.path = directory / relative
        if file_hash(self.path) != record["sha256"]:
            raise ValueError("Engram FP8 sidecar hash mismatch")
        source_rank = shard.manifest["rank_files"][rank_name]
        if record["source_sha256"] != source_rank["sha256"]:
            raise ValueError("Engram FP8 sidecar rank ownership mismatch")
        self.catalog = read_header(self.path)
        expected = {}
        for layer in LAYERS:
            prefix = f"layers.{layer}.engram.wkv."
            expected[prefix + "weight"] = ("U8", SHAPE)
            expected[prefix + "channel_scale"] = ("F32", (1, SHAPE[0]))
        if set(expected) != set(self.catalog) or any(
                (self.catalog[name].dtype, self.catalog[name].shape) != spec for name, spec in expected.items()):
            raise ValueError("Engram FP8 sidecar layout/ownership mismatch")
        stat = self.path.stat()
        self.identity = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        self.fingerprint = canonical_hash({"quantization": FINGERPRINT, "rank_sha256": record["sha256"]})
        self.max_host_chunk_bytes = 0

    def _check_identity(self):
        stat = self.path.stat()
        if self.identity != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Engram FP8 sidecar changed; invalidate weights and recipes")

    def tensor(self, name, device):
        import torch
        self._check_identity()
        source = self.catalog[name]
        dtype = torch.uint8 if source.dtype == "U8" else torch.float32
        item_bytes = ITEM_BYTES[source.dtype]
        destination_dtype = torch.float8_e4m3fn if source.dtype == "U8" else dtype
        destination = torch.empty(source.shape, dtype=destination_dtype, device=device)
        row_bytes = math.prod(source.shape[1:]) * item_bytes
        rows = source.shape[0] if source.shape else 1
        chunk_rows = max(1, COPY_BYTES // row_bytes)
        with self.path.open("rb") as stream:
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                offset = source.offset + start * row_bytes
                stream.seek(offset)
                storage = bytearray((stop - start) * row_bytes)
                if stream.readinto(storage) != len(storage):
                    raise ValueError("Engram FP8 sidecar was truncated during loading")
                shape = (stop - start, *source.shape[1:]) if source.shape else ()
                value = torch.frombuffer(storage, dtype=dtype).reshape(shape)
                if source.dtype == "U8":
                    value = value.view(torch.float8_e4m3fn)
                if source.shape:
                    destination[start:stop].copy_(value, non_blocking=False)
                else:
                    destination.copy_(value, non_blocking=False)
                self.max_host_chunk_bytes = max(self.max_host_chunk_bytes, len(storage))
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(stream.fileno(), offset, len(storage), os.POSIX_FADV_DONTNEED)
        self._check_identity()
        return destination
