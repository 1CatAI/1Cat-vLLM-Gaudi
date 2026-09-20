# SPDX-License-Identifier: Apache-2.0
"""Immutable runtime-layout expert shards for repeated model loading."""
import json
import math
from pathlib import Path

from vllm_gaudi.ops.deepseek_v41_expert_n256 import FINGERPRINT, LAYOUT
from vllm_gaudi.ops.deepseek_v41_fp8 import FINGERPRINT as QUANTIZATION_FINGERPRINT
from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash, file_hash, read_header


def runtime_specs(shard):
    specs = {}
    for name, source in shard.catalog.items():
        if not name.startswith("layers.") or ".ffn.experts." not in name or not name.endswith("_q16"):
            continue
        prefix = name.removesuffix("_q16")
        experts, blocks, stream = source.shape
        scale = shard.catalog[prefix + "_s16"]
        if (blocks % 2 or stream % 4096 or source.dtype != "I16" or scale.dtype != "BF16"
                or scale.shape != (experts, blocks, stream // 8)):
            raise ValueError("Runtime N256 shard requires the original K128 expert layout")
        specs[name] = {"dtype": "I16", "shape": [experts, blocks // 2, stream * 2]}
        specs[prefix + "_s16"] = {"dtype": "I16", "shape": [experts, blocks // 2, stream // 4]}
        specs[prefix + "_fp8_channel"] = {"dtype": "BF16", "shape": [experts, blocks // 2, 256]}
    if not specs:
        raise ValueError("No mainline experts in the prepared source shard")
    return specs


class N256PreparedShard:
    def __init__(self, directory, shard):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("layout") != LAYOUT
                or manifest.get("tensor_parallel_size") != 2 or manifest.get("pipeline_parallel_size") != 2
                or manifest.get("layout_fingerprint") != FINGERPRINT
                or manifest.get("quantization_fingerprint") != QUANTIZATION_FINGERPRINT
                or manifest.get("source_manifest_sha256") != file_hash(shard.directory / "manifest.json")):
            raise ValueError("Prepared N256 source/layout/quantization fingerprint mismatch")
        rank = f"pp{shard.pp_rank}-tp{shard.tp_rank}"
        record = manifest["rank_files"][rank]
        if record["source_sha256"] != shard.manifest["rank_files"][rank]["sha256"]:
            raise ValueError("Prepared N256 file belongs to another rank/source")
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe prepared N256 path")
        self.path = directory / relative
        stat = self.path.stat()
        if stat.st_size != record["bytes"]:
            raise ValueError("Prepared N256 file is truncated")
        # Same immutable-file policy as PreparedV41Shard: a changed identity
        # requires a full digest check; unchanged rank files avoid a second
        # full disk scan on every restart.
        if ((stat.st_ino != record["inode"] or stat.st_mtime_ns != record["mtime_ns"])
                and file_hash(self.path) != record["sha256"]):
            raise ValueError("Prepared N256 file hash mismatch")
        self.identity = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        self.catalog = read_header(self.path)
        specs = runtime_specs(shard)
        if set(specs) != set(self.catalog) or any(
                src.dtype != specs[name]["dtype"] or src.shape != tuple(specs[name]["shape"])
                for name, src in self.catalog.items()):
            raise ValueError("Prepared N256 tensor shape/dtype/ownership mismatch")
        self.fingerprint = canonical_hash({"layout": FINGERPRINT, "rank_sha256": record["sha256"]})

    def check_identity(self):
        stat = self.path.stat()
        if self.identity != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Prepared N256 weights changed during loading")

    def projection(self, prefix, device):
        import torch
        result = []
        self.check_identity()
        for suffix in ("_q16", "_s16", "_fp8_channel"):
            source = self.catalog[prefix + suffix]
            dtype = torch.int16 if source.dtype == "I16" else torch.bfloat16
            destination = torch.empty(source.shape, dtype=dtype, device=device)
            row_bytes = math.prod(source.shape[1:]) * 2
            chunk = max(1, (128 << 20) // row_bytes)
            with self.path.open("rb") as stream:
                for first in range(0, source.shape[0], chunk):
                    stop = min(first + chunk, source.shape[0])
                    stream.seek(source.offset + first * row_bytes)
                    data = bytearray((stop - first) * row_bytes)
                    if stream.readinto(data) != len(data):
                        raise RuntimeError("Prepared N256 weight read was truncated")
                    value = torch.frombuffer(data, dtype=dtype).reshape(stop - first, *source.shape[1:])
                    destination[first:stop].copy_(value, non_blocking=False)
            result.append(destination)
        self.check_identity()
        return tuple(result)
