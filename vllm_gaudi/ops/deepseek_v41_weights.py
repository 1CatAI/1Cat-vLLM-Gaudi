# SPDX-License-Identifier: Apache-2.0
"""Bounded, byte-preserving V4.1 checkpoint preparation.

The source namespace and TP axes follow DeepSeek's pinned ``inference/convert.py``
and vLLM PR #56214. Routed experts use intermediate-dimension TP, rather than
the reference converter's expert parallelism. Q16/S16 reuse the V4 Gaudi lane
layout; version 2 admits 128-element K blocks and records any padding.

This module intentionally does not import torch or initialize an accelerator.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

import numpy as np


LAYOUT_VERSION = 2
ROW_BLOCK = 128
K_BLOCK = 128
SCALE_GROUP = 32
MAX_TEMPORARY_BYTES = 2 * 2**30
COPY_BYTES = 16 * 2**20
ITEM_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
              "I16": 2, "U16": 2, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U32": 4,
              "F64": 8, "I64": 8, "U64": 8}
EXPERT = re.compile(r"^((?:layers|mtp)\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.(weight|scale)$")
ENGRAM_TABLE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def publish_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class TensorSource:
    file: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def raw_rows(self, start=0, stop=None) -> np.ndarray:
        """Read complete contiguous rows, preserving FP8 and BF16 bit patterns."""
        rows = self.shape[0] if self.shape else 1
        stop = rows if stop is None else stop
        if not 0 <= start <= stop <= rows:
            raise ValueError("Source row range is outside the tensor")
        row_bytes = math.prod(self.shape[1:]) * ITEM_BYTES[self.dtype]
        count = (stop - start) * row_bytes
        if count > MAX_TEMPORARY_BYTES // 4:
            raise ValueError("A source read exceeds the bounded preparation workspace")
        with self.file.open("rb") as stream:
            stream.seek(self.offset + start * row_bytes)
            data = stream.read(count)
        if len(data) != count:
            raise ValueError(f"Truncated tensor in {self.file.name}")
        return np.frombuffer(data, dtype=np.uint8).reshape(stop - start, row_bytes)


def read_header(path: Path) -> dict[str, TensorSource]:
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError(f"Missing safetensors header: {path}")
        length, = struct.unpack("<Q", size)
        if not 2 <= length <= 100_000_000:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(length))
    result = {}
    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype, shape = entry["dtype"], tuple(entry["shape"])
        if dtype not in ITEM_BYTES or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"Unsupported dtype or shape: {name}")
        begin, end = entry["data_offsets"]
        nbytes = math.prod(shape) * ITEM_BYTES[dtype]
        if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != nbytes:
            raise ValueError(f"Invalid byte span: {name}")
        result[name] = TensorSource(path, dtype, shape, 8 + length + begin, nbytes)
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError(f"Overlapping or missing safetensors data in {path}")
        cursor = end
    if 8 + length + cursor != path.stat().st_size:
        raise ValueError(f"Checkpoint is incomplete or has trailing data: {path}")
    return result


def checkpoint_catalog(model: Path, *, expected_tensors=96085, expected_shards=48, allow_download_headers=False):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    if len(weight_map) != expected_tensors or len(set(weight_map.values())) != expected_shards:
        raise ValueError("Checkpoint tensor/shard count differs from the pinned V4.1 export")
    catalog = {}
    for filename in sorted(set(weight_map.values())):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe filename in checkpoint index")
        path = model / relative
        if path.exists():
            tensors = read_header(path)
        elif allow_download_headers:
            # Only inspect completed header ranges. Tensor data is never read
            # from a sparse download; its final file must be hash-verified first.
            partial = path.with_name(path.name + ".dsv41-download")
            progress = json.loads(partial.with_name(partial.name + ".ranges.json").read_text())
            with partial.open("rb") as stream:
                header_bytes = struct.unpack("<Q", stream.read(8))[0] + 8
            needed = set(range(math.ceil(header_bytes / progress["chunk_bytes"])))
            if not needed.issubset(progress["complete"]):
                raise ValueError(f"Checkpoint header download is incomplete: {relative}")
            tensors = {name: replace(source, file=path) for name, source in read_header(partial).items()}
        else:
            raise FileNotFoundError(path)
        if set(tensors) != {name for name, value in weight_map.items() if value == filename}:
            raise ValueError(f"Tensor names do not match the index: {filename}")
        catalog.update(tensors)
    return catalog


def prepare_q16(packed: np.ndarray, *, original_k=None) -> tuple[np.ndarray, tuple[int, int]]:
    """Bit permutation only; one I16 word packs two rows and two K elements."""
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("Expected raw uint8 MXFP4 [N,K/2]")
    n, k_bytes = packed.shape
    k = k_bytes * 2 if original_k is None else original_k
    if not n or not k or (k + 1) // 2 != k_bytes:
        raise ValueError("Invalid logical MXFP4 shape")
    padded_n = math.ceil(n / ROW_BLOCK) * ROW_BLOCK
    padded_k = math.ceil(k / K_BLOCK) * K_BLOCK
    values = np.zeros((padded_n, padded_k // 2), dtype=np.uint8)
    values[:n, :k_bytes] = packed
    if k % 2:
        values[:n, k_bytes - 1] &= 15
    stream = values.reshape(padded_n // 128, 64, 2, padded_k // 2).transpose(0, 3, 1, 2)
    q16 = np.ascontiguousarray(stream).view("<i2").reshape(padded_n // 128, (padded_k // 2) * 64)
    return q16, (n, k)


def restore_q16(q16: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    n, k = original_shape
    if q16.dtype != np.dtype("<i2") or q16.ndim != 2 or q16.shape[1] % (K_BLOCK * 32):
        raise ValueError("Invalid prepared Q16 layout")
    blocks, stream = q16.shape
    k_bytes = stream // 64
    if not 0 < n <= blocks * ROW_BLOCK or not 0 < k <= k_bytes * 2:
        raise ValueError("Original shape exceeds prepared storage")
    data = q16.view(np.uint8).reshape(blocks, k_bytes, 64, 2).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(data.reshape(blocks * 128, k_bytes)[:n, :(k + 1) // 2])


def prepare_s16(scales: np.ndarray, original_shape: tuple[int, int]) -> tuple[np.ndarray, bool]:
    n, k = original_shape
    if scales.dtype != np.uint8 or scales.shape != (n, math.ceil(k / SCALE_GROUP)):
        raise ValueError("Scale shape does not match group-32 MXFP4")
    padded_n = math.ceil(n / ROW_BLOCK) * ROW_BLOCK
    groups = math.ceil(k / K_BLOCK) * K_BLOCK // SCALE_GROUP
    codes = np.full((padded_n, groups), 127, dtype=np.uint16)
    codes[:n, :scales.shape[1]] = scales
    bits = codes << 7
    output = np.ascontiguousarray(bits.reshape(padded_n // 128, 128, groups).transpose(0, 2, 1))
    return output.reshape(padded_n // 128, groups * 128), bool(np.all((scales >= 2) & (scales <= 254)))


def restore_s16(bits: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    n, k = original_shape
    if bits.dtype != np.uint16 or bits.ndim != 2 or bits.shape[-1] % 128:
        raise ValueError("Invalid prepared S16 layout")
    blocks, stream = bits.shape
    groups = stream // 128
    if n > blocks * 128 or math.ceil(k / 32) > groups:
        raise ValueError("Original shape exceeds prepared scales")
    codes = ((bits.reshape(blocks, groups, 128).transpose(0, 2, 1) >> 7) & 255).astype(np.uint8)
    return np.ascontiguousarray(codes.reshape(blocks * 128, groups)[:n, :math.ceil(k / 32)])


def stage_for(name: str, layers_per_stage=20) -> int:
    if name.startswith("layers."):
        layer = int(name.split(".")[1])
        if not 0 <= layer < 2 * layers_per_stage:
            raise ValueError(f"Layer is outside the TP2/PP2 model: {name}")
        return layer // layers_per_stage
    if name.startswith(("mtp.", "head.", "norm.")):
        return 1
    if name.startswith(("vision.", "aligner.", "image_", "embed.")):
        return 0
    raise ValueError(f"No PP ownership rule for {name}")


def tp_axis(name: str):
    if name.startswith(("vision.", "aligner.", "image_")):
        return None
    if ".shared_experts." in name:
        return 1 if ".w2." in name else 0
    if any(part in name for part in (".wq_b.", ".wo_a.", ".weights_proj.", ".markov_head.embed.",
                                     ".markov_head.head.")) or name in ("embed.weight", "head.weight", "mtp.embed.weight"):
        return 0
    if ".wo_b." in name:
        return 1
    if name.endswith(".attn_sink"):
        return 0
    return None


class RankWriter:
    """Streaming safetensors writer with a predetermined, audited tensor layout."""

    def __init__(self, path: Path, specs: dict, metadata: dict[str, str], *, resume=False):
        self.path = path
        header = {"__metadata__": metadata}
        self.offsets = {}
        self.specs = specs
        position = 0
        for name, spec in sorted(specs.items()):
            count = math.prod(spec["shape"]) * ITEM_BYTES[spec["dtype"]]
            header[name] = {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [position, position + count]}
            self.offsets[name] = (position, count)
            position += count
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        prefix = struct.pack("<Q", len(encoded)) + encoded
        self.data_start = len(prefix)
        self.total_bytes = len(prefix) + position
        if path.exists():
            if not resume:
                raise FileExistsError(path)
            with path.open("rb") as stream:
                if stream.read(len(prefix)) != prefix or path.stat().st_size != self.total_bytes:
                    raise ValueError("Resume file does not match the frozen preparation plan")
            self.stream = path.open("r+b", buffering=0)
        else:
            self.stream = path.open("xb", buffering=0)
            self.stream.write(prefix)
            self.stream.truncate(self.total_bytes)

    def write(self, name: str, relative_offset: int, data):
        data = memoryview(np.ascontiguousarray(data)).cast("B")
        begin, count = self.offsets[name]
        if relative_offset < 0 or relative_offset + len(data) > count:
            raise ValueError(f"Write is outside planned tensor {name}")
        self.stream.seek(self.data_start + begin + relative_offset)
        while data:
            written = self.stream.write(data)
            if not written:
                raise OSError("Unable to write prepared tensor")
            data = data[written:]

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def build_plan(catalog: dict[str, TensorSource], *, layers_per_stage=20):
    ranks = {(pp, tp): {} for pp in range(2) for tp in range(2)}
    groups = {}
    host_tables = {}
    for name, source in sorted(catalog.items()):
        match = ENGRAM_TABLE.fullmatch(name)
        if match:
            host_tables[name] = source
            continue
        match = EXPERT.fullmatch(name)
        if match:
            prefix, expert, projection, kind = match.groups()
            groups.setdefault(prefix, {}).setdefault(int(expert), {})[(projection, kind)] = source
            continue
        pp, axis = stage_for(name, layers_per_stage), tp_axis(name)
        shape = list(source.shape)
        if axis is not None:
            if len(shape) <= axis or shape[axis] % 2:
                raise ValueError(f"TP slice is not even/block aligned: {name}")
            shape[axis] //= 2
        # E8M0 is an encoding, not an integer conversion. Byte storage also
        # permits loading with Bridge versions without a torch E8M0 dtype.
        spec = {"dtype": "U8" if source.dtype == "F8_E8M0" else source.dtype,
                "shape": shape, "source": name, "source_dtype": source.dtype, "tp_axis": axis}
        for tp in range(2):
            ranks[pp, tp][name] = spec.copy()
            if name == "embed.weight":
                # vLLM #53577: DSpark on the last PP rank cannot borrow a
                # PPMissingLayer embedding from PP0. It owns this TP shard.
                ranks[1, tp]["mtp.embed.weight"] = spec.copy()
    for prefix, experts in sorted(groups.items()):
        count = len(experts)
        if set(experts) != set(range(count)):
            raise ValueError(f"Expert IDs are not contiguous: {prefix}")
        expected = {(projection, kind) for projection in ("w1", "w2", "w3") for kind in ("weight", "scale")}
        for expert, matrices in experts.items():
            if set(matrices) != expected:
                raise ValueError(f"Missing expert matrix: {prefix}.{expert}")
        rows, packed_k = experts[0]["w1", "weight"].shape
        hidden, intermediate = packed_k * 2, rows // 2
        if rows % 2 or intermediate % ROW_BLOCK or hidden % K_BLOCK or intermediate % K_BLOCK:
            raise ValueError(f"V4.1 TP intermediate/hidden shape is not block aligned: {prefix}")
        for matrices in experts.values():
            for projection in ("w1", "w3", "w2"):
                n, k = (rows, hidden) if projection != "w2" else (hidden, rows)
                if (matrices[projection, "weight"].shape != (n, k // 2)
                        or matrices[projection, "scale"].shape != (n, k // SCALE_GROUP)
                        or matrices[projection, "weight"].dtype not in ("I8", "U8")
                        or matrices[projection, "scale"].dtype not in ("F8_E8M0", "U8")):
                    raise ValueError(f"Inconsistent MXFP4 expert shape or encoding: {prefix}")
        pp = stage_for(prefix, layers_per_stage)
        for projection, n, k in (("w13", intermediate * 2, hidden), ("w2", hidden, intermediate)):
            for kind, dtype, stream in (("q16", "I16", (k // 2) * 64),
                                         ("s16", "BF16", (k // 32) * 128)):
                name = f"{prefix}.{projection}_{kind}"
                spec = {"dtype": dtype, "shape": [count, n // 128, stream], "original_shape": [count, n, k],
                        "source_pattern": prefix + ".{expert}.{projection}.{kind}", "layout_version": LAYOUT_VERSION}
                for tp in range(2):
                    ranks[pp, tp][name] = spec.copy()
    return ranks, groups, host_tables


def copy_plain(source: TensorSource, writer: RankWriter, name: str, tp: int):
    axis = writer.specs[name]["tp_axis"]
    rows = source.shape[0] if source.shape else 1
    row_bytes = source.nbytes // rows
    chunk_rows = max(1, COPY_BYTES // row_bytes)
    start, stop = (tp * (rows // 2), (tp + 1) * (rows // 2)) if axis == 0 else (0, rows)
    target_offset = 0
    for begin in range(start, stop, chunk_rows):
        data = source.raw_rows(begin, min(stop, begin + chunk_rows))
        if axis == 1:
            data = data[:, tp * (row_bytes // 2):(tp + 1) * (row_bytes // 2)]
        writer.write(name, target_offset, data)
        target_offset += data.size
    if target_offset != writer.offsets[name][1]:
        raise ValueError(f"Plain tensor copy did not cover the destination: {name}")


def copy_experts(prefix: str, experts: dict, writers: list[RankWriter]):
    """At most one expert matrix plus two small conversions is live at once."""
    normal = [True, True]
    for expert, matrices in sorted(experts.items()):
        for projection in ("w1", "w3", "w2"):
            source, scale = matrices[projection, "weight"], matrices[projection, "scale"]
            packed, codes = source.raw_rows(), scale.raw_rows()
            for tp, writer in enumerate(writers):
                if projection != "w2":
                    n = packed.shape[0] // 2
                    raw, scales = packed[tp * n:(tp + 1) * n], codes[tp * n:(tp + 1) * n]
                    destination = "w13"
                else:
                    columns, groups = packed.shape[1] // 2, codes.shape[1] // 2
                    raw = packed[:, tp * columns:(tp + 1) * columns]
                    scales = codes[:, tp * groups:(tp + 1) * groups]
                    destination = "w2"
                q16, logical = prepare_q16(raw)
                s16, eligible = prepare_s16(scales, logical)
                normal[tp] &= eligible
                for kind, data in (("q16", q16), ("s16", s16)):
                    name = f"{prefix}.{destination}_{kind}"
                    stride = writer.offsets[name][1] // len(experts)
                    offset = expert * stride + (stride // 2 if projection == "w3" else 0)
                    writer.write(name, offset, data)
    return normal


def host_manifest(tables: dict[str, TensorSource], tp: int, revision: str, source_hashes: dict, *, hash_layout=None):
    records = {}
    for name, source in sorted(tables.items()):
        rows = source.shape[0]
        if hash_layout is None:
            # Useful for synthetic/legacy manifests; production V4.1 uses the
            # complete-head sharding of vLLM #56214 below.
            shard_rows = math.ceil(rows / 2)
            begin, end = tp * shard_rows, min(rows, (tp + 1) * shard_rows)
            head = {}
        else:
            head = hash_layout.head_shard(int(ENGRAM_TABLE.fullmatch(name).group(1)), tp)
            begin, end = head["row_start"], head["row_stop"]
            shard_rows = end - begin
            if not 0 <= begin <= end <= rows:
                raise ValueError("Engram head shard is outside the original table")
        row_bytes = source.nbytes // rows
        records[name] = {"file": str(source.file.resolve()), "source_sha256": source_hashes[source.file.name],
                         "dtype": source.dtype, "shape": list(source.shape), "tensor_offset": source.offset,
                         "row_start": begin, "row_stop": end, "padded_rows": shard_rows,
                         "shard_offset": source.offset + begin * row_bytes, "row_bytes": row_bytes,
                         "shard_bytes": (end - begin) * row_bytes, **head}
    return {"format_version": 1, "model_revision": revision, "tp_rank": tp, "pp_owner": 0,
            "shared_read_only": True, "storage": "original_safetensors_mmap", "tables": records,
            "sharding": "complete_hash_heads" if hash_layout is not None else "ceil_rows"}
