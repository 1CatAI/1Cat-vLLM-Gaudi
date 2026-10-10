# SPDX-License-Identifier: Apache-2.0
"""Direct rank-local loading with bounded host buffers and manifest validation."""

import json
import math
import os
from pathlib import Path
import struct
import time
from concurrent.futures import ThreadPoolExecutor

from vllm_gaudi.ops.deepseek_v41_weights import (
    COPY_BYTES,
    ITEM_BYTES,
    LAYOUT_VERSION,
    canonical_hash,
    file_hash,
    read_header,
)


class PreparedV41Shard:

    def __init__(self, directory, pp_rank: int, tp_rank: int, *, verify_hash=False):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.plan = json.loads((self.directory / "preparation-plan.json").read_text())
        manifest = self.manifest
        tp_size = manifest.get("tensor_parallel_size")
        pp_size = manifest.get("pipeline_parallel_size")
        expected_ranges = [[0, 40]] if pp_size == 1 else [[0, 20], [20, 40]] if pp_size == 2 else None
        if ((tp_size, pp_size) not in ((2, 2), (4, 1)) or not 0 <= pp_rank < pp_size or not 0 <= tp_rank < tp_size
                or manifest["pp_layer_ranges"] != expected_ranges
                or manifest["prepared_layout_version"] != LAYOUT_VERSION):
            raise ValueError("Prepared V4.1 topology or layout version mismatch (expected TP4 x PP1; TP2 x PP2 legacy)")
        if canonical_hash(self.plan) != manifest["plan_fingerprint"]:
            raise ValueError("Prepared plan differs from the published manifest")
        for key in ("model_revision", "quantization_fingerprint", "upstream_lock_sha256", "tensor_parallel_size",
                    "pipeline_parallel_size", "pp_layer_ranges"):
            if self.plan[key] != manifest[key]:
                raise ValueError(f"Prepared manifest {key} mismatch")
        for name, expected in manifest["metadata_sha256"].items():
            if file_hash(self._local_path(name)) != expected:
                raise ValueError(f"Prepared metadata changed: {name}")
        for name, expected in manifest["encoding_sha256"].items():
            if file_hash(self._local_path(name)) != expected:
                raise ValueError(f"Frozen model encoding changed: {name}")
        rank = f"pp{pp_rank}-tp{tp_rank}"
        self.specs = self.plan["ranks"][rank]
        record = manifest["rank_files"][rank]
        self.path = self._local_path(record["file"])
        stat = self.path.stat()
        if stat.st_size != record["bytes"]:
            raise ValueError("Prepared rank file size differs from the manifest")
        unchanged = (stat.st_ino == record.get("inode") and stat.st_mtime_ns == record.get("mtime_ns"))
        if (verify_hash or not unchanged) and file_hash(self.path) != record["sha256"]:
            raise ValueError("Prepared rank file hash differs from the manifest")
        self.source_identity = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        self.catalog = read_header(self.path)
        with self.path.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        expected_metadata = {
            "plan_fingerprint": manifest["plan_fingerprint"],
            "model_revision": manifest["model_revision"],
            "prepared_layout_version": str(LAYOUT_VERSION),
            "pp_rank": str(pp_rank),
            "tp_rank": str(tp_rank)
        }
        if any(header.get("__metadata__", {}).get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("Rank-local header is bound to another preparation plan")
        if set(self.catalog) != set(self.specs):
            raise ValueError("Prepared rank tensor names differ from the plan")
        for name, source in self.catalog.items():
            spec = self.specs[name]
            if source.dtype != spec["dtype"] or source.shape != tuple(spec["shape"]):
                raise ValueError(f"Prepared tensor shape or dtype mismatch: {name}")
        self.pp_rank, self.tp_rank = pp_rank, tp_rank
        self.tensor_parallel_size, self.pipeline_parallel_size = tp_size, pp_size
        self.max_host_chunk_bytes = 0
        self.startup_timings = dict(read_seconds=0.,
                                    upload_seconds=0.,
                                    conversion_enqueue_seconds=0.,
                                    device_finish_seconds=0.,
                                    derived_seconds=0.,
                                    source_bytes=0,
                                    upload_bytes=0)
        self.startup_families = {}

    def timed_specs(self, specs):
        """Family wall times are exclusive; reader/queue counters may overlap."""
        for name, spec in specs.items():
            family = ("experts" if ".ffn.experts." in name else
                      "draft" if name.startswith("mtp.") else "dense_fp8" if spec["dtype"] == "F8_E4M3" else "direct")
            started = time.perf_counter()
            try:
                yield name, spec
            finally:
                self.startup_families[family] = (self.startup_families.get(family, 0.) + time.perf_counter() - started)

    def _local_path(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe path in prepared checkpoint manifest")
        return self.directory / relative

    def check_identity(self):
        stat = self.path.stat()
        if self.source_identity != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Prepared weight file changed during loading; invalidate the model and recipes")

    def staged_rows(self, source, row_bytes, rows, chunk_rows):
        """Read the next bounded chunk while the caller consumes this one."""
        timings = getattr(self, "startup_timings", None)
        with self.path.open("rb") as stream:

            def read(start):
                self.check_identity()
                stop = min(rows, start + chunk_rows)
                storage = bytearray((stop - start) * row_bytes)
                stream.seek(source.offset + start * row_bytes)
                started = time.perf_counter()
                if stream.readinto(storage) != len(storage):
                    raise ValueError("Prepared rank file was truncated during loading")
                if timings is not None:
                    timings["read_seconds"] += time.perf_counter() - started
                    timings["source_bytes"] += len(storage)
                return start, stop, storage

            if rows <= chunk_rows:
                yield read(0)
                return
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(read, 0)
                for start in range(0, rows, chunk_rows):
                    current = future.result()
                    if start + chunk_rows < rows:
                        future = pool.submit(read, start + chunk_rows)
                    yield current
                    del current

    def tensor(self, name: str, device, *, keep_file_cache=True):
        """Copy one pre-sliced tensor; no expert gather, transpose or repacking."""
        import torch
        timings = getattr(self, "startup_timings", None)
        self.check_identity()
        source = self.catalog[name]
        dtype_names = {
            "BF16": "bfloat16",
            "F16": "float16",
            "F32": "float32",
            "F64": "float64",
            "I8": "int8",
            "U8": "uint8",
            "I16": "int16",
            "I32": "int32",
            "I64": "int64",
            "BOOL": "bool",
            "F8_E4M3": "float8_e4m3fn",
            "F8_E5M2": "float8_e5m2"
        }
        if source.dtype not in dtype_names:
            raise ValueError(f"Prepared tensor has an unsupported runtime dtype: {source.dtype}")
        dtype = getattr(torch, dtype_names[source.dtype])
        destination = torch.empty(source.shape, dtype=dtype, device=device)
        row_bytes = math.prod(source.shape[1:]) * ITEM_BYTES[source.dtype]
        chunk_rows = max(1, COPY_BYTES // row_bytes)
        rows = source.shape[0] if source.shape else 1
        for start, stop, storage in self.staged_rows(source, row_bytes, rows, chunk_rows):
            # bytearray is writable; frombuffer keeps the owner alive until
            # the synchronous H2D copy has consumed it. The destination is the
            # single persistent prepared tensor, not a temporary weight copy.
            offset = source.offset + start * row_bytes
            chunk_shape = (stop - start, *source.shape[1:]) if source.shape else ()
            chunk = torch.frombuffer(storage, dtype=dtype).reshape(chunk_shape)
            started = time.perf_counter()
            if source.shape:
                destination[start:stop].copy_(chunk, non_blocking=False)
            else:
                destination.copy_(chunk, non_blocking=False)
            if timings is not None:
                timings["upload_seconds"] += time.perf_counter() - started
                timings["upload_bytes"] += len(storage)
            self.max_host_chunk_bytes = max(self.max_host_chunk_bytes, len(storage))
            # The four rank loaders read distinct 75–78 GiB files in
            # parallel.  Synchronous DONTNEED for every copied chunk can
            # block all workers in generic_fadvise/lru_add_drain_all;
            # normally leave eviction to the VM.  The explicit opt-out
            # remains for an isolated/offline loader.
            if not keep_file_cache and hasattr(os, "posix_fadvise"):
                with self.path.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), offset, len(storage), os.POSIX_FADV_DONTNEED)
            del chunk, storage
        self.check_identity()
        return destination

    def bind(self, model, mapping: dict[str, str], device):
        """Assign fully prepared tensors to registered parameters/buffers once."""
        import torch
        if set(mapping) != set(self.catalog):
            raise ValueError("The model must account for every tensor in its rank-local file")
        destinations = set(mapping.values())
        if len(destinations) != len(mapping):
            raise ValueError("Multiple prepared tensors cannot silently replace the same model parameter")
        for source_name, target_name in mapping.items():
            module_name, _, attribute = target_name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            if attribute not in module._parameters and attribute not in module._buffers:
                raise ValueError(f"Prepared destination is not registered in the model: {target_name}")
            value = self.tensor(source_name, device)
            if attribute in module._parameters:
                setattr(module, attribute, torch.nn.Parameter(value, requires_grad=False))
            else:
                setattr(module, attribute, value)
        self.check_identity()

    def dense(self, name: str, device):
        """Load a block-32 FP8 matrix into the existing BF16 dense MME path.

        Checkpoint E4M3FN bytes are decoded on CPU, where all 448-range codes
        are supported. Only a bounded row block exists in host memory. The
        caller retains this one device matrix and no duplicate FP8 cache.
        """
        import torch
        self.check_identity()
        source = self.catalog[name]
        scale_source = self.catalog[name.removesuffix("weight") + "scale"]
        if source.dtype != "F8_E4M3" or len(source.shape) != 2 or scale_source.dtype != "U8":
            raise ValueError("Dense preparation requires E4M3FN weights and raw UE8M0 scales")
        n, k = source.shape
        if scale_source.shape != ((n + 31) // 32, (k + 31) // 32):
            raise ValueError("Dense FP8 scale geometry is not block 32 x 32")
        destination = torch.empty((n, k), device=device, dtype=torch.bfloat16)
        chunk_rows = max(32, (COPY_BYTES // (k * 8) // 32) * 32)
        with self.path.open("rb") as stream:
            for start in range(0, n, chunk_rows):
                stop = min(n, start + chunk_rows)
                stream.seek(source.offset + start * k)
                raw = bytearray(stream.read((stop - start) * k))
                if len(raw) != (stop - start) * k:
                    raise ValueError("Dense FP8 weight source was truncated")
                stream.seek(scale_source.offset + start // 32 * scale_source.shape[1])
                scale_rows = (stop + 31) // 32 - start // 32
                raw_scale = bytearray(stream.read(scale_rows * scale_source.shape[1]))
                if len(raw_scale) != scale_rows * scale_source.shape[1]:
                    raise ValueError("Dense FP8 scale source was truncated")
                value = torch.frombuffer(raw, dtype=torch.float8_e4m3fn).reshape(stop - start, k).float()
                codes = torch.frombuffer(raw_scale, dtype=torch.uint8).reshape(scale_rows, -1)
                scales = torch.exp2(codes.float() - 127)
                scales.masked_fill_(codes == 255, float("nan"))
                expanded = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)[:stop - start, :k]
                value.mul_(expanded)
                destination[start:stop].copy_(value.to(torch.bfloat16))
                temporary = len(raw) + len(raw_scale) + value.numel() * 4 + expanded.numel() * 4
                self.max_host_chunk_bytes = max(self.max_host_chunk_bytes, temporary)
                # Do not issue per-chunk DONTNEED while TP2×PP2 workers load
                # concurrently; natural reclaim protects the resident Engram
                # pages without a cross-CPU LRU drain at every weight chunk.
        self.check_identity()
        return destination
