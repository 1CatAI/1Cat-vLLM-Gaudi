# SPDX-License-Identifier: Apache-2.0
"""Prepare all wo_a FP8 sidecars directly from immutable rank-local files."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, file_hash, publish_json
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import FINGERPRINT, QUANTIZATION, prepare_rows


def read_bytes(source, offset, size):
    with source.file.open("rb") as stream:
        stream.seek(source.offset + offset)
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("Truncated wo_a source")
    return np.frombuffer(raw, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.prepared / "sidecars" / "wo_a_fp8"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": 1,
        "quantization": QUANTIZATION,
        "quantization_fingerprint": FINGERPRINT,
        "source_manifest_sha256": file_hash(args.prepared / "manifest.json"),
        "rank_files": {}
    }
    for pp in range(2):
        for tp in range(2):
            shard = PreparedV41Shard(args.prepared, pp, tp)
            rank = f"pp{pp}-tp{tp}"
            specs = {}
            for layer in range(pp * 20, pp * 20 + 20):
                prefix = f"layers.{layer}.attn.wo_a."
                specs[prefix + "weight"] = {"dtype": "U8", "shape": [4, 4096, 1024]}
                specs[prefix + "channel_scale"] = {"dtype": "F32", "shape": [4, 1, 1024]}
            path = args.output / f"{rank}.safetensors"
            partial = path.with_suffix(".partial")
            writer = RankWriter(partial, specs, {"quantization_fingerprint": FINGERPRINT, "source_rank": rank})
            audit = []
            try:
                for layer in range(pp * 20, pp * 20 + 20):
                    prefix = f"layers.{layer}.attn.wo_a."
                    weight, scale = shard.catalog[prefix + "weight"], shard.catalog[prefix + "scale"]
                    if weight.shape != (4096, 4096) or weight.dtype != "F8_E4M3" or scale.shape != (128, 128):
                        raise ValueError("wo_a checkpoint dimensions changed")
                    packed = np.empty((4, 4096, 1024), dtype=np.uint8)
                    channels = np.empty((4, 1, 1024), dtype=np.float32)
                    records = []
                    for row in range(0, 4096, 256):
                        codes = read_bytes(weight, row * 4096, 256 * 4096).reshape(256, 4096)
                        powers = read_bytes(scale, row // 32 * 128, 8 * 128).reshape(8, 128)
                        q, s, record = prepare_rows(codes, powers)
                        group, begin = divmod(row, 1024)
                        packed[group, :, begin:begin + 256] = q.T
                        channels[group, 0, begin:begin + 256] = s[:, 0]
                        record["row_start"] = row
                        records.append(record)
                    temporary = max(r["temporary_upper_bound_bytes"] for r in records) + packed.nbytes + channels.nbytes
                    if temporary > 2 * 2**30:
                        raise RuntimeError("wo_a preparation exceeds temporary budget")
                    writer.write(prefix + "weight", 0, packed)
                    writer.write(prefix + "channel_scale", 0, channels)
                    total = {
                        key: sum(r[key] for r in records)
                        for key in ("values", "changed", "new_zero", "source_energy", "error_energy")
                    }
                    total.update(layer=layer, temporary_upper_bound_bytes=temporary, blocks=records)
                    total["relative_l2"] = (total["error_energy"] /
                                            total["source_energy"])**0.5 if total["source_energy"] else 0
                    audit.append(total)
                    shard.check_identity()
                    print(f"{rank} layer={layer} relative_l2={total['relative_l2']:.9g}", flush=True)
                writer.sync()
            finally:
                writer.close()
                publish_json(args.output / f"{rank}-audit.json", audit)
            partial.replace(path)
            manifest["rank_files"][rank] = {
                "file": path.name,
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "source_sha256": shard.manifest["rank_files"][rank]["sha256"],
                "temporary_upper_bound_bytes": max(r["temporary_upper_bound_bytes"] for r in audit),
            }
            publish_json(args.output / "progress.json", manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    publish_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
