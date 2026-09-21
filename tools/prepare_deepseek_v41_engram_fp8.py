# SPDX-License-Identifier: Apache-2.0
"""Prepare bounded PP0 Engram FP8 sidecars from immutable rank files."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from prepare_deepseek_v41_woa_fp8 import read_bytes
from vllm_gaudi.ops.deepseek_v41_engram_fp8 import FINGERPRINT, LAYERS, QUANTIZATION, SHAPE
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, file_hash, publish_json
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import prepare_block32_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.prepared / "sidecars" / "engram_fp8"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": 1,
        "quantization": QUANTIZATION,
        "quantization_fingerprint": FINGERPRINT,
        "source_manifest_sha256": file_hash(args.prepared / "manifest.json"),
        "rank_files": {},
    }
    n, k = SHAPE
    for tp in range(2):
        shard = PreparedV41Shard(args.prepared, 0, tp)
        rank = f"pp0-tp{tp}"
        specs = {}
        for layer in LAYERS:
            prefix = f"layers.{layer}.engram.wkv."
            specs[prefix + "weight"] = {"dtype": "U8", "shape": SHAPE}
            specs[prefix + "channel_scale"] = {"dtype": "F32", "shape": [1, n]}
        path = args.output / f"{rank}.safetensors"
        partial = path.with_suffix(".partial")
        writer = RankWriter(partial, specs, {"quantization_fingerprint": FINGERPRINT, "source_rank": rank})
        audit = []
        try:
            for layer in LAYERS:
                prefix = f"layers.{layer}.engram.wkv."
                weight = shard.catalog[prefix + "weight"]
                scale = shard.catalog[prefix + "scale"]
                if weight.shape != SHAPE or weight.dtype != "F8_E4M3" or scale.shape != (n // 32, k // 32):
                    raise ValueError("Engram checkpoint dimensions changed")
                records = []
                for row in range(0, n, 256):
                    count = min(256, n - row)
                    codes = read_bytes(weight, row * k, count * k).reshape(count, k)
                    powers = read_bytes(scale, row // 32 * (k // 32), ((count + 31) // 32) *
                                       (k // 32)).reshape((count + 31) // 32, k // 32)
                    q, channel, record = prepare_block32_rows(codes, powers)
                    if record["temporary_upper_bound_bytes"] > 2 * 2**30:
                        raise RuntimeError("Engram preparation exceeds temporary budget")
                    writer.write(prefix + "weight", row * k, q)
                    writer.write(prefix + "channel_scale", row * 4, channel)
                    record["row_start"] = row
                    records.append(record)
                total = {
                    key: sum(record[key] for record in records)
                    for key in ("values", "changed", "new_zero", "source_energy", "error_energy")
                }
                total.update(layer=layer,
                             blocks=records,
                             temporary_upper_bound_bytes=max(record["temporary_upper_bound_bytes"]
                                                             for record in records))
                total["relative_l2"] = ((total["error_energy"] / total["source_energy"])**.5
                                        if total["source_energy"] else 0)
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
            "temporary_upper_bound_bytes": max(record["temporary_upper_bound_bytes"] for record in audit),
        }
        publish_json(args.output / "progress.json", manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    publish_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
