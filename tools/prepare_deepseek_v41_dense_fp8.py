# SPDX-License-Identifier: Apache-2.0
"""Prepare bounded Attention dense FP8 sidecars from immutable rank files."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from prepare_deepseek_v41_woa_fp8 import read_bytes
from vllm_gaudi.ops.deepseek_v41_dense_fp8 import FINGERPRINT, QUANTIZATION, SHAPES
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, file_hash, publish_json
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import prepare_block32_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.prepared / "sidecars" / "attention_dense_fp8"
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
            rank, specs = f"pp{pp}-tp{tp}", {}
            for layer in range(pp * 20, pp * 20 + 20):
                for projection, shape in SHAPES.items():
                    prefix = f"layers.{layer}.attn.{projection}."
                    specs[prefix + "weight"] = {"dtype": "U8", "shape": shape}
                    specs[prefix + "channel_scale"] = {"dtype": "F32", "shape": [1, shape[0]]}
            path = args.output / f"{rank}.safetensors"
            partial = path.with_suffix(".partial")
            writer = RankWriter(partial, specs, {"quantization_fingerprint": FINGERPRINT, "source_rank": rank})
            audit = []
            try:
                for layer in range(pp * 20, pp * 20 + 20):
                    for projection, (n, k) in SHAPES.items():
                        prefix = f"layers.{layer}.attn.{projection}."
                        weight, scale = shard.catalog[prefix + "weight"], shard.catalog[prefix + "scale"]
                        if weight.shape != (n, k) or weight.dtype != "F8_E4M3" or scale.shape != (n // 32, k // 32):
                            raise ValueError("Dense checkpoint dimensions changed")
                        records = []
                        for row in range(0, n, 256):
                            codes = read_bytes(weight, row * k, 256 * k).reshape(256, k)
                            powers = read_bytes(scale, row // 32 * (k // 32), 8 * (k // 32)).reshape(8, k // 32)
                            q, s, record = prepare_block32_rows(codes, powers)
                            if record["temporary_upper_bound_bytes"] > 2 * 2**30:
                                raise RuntimeError("Dense preparation exceeds temporary budget")
                            writer.write(prefix + "weight", row * k, q)
                            writer.write(prefix + "channel_scale", row * 4, s)
                            record["row_start"] = row
                            records.append(record)
                        total = {
                            key: sum(r[key] for r in records)
                            for key in ("values", "changed", "new_zero", "source_energy", "error_energy")
                        }
                        total.update(layer=layer,
                                     projection=projection,
                                     blocks=records,
                                     temporary_upper_bound_bytes=max(r["temporary_upper_bound_bytes"] for r in records))
                        total["relative_l2"] = ((total["error_energy"] /
                                                 total["source_energy"])**.5 if total["source_energy"] else 0)
                        audit.append(total)
                        shard.check_identity()
                        print(f"{rank} layer={layer} {projection} relative_l2={total['relative_l2']:.9g}", flush=True)
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
                "temporary_upper_bound_bytes": max(r["temporary_upper_bound_bytes"] for r in audit)
            }
            publish_json(args.output / "progress.json", manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    publish_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
