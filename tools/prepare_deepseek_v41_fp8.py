# SPDX-License-Identifier: Apache-2.0
"""Prepare small FP8 channel sidecars without duplicating compressed weights."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from vllm_gaudi.ops.deepseek_v41_fp8 import FINGERPRINT, QUANTIZATION, channel_scales, read_expert
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_weights import RankWriter, file_hash, publish_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "quantization": QUANTIZATION,
        "quantization_fingerprint": FINGERPRINT,
        "source_manifest_sha256": file_hash(args.prepared / "manifest.json"),
        "rank_files": {}
    }
    for pp in range(2):
        for tp in range(2):
            shard = PreparedV41Shard(args.prepared, pp, tp)
            rank = f"pp{pp}-tp{tp}"
            specs = {
                name.removesuffix("_q16") + "_fp8_channel_scale": {
                    "dtype": "BF16",
                    "shape": [spec["shape"][0], spec["shape"][1], 128]
                }
                for name, spec in shard.specs.items() if name.startswith("layers.") and name.endswith("_q16")
            }
            path = args.output / f"{rank}.safetensors"
            writer = RankWriter(path.with_suffix(".partial"), specs, {
                "quantization_fingerprint": FINGERPRINT,
                "source_rank": rank
            })
            audit, peak = [], 0
            try:
                for name in sorted(specs):
                    prefix = name.removesuffix("_fp8_channel_scale")
                    q, s = (shard.catalog[prefix + suffix] for suffix in ("_q16", "_s16"))
                    for expert in range(q.shape[0]):
                        bits, record = channel_scales(read_expert(q, expert), read_expert(s, expert))
                        peak = max(peak, record["temporary_upper_bound_bytes"])
                        if peak > 2 * 2**30:
                            raise RuntimeError("FP8 preparation temporary bound exceeded")
                        writer.write(name, expert * bits.nbytes, bits)
                        audit.append({"tensor": name, "expert": expert, **record})
                    shard.check_identity()
                    print(f"{rank}: {name} qualified", flush=True)
                writer.sync()
            finally:
                writer.close()
                publish_json(args.output / f"{rank}-ranges.json", audit)
            path.with_suffix(".partial").replace(path)
            manifest["rank_files"][rank] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "source_sha256": shard.manifest["rank_files"][rank]["sha256"],
                "temporary_upper_bound_bytes": peak,
                "qualified_expert_matrices": len(audit)
            }
            publish_json(args.output / "progress.json", manifest)
    manifest["prepared_at"] = datetime.now(timezone.utc).isoformat()
    publish_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
