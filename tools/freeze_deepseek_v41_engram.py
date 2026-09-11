# SPDX-License-Identifier: Apache-2.0
"""Freeze Engram token normalization against the verified checkpoint tokenizer."""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, build_compressed_token_map
from vllm_gaudi.ops.deepseek_v41_weights import file_hash, publish_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    args = parser.parse_args()
    names = ("config.json", "tokenizer.json", "tokenizer_config.json")
    hashes = {}
    revision = None
    for name in names:
        record = json.loads((args.checkpoint_audit / "files" / (name + ".json")).read_text())
        hashes[name] = file_hash(args.model / name)
        if hashes[name] != record["sha256"]:
            raise ValueError(f"Tokenizer source changed after checkpoint audit: {name}")
        if revision is not None and revision != record["revision"]:
            raise ValueError("Mixed checkpoint revisions in tokenizer audit")
        revision = record["revision"]
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    token_map, compressed_size = build_compressed_token_map(tokenizer)
    if len(token_map) != config["vocab_size"] or compressed_size != config["engram_compressed_vocab_size"]:
        raise ValueError(f"Engram vocabulary mismatch: raw={len(token_map)}, compressed={compressed_size}")
    layout = EngramHashLayout.from_config(config)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "compressed-token-map.npy", np.asarray(token_map, dtype=np.int32), allow_pickle=False)
    np.savez(args.output / "hash-layout.npz", primes=layout.primes, offsets=layout.offsets,
             multipliers=layout.multipliers)
    record = {"model_revision": revision, "source_sha256": hashes, "vocab_size": len(token_map),
              "compressed_vocab_size": compressed_size, "layer_ids": list(layout.layer_ids),
              "token_map_sha256": file_hash(args.output / "compressed-token-map.npy"),
              "layout_sha256": file_hash(args.output / "hash-layout.npz"),
              "tp_head_shards": {str(layer): [layout.head_shard(layer, tp) for tp in range(2)]
                                 for layer in layout.layer_ids}}
    publish_json(args.output / "manifest.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
