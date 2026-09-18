#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the official BF16 MiniMax H3 base used by FastH3 on HPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

BASE_MODEL_ID = "MiniMax/MiniMax-H3"
BASE_PATTERNS = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "configuration.json",
    "FL2VA/model_index.json",
    "FL2VA/transformer/**",
    "FL2VA/text_encoder/**",
)
REUSED_COMPONENTS = ("video_vae", "audio_vae", "tokenizer", "processor")


def _partition(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path if path.name == "FL2VA" else path / "FL2VA"


def _validate_bf16_tensor_headers(component: Path, shards: list[str]) -> dict[str, int]:
    from safetensors import safe_open

    dtypes: dict[str, int] = {}
    for name in shards:
        with safe_open(component / name, framework="pt", device="cpu") as checkpoint:
            keys = checkpoint.keys()
            for key in keys:
                dtype = checkpoint.get_slice(key).get_dtype()
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
    unsupported = sorted(set(dtypes) - {"BF16", "F32"})
    if unsupported or not dtypes.get("BF16"):
        raise ValueError(f"{component} is not an official BF16 component; tensor dtypes={dtypes}")
    return dtypes


def _validate_bf16_transformer(partition: Path) -> dict[str, object]:
    component = partition / "transformer"
    config_path = component / "config.json"
    index_path = component / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("quantization_config") is not None:
        raise ValueError(f"FastH3 base transformer must be BF16, got {config.get('quantization_config')}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if len(shards) != 13:
        raise ValueError(f"official FL2VA transformer must contain 13 shards, got {len(shards)}")
    missing = [name for name in shards if not (component / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing official BF16 transformer shards: {missing}")
    dtypes = _validate_bf16_tensor_headers(component, shards)
    return {
        "format": "bf16",
        "shards": len(shards),
        "bytes": sum((component / name).stat().st_size for name in shards),
        "tensor_dtypes": dtypes,
    }


def _validate_bf16_text_encoder(partition: Path) -> dict[str, object]:
    component = partition / "text_encoder"
    config_path = component / "config.json"
    index_path = component / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("quantization_config") is not None:
        raise ValueError(f"FastH3 text encoder must be BF16, got {config.get('quantization_config')}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if len(shards) != 14:
        raise ValueError(f"official FL2VA text encoder must contain 14 shards, got {len(shards)}")
    missing = [name for name in shards if not (component / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing official BF16 text-encoder shards: {missing}")
    dtypes = _validate_bf16_tensor_headers(component, shards)
    return {
        "format": "bf16",
        "shards": len(shards),
        "bytes": sum((component / name).stat().st_size for name in shards),
        "tensor_dtypes": dtypes,
    }


def _validate_reuse_source(partition: Path) -> dict[str, object]:
    for component in REUSED_COMPONENTS:
        if not (partition / component).is_dir():
            raise FileNotFoundError(partition / component)
    return {
        "video_vae": "original_precision",
        "audio_vae": "original_precision",
        "tokenizer": "reused",
        "processor": "reused",
    }


def _link_reused_components(destination: Path, source: Path) -> None:
    for component in REUSED_COMPONENTS:
        target = destination / component
        expected = source / component
        if target.is_symlink():
            if target.resolve() != expected.resolve():
                raise ValueError(f"{target} points to {target.resolve()}, expected {expected.resolve()}")
            continue
        if target.exists():
            raise FileExistsError(f"refusing to replace existing component {target}")
        target.symlink_to(Path(os.path.relpath(expected, target.parent)), target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True, help="destination for the assembled ModelScope model")
    parser.add_argument(
        "--reuse-components-from",
        type=Path,
        required=True,
        help="existing H3 root or FL2VA partition supplying unchanged VAE/tokenizer/processor files",
    )
    parser.add_argument("--revision", default="master")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    destination = args.local_dir.expanduser().resolve()
    reuse_partition = _partition(args.reuse_components_from)
    reused = _validate_reuse_source(reuse_partition)

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    from modelscope import snapshot_download

    resolved = Path(
        snapshot_download(
            model_id=BASE_MODEL_ID,
            revision=args.revision,
            local_dir=str(destination),
            allow_patterns=list(BASE_PATTERNS),
            max_workers=args.max_workers,
        )).resolve()
    destination_partition = resolved / "FL2VA"
    transformer = _validate_bf16_transformer(destination_partition)
    text_encoder = _validate_bf16_text_encoder(destination_partition)
    _link_reused_components(destination_partition, reuse_partition)
    if not (destination_partition / "model_index.json").is_file():
        raise FileNotFoundError(destination_partition / "model_index.json")

    result = {
        "schema_version": 1,
        "status": "pass",
        "download_client": "modelscope.snapshot_download",
        "huggingface_network_disabled": True,
        "base_model_id": BASE_MODEL_ID,
        "requested_revision": args.revision,
        "resolved_path": str(resolved),
        "allow_patterns": list(BASE_PATTERNS),
        "partition": "FL2VA",
        "transformer": transformer,
        "text_encoder": text_encoder,
        "reused_components_from": str(reuse_partition),
        "reused_components": reused,
        "fusion_order": ["bf16_base", "fasth3_dense_datafree_delta"],
        "optional_runtime_quantization": "hpu_online_fp8_ptpc",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    manifest_path = args.manifest or destination / "fasth3-base-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
