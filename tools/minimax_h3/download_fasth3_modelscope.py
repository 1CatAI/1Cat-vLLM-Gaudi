#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and audit the FastH3 Dense-DataFree adapter via ModelScope."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA"
RELATIVE_PATH = Path("dense-datafree/adapter_model.safetensors")
EXPECTED_BYTES = 1485626152
EXPECTED_SHA256 = "4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12"
EXPECTED_TENSORS = 809
EXPECTED_METADATA = {
    "format": "fastvideo-lora-v2",
    "base_model": "MiniMaxAI/MiniMax-H3",
    "finetuned_model": "FastVideo/FastVideo-FastH3-Dense-4-step-v1",
    "rank": "64",
    "low_rank_tensors": "724",
    "diff_tensors": "85",
    "set_weight_tensors": "0",
}
BASE_SCHEDULE = (0.999, 0.749, 0.5, 0.25, 0.0)


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    mismatches = {
        key: {
            "expected": expected,
            "observed": metadata.get(key)
        }
        for key, expected in EXPECTED_METADATA.items() if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"FastH3 adapter metadata mismatch: {mismatches}")


def _validate_bundle_manifest(manifest: Mapping[str, object]) -> None:
    sampling = manifest.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("transformer_forwards") != 4:
        raise ValueError("FastH3 bundle must declare four transformer forwards")
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError("FastH3 bundle has no variants list")
    dense = [item for item in variants if isinstance(item, Mapping) and item.get("slug") == "dense-datafree"]
    if len(dense) != 1:
        raise ValueError("FastH3 bundle must contain exactly one dense-datafree variant")
    variant = dense[0]
    expected = {
        "adapter_path": str(RELATIVE_PATH),
        "adapter_size_bytes": EXPECTED_BYTES,
        "adapter_sha256": EXPECTED_SHA256,
        "requires_vsa": False,
    }
    mismatches = {key: (variant.get(key), value) for key, value in expected.items() if variant.get(key) != value}
    if mismatches:
        raise ValueError(f"FastH3 dense-datafree bundle mismatch: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    destination = args.local_dir.expanduser().resolve()

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    from modelscope import snapshot_download
    from safetensors import safe_open

    resolved = Path(
        snapshot_download(
            model_id=MODEL_ID,
            revision=args.revision,
            local_dir=str(destination),
            allow_patterns=[str(RELATIVE_PATH), "adapter_manifest.json", "README.md", "configuration.json"],
            max_workers=4,
        )).resolve()
    artifact = resolved / RELATIVE_PATH
    bundle_path = resolved / "adapter_manifest.json"
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if artifact.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"FastH3 adapter size mismatch: {artifact.stat().st_size} != {EXPECTED_BYTES}")
    with safe_open(artifact, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = checkpoint.keys()
        tensor_count = len(keys)
        tensor_dtypes: dict[str, int] = {}
        for key in keys:
            dtype = checkpoint.get_slice(key).get_dtype()
            tensor_dtypes[dtype] = tensor_dtypes.get(dtype, 0) + 1
    _validate_metadata(metadata)
    if tensor_count != EXPECTED_TENSORS:
        raise ValueError(f"FastH3 adapter tensor count mismatch: {tensor_count} != {EXPECTED_TENSORS}")
    if tensor_dtypes != {"BF16": EXPECTED_TENSORS}:
        raise ValueError(f"FastH3 adapter must contain only BF16 tensors, got {tensor_dtypes}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    _validate_bundle_manifest(bundle)
    with artifact.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256 != EXPECTED_SHA256:
        raise ValueError(f"FastH3 adapter SHA256 mismatch: {sha256} != {EXPECTED_SHA256}")

    result = {
        "schema_version": 1,
        "status": "pass",
        "download_client": "modelscope.snapshot_download",
        "huggingface_network_disabled": True,
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_path": str(resolved),
        "artifact": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": sha256,
            "tensor_count": tensor_count,
            "tensor_dtypes": tensor_dtypes,
            "metadata": metadata,
            "base_schedule": list(BASE_SCHEDULE),
            "actual_joint_dit_forwards": len(BASE_SCHEDULE) - 1,
            "attention": "dense",
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    manifest_path = args.manifest or destination / "modelscope-download-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
