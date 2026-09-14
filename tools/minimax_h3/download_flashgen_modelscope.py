#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and audit the published H3 FlashGen four-step LoRA via ModelScope."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "FlashGen/Minimax-H3-4step-lora-flashgen"
FILENAME = "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
EXPECTED_SHA256 = "0e17fff71d76db497707328a49d61dd7bcd0a375beabef5ae7e93234d15dff00"
EXPECTED_METADATA = {
    "key_format": "minimax-h3-native",
    "qkv_layout": "grouped",
    "lora_rank": "64",
    "lora_alpha": "64",
    "tasks": "t2va",
    "base_schedule": "1.0,0.7,0.4,0.15,0.0",
}


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    mismatches = {
        key: {
            "expected": expected,
            "observed": metadata.get(key)
        }
        for key, expected in EXPECTED_METADATA.items() if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"FlashGen LoRA metadata mismatch: {mismatches}")


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
            allow_patterns=[FILENAME, "README.md", "configuration.json"],
            max_workers=4,
        )).resolve()
    artifact = resolved / FILENAME
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    with safe_open(artifact, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        tensor_count = len(list(checkpoint.keys()))
    _validate_metadata(metadata)
    with artifact.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256 != EXPECTED_SHA256:
        raise ValueError(f"FlashGen LoRA SHA256 mismatch: expected {EXPECTED_SHA256}, observed {sha256}")
    manifest = {
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
            "metadata": metadata,
            "actual_joint_dit_forwards": 4,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path = args.manifest or destination / "modelscope-download-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
