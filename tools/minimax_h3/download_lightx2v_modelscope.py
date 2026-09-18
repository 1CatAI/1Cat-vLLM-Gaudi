#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and audit the qualified LightX2V H3 Turbo LoRA via ModelScope."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "lightx2v/Minimax-h3-Turbo"
EXPECTED_TENSORS = 624
EXPECTED_SHAPES = {
    (128, 5376): 208,
    (128, 7168): 52,
    (128, 14336): 52,
    (5376, 128): 104,
    (7168, 128): 156,
    (28672, 128): 52,
}
SIGMA_POINTS = 5
DENOISER_FORWARDS = 4
RANK = 128


@dataclass(frozen=True)
class ArtifactProfile:
    filename: str
    expected_bytes: int
    expected_sha256: str
    expected_metadata: Mapping[str, str | None]
    task_family: str
    resolution: str
    video_flow_shift: float
    audio_flow_shift: float
    alpha: float


PROFILES = {
    "fl2v-4step-768p":
    ArtifactProfile(
        filename="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
        expected_bytes=1383677808,
        expected_sha256="1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c",
        expected_metadata={
            "format": "pt",
            "floating_dtype": "bfloat16",
            "alpha": "128",
            "key_format": "minimax-h3-diffusers",
        },
        task_family="t2va_fl2va",
        resolution="1344x768",
        video_flow_shift=6.0,
        audio_flow_shift=3.0,
        alpha=128.0,
    ),
    "ref2v-4step-544p":
    ArtifactProfile(
        filename="minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
        expected_bytes=1383677768,
        expected_sha256="9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
        expected_metadata={
            "format": "pt",
            "floating_dtype": "bfloat16",
            "alpha": "8",
            "key_format": None,
        },
        task_family="ref2va",
        resolution="544p_mixed_aspect_ratio",
        video_flow_shift=12.0,
        audio_flow_shift=3.0,
        alpha=8.0,
    ),
}
DEFAULT_PROFILE = "fl2v-4step-768p"

# Keep the original constants importable for callers pinned to the qualified
# FL2VA profile.
FILENAME = PROFILES[DEFAULT_PROFILE].filename
EXPECTED_BYTES = PROFILES[DEFAULT_PROFILE].expected_bytes
EXPECTED_SHA256 = PROFILES[DEFAULT_PROFILE].expected_sha256
EXPECTED_METADATA = PROFILES[DEFAULT_PROFILE].expected_metadata
VIDEO_FLOW_SHIFT = PROFILES[DEFAULT_PROFILE].video_flow_shift
AUDIO_FLOW_SHIFT = PROFILES[DEFAULT_PROFILE].audio_flow_shift
ALPHA = PROFILES[DEFAULT_PROFILE].alpha


def _validate_metadata(
    metadata: Mapping[str, str],
    expected: Mapping[str, str | None] = EXPECTED_METADATA,
) -> None:
    mismatches = {
        key: {
            "expected": expected_value,
            "observed": metadata.get(key)
        }
        for key, expected_value in expected.items() if metadata.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"LightX2V Turbo metadata mismatch: {mismatches}")


def _audit_checkpoint(
    path: Path,
    profile: ArtifactProfile = PROFILES[DEFAULT_PROFILE],
) -> dict[str, object]:
    from safetensors import safe_open

    if path.name != profile.filename:
        raise ValueError(f"LightX2V preset requires the published filename {profile.filename}")
    if path.stat().st_size != profile.expected_bytes:
        raise ValueError(f"LightX2V Turbo size mismatch: {path.stat().st_size} != {profile.expected_bytes}")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = list(checkpoint.keys())
        dtypes = Counter(checkpoint.get_slice(key).get_dtype() for key in keys)
        shapes = Counter(tuple(checkpoint.get_slice(key).get_shape()) for key in keys)
    _validate_metadata(metadata, profile.expected_metadata)
    if len(keys) != EXPECTED_TENSORS:
        raise ValueError(f"LightX2V Turbo tensor count mismatch: {len(keys)} != {EXPECTED_TENSORS}")
    if dtypes != Counter({"BF16": EXPECTED_TENSORS}):
        raise ValueError(f"LightX2V Turbo must contain only BF16 tensors, got {dict(dtypes)}")
    if shapes != Counter(EXPECTED_SHAPES):
        raise ValueError(f"LightX2V Turbo shape contract mismatch: {dict(shapes)}")
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256 != profile.expected_sha256:
        raise ValueError(f"LightX2V Turbo SHA256 mismatch: {sha256} != {profile.expected_sha256}")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256,
        "tensor_count": len(keys),
        "tensor_dtypes": dict(dtypes),
        "tensor_shapes": {
            "x".join(map(str, shape)): count
            for shape, count in sorted(shapes.items())
        },
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    destination = args.local_dir.expanduser().resolve()
    profile = PROFILES[args.profile]

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    from modelscope import snapshot_download

    resolved = Path(
        snapshot_download(
            model_id=MODEL_ID,
            revision=args.revision,
            local_dir=str(destination),
            allow_patterns=[profile.filename, "README.md", "configuration.json"],
            max_workers=4,
        )).resolve()
    artifact = resolved / profile.filename
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    audit = _audit_checkpoint(artifact, profile)
    result = {
        "schema_version": 1,
        "status": "pass",
        "download_client": "modelscope.snapshot_download",
        "huggingface_network_disabled": True,
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "profile": args.profile,
        "resolved_path": str(resolved),
        "artifact": {
            "path": str(artifact),
            **audit,
            "task_family": profile.task_family,
            "precision": "bf16_base_plus_bf16_lora",
            "training_resolution": profile.resolution,
            "rank": RANK,
            "alpha": profile.alpha,
            "sigma_points": SIGMA_POINTS,
            "actual_joint_dit_forwards": DENOISER_FORWARDS,
            "video_flow_shift": profile.video_flow_shift,
            "audio_flow_shift": profile.audio_flow_shift,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    manifest_path = args.manifest or destination / f"modelscope-download-{args.profile}-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
