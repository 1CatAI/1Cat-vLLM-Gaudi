#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submit a validated MiniMax H3 request to vLLM Omni's synchronous video API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import time
from typing import BinaryIO

import requests

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
_VIDEO_TYPES = {"video/mp4", "video/quicktime"}
_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg"}
_T2VA_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
_OFFICIAL_BASE_SIGMA_POINTS = 50
_BALANCED_SIGMA_POINTS = 10


def _response_metrics(headers: dict[str, str]) -> dict[str, object]:
    """Decode the sync endpoint's timing headers into stable JSON fields."""

    normalized = {name.lower(): value for name, value in headers.items()}
    raw_stages = normalized.get("x-stage-durations", "{}")
    try:
        stages = json.loads(raw_stages)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid X-Stage-Durations response header: {raw_stages!r}") from exc
    if not isinstance(stages, dict) or any(not isinstance(name, str) or not isinstance(value, (int, float))
                                           for name, value in stages.items()):
        raise ValueError(f"invalid X-Stage-Durations response header: {raw_stages!r}")

    def optional_float(name: str) -> float | None:
        value = normalized.get(name)
        return None if value is None else float(value)

    return {
        "request_id": normalized.get("x-request-id"),
        "model": normalized.get("x-model"),
        "server_inference_seconds": optional_float("x-inference-time-s"),
        "stage_durations_seconds": {
            name: float(value)
            for name, value in stages.items()
        },
        "peak_device_memory_mb": optional_float("x-peak-memory-mb"),
    }


def _resolve_sigma_points(requested: int | None, official_base_default: bool) -> int | None:
    """Use a short practical schedule unless the official Base grid is requested."""

    if official_base_default:
        return None
    return _BALANCED_SIGMA_POINTS if requested is None else requested


def _mime_type(path: Path) -> str:
    overrides = {".heic": "image/heic", ".heif": "image/heif", ".wav": "audio/wav", ".mp3": "audio/mpeg"}
    result = overrides.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    if result not in _IMAGE_TYPES | _VIDEO_TYPES | _AUDIO_TYPES:
        raise ValueError(f"unsupported reference type for {path}: {result}")
    return result


def _validate(args: argparse.Namespace, references: list[tuple[Path, str]]) -> None:
    if not args.prompt.strip():
        raise ValueError("prompt must be non-empty")
    if not 4.0 <= args.duration <= 15.0:
        raise ValueError("duration must be between 4 and 15 seconds")
    if args.fps != 24:
        raise ValueError("MiniMax H3 output is fixed at 24 FPS")
    if args.sigma_points is not None and args.sigma_points < 2:
        raise ValueError("MiniMax H3 needs at least two sigma grid points")
    if args.flow_shift is not None and args.flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    if args.audio_flow_shift is not None and args.audio_flow_shift <= 0:
        raise ValueError("audio_flow_shift must be positive")
    if (args.width is not None or args.height is not None) and (args.width is None or args.height is None
                                                                or args.width % 32 or args.height % 32):
        raise ValueError("width and height must both be present and divisible by 32")
    images = sum(mime in _IMAGE_TYPES for _, mime in references)
    videos = sum(mime in _VIDEO_TYPES for _, mime in references)
    audios = sum(mime in _AUDIO_TYPES for _, mime in references)
    if args.task == "t2va":
        if references or args.audio_url:
            raise ValueError("T2VA is text-only")
        if args.aspect_ratio not in _T2VA_RATIOS:
            raise ValueError(f"T2VA aspect_ratio must be one of {sorted(_T2VA_RATIOS)}")
    elif args.task == "fl2va":
        if not 1 <= images <= 2 or videos or audios or args.audio_url:
            raise ValueError("FL2VA requires one or two image references")
        expected = [0] if images == 1 else [0, -1]
        frame_indices = args.frame_indices if args.frame_indices is not None else expected
        if images == 1 and frame_indices not in ([0], [-1]):
            raise ValueError("single-image FL2VA frame_indices must be [0] or [-1]")
        if images == 2 and frame_indices != [0, -1]:
            raise ValueError("two-image FL2VA frame_indices must be [0, -1]")
        args.frame_indices = frame_indices
    else:
        if images > 9 or videos > 3 or audios > 3 or len(references) > 12:
            raise ValueError("Ref2VA exceeds the image/video/audio or 12-file limit")
        if not images and not videos:
            raise ValueError("Ref2VA requires at least one visual reference")
        if args.audio_url and audios:
            raise ValueError("use either --audio-url or uploaded audio references")
        if args.short_edge != 768:
            raise ValueError("Ref2VA short_edge must be 768")


def _request_data(args: argparse.Namespace) -> dict[str, str]:
    extra = {"task": args.task, "duration": args.duration}
    if args.audio_flow_shift is not None:
        extra["audio_flow_shift"] = args.audio_flow_shift
    if args.frame_indices is not None:
        extra["frame_indices"] = args.frame_indices
    data = {
        "prompt": args.prompt,
        "fps": str(args.fps),
        "seed": str(args.seed),
        "extra_params": json.dumps(extra, separators=(",", ":")),
    }
    # MiniMax H3 Base defines 50 sigma grid points in its official serving
    # workflow. The terminal zero is included, so this resolves to 49 joint
    # video/audio DiT forwards. Omitting the field exercises that model-owned
    # default instead of presenting it as 50 transformer steps.
    if args.sigma_points is not None:
        data["num_inference_steps"] = str(args.sigma_points)
    if args.flow_shift is not None:
        data["flow_shift"] = str(args.flow_shift)
    if args.width is not None:
        data.update(width=str(args.width), height=str(args.height))
    if args.task != "fl2va":
        data["aspect_ratio"] = args.aspect_ratio
    if args.task == "ref2va":
        data["short_edge"] = str(args.short_edge)
    if args.audio_url:
        data["audio_reference"] = json.dumps({"audio_url": args.audio_url}, separators=(",", ":"))
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8097/v1/videos/sync")
    parser.add_argument("--task", required=True, choices=("t2va", "fl2va", "ref2va"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--audio-url")
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--short-edge", type=int, default=768)
    parser.add_argument("--fps", type=int, default=24)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument(
        "--sigma-points",
        "--steps",
        dest="sigma_points",
        type=int,
        help="sigma grid points including terminal zero; default: 10 points / 9 DiT forwards",
    )
    schedule.add_argument(
        "--official-base-default",
        action="store_true",
        help="omit the field and use the Base checkpoint's 50-point / 49-forward schedule",
    )
    parser.add_argument("--flow-shift", type=float, help="omitted uses checkpoint release metadata (Base: 12)")
    parser.add_argument(
        "--audio-flow-shift",
        type=float,
        help="omitted uses checkpoint release metadata (Base: 3)",
    )
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--timeout", type=float, default=14400.0)
    args = parser.parse_args()
    args.sigma_points = _resolve_sigma_points(args.sigma_points, args.official_base_default)

    references = [(path.expanduser().resolve(), _mime_type(path)) for path in args.reference]
    for path, _ in references:
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate(args, references)
    data = _request_data(args)

    streams: list[BinaryIO] = []
    files = []
    field = "input_reference" if len(references) == 1 else "input_references"
    try:
        for path, mime in references:
            stream = path.open("rb")
            streams.append(stream)
            files.append((field, (path.name, stream, mime)))
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        with requests.post(args.api_url, data=data, files=files, timeout=args.timeout, stream=True) as response:
            response.raise_for_status()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("wb") as output:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    output.write(chunk)
            headers = dict(response.headers)
        elapsed = time.perf_counter() - started
    finally:
        for stream in streams:
            stream.close()

    metadata = {
        "schema_version": 1,
        "status": "pass",
        "started_at": started_at,
        "wall_seconds": elapsed,
        "api_url": args.api_url,
        "task": args.task,
        "request": data,
        "sampling_contract": {
            "request_field": "num_inference_steps",
            "meaning": "sigma_grid_points_including_terminal_zero",
            "field_was_omitted": args.sigma_points is None,
            "resolved_sigma_points": args.sigma_points or _OFFICIAL_BASE_SIGMA_POINTS,
            "expected_joint_dit_forwards": (args.sigma_points or _OFFICIAL_BASE_SIGMA_POINTS) - 1,
            "video_and_audio_share_each_forward": True,
        },
        "references": [{
            "path": str(path),
            "mime_type": mime
        } for path, mime in references],
        "response_headers": headers,
        "response_metrics": _response_metrics(headers),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
    }
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
