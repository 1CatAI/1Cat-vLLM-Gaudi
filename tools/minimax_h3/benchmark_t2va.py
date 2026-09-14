#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and validate a single-Gaudi MiniMax H3 Base T2VA benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import imageio_ffmpeg

DEFAULT_PROMPT = ("A red fox walks through a snowy forest while its footsteps crunch softly "
                  "and winter wind moves the trees.")
DEFAULT_FFPROBE = Path("/opt/habanalabs/media/ffmpeg/bin/ffprobe")
DEFAULT_VALIDATION_FFMPEG = Path("/opt/habanalabs/media/ffmpeg/bin/ffmpeg")
DEFAULT_ENCODING_FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
PINNED_OMNI_REFERENCE_SIGMA_POINTS = 50
FLASHGEN_INFERENCE_STEPS = 4
FLASHGEN_SIGMA_POINTS = 5


def _sampling_plan(args: argparse.Namespace) -> tuple[int, int, str]:
    if getattr(args, "flashgen_lora", None) is not None:
        return FLASHGEN_SIGMA_POINTS, FLASHGEN_INFERENCE_STEPS, "flashgen_4step_768p"
    if args.pinned_omni_reference:
        return PINNED_OMNI_REFERENCE_SIGMA_POINTS, PINNED_OMNI_REFERENCE_SIGMA_POINTS - 1, "pinned_omni_base"
    return args.sigma_points, args.sigma_points - 1, "explicit_base_grid"


def _run(command: list[str], log_path: Path, *, timeout: float | None = None) -> None:
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def _probe(path: Path, ffprobe: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,"
        "duration,nb_frames,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def _stream(probe: dict[str, Any], codec_type: str) -> dict[str, Any]:
    streams = [item for item in probe.get("streams", []) if item.get("codec_type") == codec_type]
    if len(streams) != 1:
        raise ValueError(f"expected one {codec_type} stream, got {len(streams)}")
    return streams[0]


def _validate_media(probe: dict[str, Any], *, width: int, height: int, frames: int,
                    exact_duration: float | None) -> dict[str, Any]:
    video = _stream(probe, "video")
    audio = _stream(probe, "audio")
    observed_frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    checks = {
        "video_codec_h264": video.get("codec_name") == "h264",
        "dimensions": (int(video.get("width", 0)), int(video.get("height", 0))) == (width, height),
        "frame_rate_24": video.get("r_frame_rate") == "24/1",
        "frame_count": observed_frames == frames,
        "audio_codec_aac": audio.get("codec_name") == "aac",
        "audio_32khz_stereo": (audio.get("sample_rate"), int(audio.get("channels", 0))) == ("32000", 2),
    }
    if exact_duration is not None:
        duration = float(probe.get("format", {}).get("duration", 0.0))
        checks["duration"] = abs(duration - exact_duration) <= 0.02
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"media validation failed: {failed}; probe={probe}")
    return {"status": "pass", "checks": checks, "probe": probe}


def _trim_delivery(source: Path, destination: Path, ffmpeg: Path) -> None:
    filter_graph = ("[0:v:0]trim=end_frame=120,setpts=PTS-STARTPTS[v];"
                    "[0:a:0]atrim=duration=5,asetpts=PTS-STARTPTS[a]")
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-r",
        "24",
        "-c:a",
        "aac",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-t",
        "5",
        str(destination),
    ]
    subprocess.run(command, check=True)


def _module_status(module: int) -> tuple[int, float]:
    command = [
        "hl-smi",
        "--query-aip=module_id,memory.used,utilization.aip",
        "--format=csv,noheader,nounits",
    ]
    for line in subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if int(fields[0]) == module:
            return int(fields[1]), float(fields[2])
    raise RuntimeError(f"hl-smi did not report module {module}")


def _rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


def _available_memory_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _monitor_resources(*, stop: threading.Event, path: Path, module: int, server_pid: int, interval: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "epoch_seconds",
            "hbm_used_mib",
            "hpu_util_percent",
            "server_rss_kib",
            "host_mem_available_kib",
            "error",
        ])
        while True:
            try:
                hbm, utilization = _module_status(module)
                writer.writerow(
                    [time.time(), hbm, utilization,
                     _rss_kib(server_pid) or "",
                     _available_memory_kib(), ""])
                output.flush()
            except Exception as exc:
                writer.writerow([time.time(), "", "", "", "", f"monitor_error={exc!r}"])
                output.flush()
            if stop.wait(interval):
                break


def _resource_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    hbm = [int(row["hbm_used_mib"]) for row in rows if row.get("hbm_used_mib")]
    rss = [int(row["server_rss_kib"]) for row in rows if row.get("server_rss_kib")]
    available = [int(row["host_mem_available_kib"]) for row in rows if row.get("host_mem_available_kib")]
    return {
        "samples": len(rows),
        "hbm_used_mib": {
            "min": min(hbm),
            "max": max(hbm)
        },
        "server_rss_kib": {
            "min": min(rss),
            "max": max(rss)
        },
        "host_mem_available_kib": {
            "min": min(available),
            "max": max(available)
        },
    }


def _progress_total(server_log: Path, start_offset: int) -> int | None:
    with server_log.open("rb") as source:
        source.seek(start_offset)
        text = source.read().decode("utf-8", errors="replace")
    matches = [(int(done), int(total)) for done, total in re.findall(r"(\d+)/(\d+)", text)]
    completed = [total for done, total in matches if done == total and total > 0]
    return completed[-1] if completed else None


def _request_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("request_video.py")),
        "--api-url",
        args.api_url,
        "--task",
        "t2va",
        "--prompt",
        args.prompt,
        "--output",
        str(run_dir / "internal-124f.mp4"),
        "--metadata",
        str(run_dir / "request.json"),
        "--duration",
        "5",
        "--width",
        "1344",
        "--height",
        "768",
        "--aspect-ratio",
        "16:9",
        "--fps",
        "24",
        "--seed",
        "2101",
        "--timeout",
        str(args.timeout),
    ]
    if getattr(args, "flashgen_lora", None) is not None:
        command.extend(("--flashgen-4step-lora", str(args.flashgen_lora)))
    elif args.pinned_omni_reference:
        command.append("--pinned-omni-reference")
    else:
        command.extend(("--sigma-points", str(args.sigma_points)))
    return command


def _one_run(args: argparse.Namespace, index: int) -> dict[str, Any]:
    run_dir = args.output_dir / f"request-{index:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_offset = args.server_log.stat().st_size
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_resources,
        kwargs={
            "stop": stop,
            "path": run_dir / "resources.csv",
            "module": args.module,
            "server_pid": args.server_pid,
            "interval": args.poll_seconds,
        },
        daemon=True,
    )
    monitor.start()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        _run(_request_command(args, run_dir), run_dir / "client.log", timeout=args.timeout + 60)
    finally:
        stop.set()
        monitor.join()
    harness_wall_seconds = time.perf_counter() - started

    request_metadata = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    # request_video measures immediately around requests.post through the last
    # response byte. Keep subprocess startup and evidence serialization out of
    # the protocol's submit-to-complete-MP4 wall-clock metric.
    wall_seconds = float(request_metadata["wall_seconds"])

    internal = run_dir / "internal-124f.mp4"
    delivery = run_dir / "delivery-120f-5s.mp4"
    internal_validation = _validate_media(_probe(internal, args.ffprobe),
                                          width=1344,
                                          height=768,
                                          frames=124,
                                          exact_duration=None)
    _run(
        [
            str(args.validation_ffmpeg),
            "-v",
            "error",
            "-i",
            str(internal),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        run_dir / "full-decode.log",
    )
    _trim_delivery(internal, delivery, args.encoding_ffmpeg)
    delivery_validation = _validate_media(_probe(delivery, args.ffprobe),
                                          width=1344,
                                          height=768,
                                          frames=120,
                                          exact_duration=5.0)
    sigma_points, expected_dit_forwards, sampling_profile = _sampling_plan(args)
    dit_forwards = _progress_total(args.server_log, log_offset)
    if dit_forwards != expected_dit_forwards:
        raise RuntimeError(
            f"request completed with {dit_forwards!r} logged DiT forwards, expected {expected_dit_forwards}")
    result = {
        "schema_version": 1,
        "status": "pass",
        "run_index": index,
        "started_at": started_at,
        "wall_seconds": wall_seconds,
        "harness_wall_seconds": harness_wall_seconds,
        "sampling_profile": sampling_profile,
        "request_sampling_fields_omitted": args.pinned_omni_reference,
        "resolved_sigma_points": sigma_points,
        "actual_joint_dit_forwards": dit_forwards,
        "profiler_enabled": False,
        "server_response_metrics": request_metadata.get("response_metrics", {}),
        "resources": _resource_summary(run_dir / "resources.csv"),
        "internal_output": str(internal),
        "delivery_output": str(delivery),
        "internal_validation": internal_validation,
        "delivery_validation": delivery_validation,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8097/v1/videos/sync")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--module", type=int, required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument(
        "--flashgen-4step-lora",
        "--flashgen-lora",
        dest="flashgen_lora",
        type=Path,
        help="benchmark the published ModelScope FlashGen four-forward 768p schedule",
    )
    schedule.add_argument(
        "--denoise-steps",
        type=int,
        help="experimental Base Euler DiT calls (translated to one more sigma grid point)",
    )
    schedule.add_argument(
        "--sigma-points",
        type=int,
        help="expert override: Omni sigma grid points including terminal zero",
    )
    schedule.add_argument(
        "--pinned-omni-reference",
        "--official-base-default",
        dest="pinned_omni_reference",
        action="store_true",
        help="deliberately reproduce pinned Omni's 50-point / 49-forward Base schedule",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--validation-ffmpeg", type=Path, default=DEFAULT_VALIDATION_FFMPEG)
    parser.add_argument("--encoding-ffmpeg", type=Path, default=DEFAULT_ENCODING_FFMPEG)
    args = parser.parse_args()
    if args.runs < 4:
        raise ValueError("the acceptance protocol needs one first request and at least three subsequent requests")
    if args.flashgen_lora is not None:
        args.flashgen_lora = args.flashgen_lora.expanduser().resolve()
        if not args.flashgen_lora.is_file():
            raise FileNotFoundError(args.flashgen_lora)
    elif args.sigma_points is None and not args.pinned_omni_reference:
        if args.denoise_steps is None:
            args.pinned_omni_reference = True
        elif args.denoise_steps < 1:
            raise ValueError("MiniMax H3 needs at least one denoise step")
        else:
            args.sigma_points = args.denoise_steps + 1
    if args.sigma_points is not None and args.sigma_points < 2:
        raise ValueError("MiniMax H3 needs at least two sigma grid points")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.server_log = args.server_log.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    flashgen = args.flashgen_lora is not None
    sigma_points, actual_forwards, sampling_profile = _sampling_plan(args)
    sampler = "flashgen_dmd2_base_schedule" if flashgen else "omni_euler_eta0"
    results = [_one_run(args, index) for index in range(args.runs)]
    warmed = [item["wall_seconds"] for item in results[1:4]]
    summary = {
        "schema_version": 1,
        "status": "pass",
        "workload": {
            "task": "t2va",
            "width": 1344,
            "height": 768,
            "fps": 24,
            "requested_duration_seconds": 5,
            "internal_frames": 124,
            "delivery_frames": 120,
            "seed": 2101,
            "sampling_profile": sampling_profile,
            "sampling_fields_omitted": args.pinned_omni_reference,
            "sampler": sampler,
            "resolved_sigma_points": sigma_points,
            "actual_joint_dit_forwards": actual_forwards,
            "profiler_enabled": False,
        },
        "first_request_wall_seconds": results[0]["wall_seconds"],
        "subsequent_three_wall_seconds": warmed,
        "subsequent_three_median_seconds": statistics.median(warmed),
        "subsequent_three_range_seconds": [min(warmed), max(warmed)],
        "runs": results,
    }
    encoded = json.dumps(summary, indent=2) + "\n"
    (args.output_dir / "summary.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
