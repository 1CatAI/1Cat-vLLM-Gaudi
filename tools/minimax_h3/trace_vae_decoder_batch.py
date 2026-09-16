#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace one production-shape MiniMax H3 video-VAE decoder tile batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from benchmark_vae_tile_batch import _first_temporal_clip


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", type=Path, help="local MiniMax H3 video_vae component directory")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-npy", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--latent-t", type=int, default=37)
    parser.add_argument("--latent-height", type=int, default=48)
    parser.add_argument("--latent-width", type=int, default=84)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    return args


def _spatial_tiles(model: torch.nn.Module, latent: torch.Tensor) -> list[torch.Tensor]:
    ratio = int(model.vae_ratio)
    y_indices, y_lengths, _ = model.split_tiles(int(latent.shape[-2]) * ratio, True)
    x_indices, x_lengths, _ = model.split_tiles(int(latent.shape[-1]) * ratio, True)
    return [
        latent[
            ...,
            y_pos // ratio:(y_pos + y_length) // ratio,
            x_pos // ratio:(x_pos + x_length) // ratio,
        ] for y_pos, y_length in zip(y_indices, y_lengths) for x_pos, x_length in zip(x_indices, x_lengths)
    ]


def _maximum_hbm_bytes() -> int | None:
    for name in ("max_memory_allocated", "memory_allocated"):
        function = getattr(torch.hpu, name, None)
        if callable(function):
            try:
                return int(function())
            except Exception:
                continue
    return None


def _reset_peak_hbm() -> None:
    function = getattr(torch.hpu, "reset_peak_memory_stats", None)
    if callable(function):
        function()


def _compare_first_sample(output: torch.Tensor, reference_path: Path, record_reference: bool) -> dict[str, Any]:
    sample = output[:1].float().cpu().numpy()
    digest = hashlib.sha256(sample.tobytes()).hexdigest()
    if record_reference:
        np.save(reference_path, sample)
        return {
            "first_sample_sha256": digest,
            "reference_sha256": digest,
            "mismatch_count": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }

    reference = np.load(reference_path, mmap_mode="r")
    if reference.shape != sample.shape:
        raise ValueError(f"reference shape {reference.shape} does not match output shape {sample.shape}")
    difference = sample - reference
    return {
        "first_sample_sha256": digest,
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "mismatch_count": int(np.count_nonzero(difference)),
        "max_abs": float(np.abs(difference).max(initial=0.0)),
        "mean_abs": float(np.abs(difference).mean()),
    }


def main() -> int:
    args = _parse_args()
    if not torch.hpu.is_available():
        raise RuntimeError("this trace requires one visible HPU")
    if os.environ.get("PT_HPU_LAZY_MODE", "0") != "0":
        raise RuntimeError("set PT_HPU_LAZY_MODE=0 for comparable eager traces")

    args.component = args.component.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.reference_npy = args.reference_npy.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.reference_npy.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / f"vae-decoder-batch-{args.batch_size}.trace.json"
    result_path = args.output_dir / "result.json"

    from vllm_gaudi.omni import register_omni

    os.environ.setdefault("VLLM_GAUDI_H3_PHASE_OFFLOAD", "0")
    register_omni()

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    device = torch.device("hpu")
    load_started = time.perf_counter()
    vae = MiniMaxH3VideoVAE(str(args.component), device=device, load_device=device)
    torch.hpu.synchronize()
    load_seconds = time.perf_counter() - load_started
    model = vae.model

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    normalized_latent = torch.randn(
        1,
        24,
        args.latent_t,
        args.latent_height,
        args.latent_width,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    denormalized_latent = vae._denormalize_latent(normalized_latent)
    temporal_clip = _first_temporal_clip(model, denormalized_latent)
    tiles = _spatial_tiles(model, temporal_clip)
    if args.batch_size > len(tiles):
        raise ValueError(f"batch size {args.batch_size} exceeds the {len(tiles)} spatial tiles")
    decoder_input = torch.cat(tiles[:args.batch_size], dim=0)

    with torch.inference_mode(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
        for _ in range(args.warmups):
            model.decode(decoder_input)
    torch.hpu.synchronize()
    _reset_peak_hbm()

    marker = f"minimax_h3_vae_decoder_batch_{args.batch_size}"
    begin = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    torch.hpu.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
    ) as profiler, torch.profiler.record_function(marker):
        host_started = time.perf_counter_ns()
        begin.record()
        with torch.inference_mode(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            output = model.decode(decoder_input)
        end.record()
        torch.hpu.synchronize()
        synchronized_host_ms = (time.perf_counter_ns() - host_started) / 1e6
    device_ms = begin.elapsed_time(end)
    profiler.export_chrome_trace(str(trace_path))

    record_reference = args.batch_size == 1
    if not record_reference and not args.reference_npy.is_file():
        raise ValueError(f"candidate trace requires the batch-1 reference at {args.reference_npy}")
    quality = _compare_first_sample(output, args.reference_npy, record_reference)
    report = {
        "schema_version": 1,
        "status": "pass",
        "device": torch.hpu.get_device_name(),
        "visible_modules": os.environ.get("HABANA_VISIBLE_MODULES"),
        "hls_module_id": os.environ.get("HLS_MODULE_ID"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "component": str(args.component),
        "load_seconds": load_seconds,
        "batch_size": args.batch_size,
        "spatial_tile_count": len(tiles),
        "decoder_input_shape": list(decoder_input.shape),
        "decoder_input_dtype": str(decoder_input.dtype),
        "decoder_output_shape": list(output.shape),
        "decoder_output_dtype": str(output.dtype),
        "device_ms": device_ms,
        "synchronized_host_ms": synchronized_host_ms,
        "max_hbm_bytes": _maximum_hbm_bytes(),
        "marker": marker,
        "trace": trace_path.name,
        "trace_bytes": trace_path.stat().st_size,
        "quality": quality,
    }
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
