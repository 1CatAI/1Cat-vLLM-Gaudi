#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MiniMax H3 video-VAE spatial tile micro-batching on one HPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be a comma-separated list of positive integers")
    if 1 in sizes and sizes[0] != 1:
        raise argparse.ArgumentTypeError("batch size 1 must be first when this run records a component reference")
    if len(sizes) != len(set(sizes)):
        raise argparse.ArgumentTypeError("batch sizes must not contain duplicates")
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", type=Path, help="local MiniMax H3 video_vae component directory")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=_parse_batch_sizes("1,2,4,7,14,28"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--latent-t", type=int, default=37)
    parser.add_argument("--latent-height", type=int, default=48)
    parser.add_argument("--latent-width", type=int, default=84)
    parser.add_argument("--decoder-tile-size", type=int)
    parser.add_argument("--decoder-tile-overlap", type=int)
    args = parser.parse_args()
    if (args.decoder_tile_size is None) != (args.decoder_tile_overlap is None):
        parser.error("--decoder-tile-size and --decoder-tile-overlap must be set together")
    if args.decoder_tile_size is not None:
        if args.decoder_tile_size <= 0:
            parser.error("--decoder-tile-size must be positive")
        if not 0 <= args.decoder_tile_overlap < args.decoder_tile_size:
            parser.error("--decoder-tile-overlap must be non-negative and smaller than the tile size")
    return args


def _first_temporal_clip(model: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    token_drop = int(model.token_drop)
    chunk_size = int(model.tokens_chunk_size)
    overlap = int(model.token_overlap)
    pseudo_tokens = int(latent.shape[2]) + token_drop
    pad_tokens = (-pseudo_tokens) % chunk_size
    if pad_tokens:
        latent = torch.cat((latent, latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)
    return latent[:, :, :chunk_size + overlap]


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


def main() -> int:
    args = _parse_args()
    if not torch.hpu.is_available():
        raise RuntimeError("this benchmark requires one visible HPU")
    if os.environ.get("PT_HPU_LAZY_MODE", "0") != "0":
        raise RuntimeError("set PT_HPU_LAZY_MODE=0 for comparable eager timings")

    args.component = args.component.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.reference_raw = args.reference_raw.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reference_raw.parent.mkdir(parents=True, exist_ok=True)

    from vllm_gaudi.omni import register_omni
    from vllm_gaudi.omni.minimax_h3_vae import _install_h3_vae_decode_tile_batching

    os.environ.setdefault("VLLM_GAUDI_H3_PHASE_OFFLOAD", "0")
    register_omni()

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _prepare_minimax_h3_video_output, )
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    device = torch.device("hpu")
    load_started = time.perf_counter()
    vae = MiniMaxH3VideoVAE(str(args.component), device=device, load_device=device)
    torch.hpu.synchronize()
    load_seconds = time.perf_counter() - load_started
    model = vae.model
    if args.decoder_tile_size is not None:
        ratio = int(model.vae_ratio)
        if args.decoder_tile_size % ratio or args.decoder_tile_overlap % ratio:
            raise ValueError(f"decoder tile size and overlap must be divisible by the VAE ratio {ratio}")
        model.decoder_tile_size = args.decoder_tile_size
        model.decoder_tile_overlap_min = args.decoder_tile_overlap
    if not _install_h3_vae_decode_tile_batching(model, args.batch_sizes[0]):
        raise RuntimeError("loaded video VAE does not expose the expected tile contract")

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
    first_clip = _first_temporal_clip(model, denormalized_latent)

    output_frames = 17 * ((args.latent_t - 2) // 5) + 5
    raw_shape = (output_frames, args.latent_height * 16, args.latent_width * 16, 3)
    raw_bytes = math.prod(raw_shape)
    results: list[dict[str, Any]] = []
    reference_sha256: str | None = None
    if 1 not in args.batch_sizes:
        if not args.reference_raw.is_file() or args.reference_raw.stat().st_size != raw_bytes:
            raise ValueError("candidate-only runs require an existing reference raw file with "
                             f"{raw_bytes} bytes, got {args.reference_raw}")
        with args.reference_raw.open("rb") as stream:
            reference_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    for batch_size in args.batch_sizes:
        model._vllm_gaudi_decode_tile_batch_size = batch_size
        # Warm the exact per-tile temporal and spatial shapes without repeating
        # another full component reference.
        with torch.inference_mode(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            model._adaptive_decode(first_clip)
        torch.hpu.synchronize()
        _reset_peak_hbm()

        offset = 0
        digest = hashlib.sha256()
        mismatch_count = 0
        sum_squared_error = 0
        max_abs = 0
        if batch_size == 1:
            reference = np.memmap(args.reference_raw, mode="w+", dtype=np.uint8, shape=raw_shape)
        else:
            reference = np.memmap(args.reference_raw, mode="r", dtype=np.uint8, shape=raw_shape)

        def consume(
            frames: torch.Tensor,
            _batch_size: int = batch_size,
            _digest: Any = digest,
            _reference: np.memmap = reference,
        ) -> None:
            nonlocal offset, mismatch_count, sum_squared_error, max_abs
            prepared = _prepare_minimax_h3_video_output(frames)[0].cpu().numpy()
            count = int(prepared.shape[0])
            _digest.update(prepared.tobytes())
            if _batch_size == 1:
                _reference[offset:offset + count] = prepared
            else:
                expected = np.asarray(_reference[offset:offset + count])
                difference = prepared.astype(np.int16) - expected.astype(np.int16)
                mismatch_count += int(np.count_nonzero(difference))
                sum_squared_error += int(np.square(difference.astype(np.int32)).sum())
                max_abs = max(max_abs, int(np.abs(difference).max(initial=0)))
            offset += count

        begin = torch.hpu.Event(enable_timing=True)
        end = torch.hpu.Event(enable_timing=True)
        torch.hpu.synchronize()
        host_started = time.perf_counter_ns()
        begin.record()
        with torch.inference_mode(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            vae.decode_with_chunks(normalized_latent, on_chunk=consume)
        end.record()
        torch.hpu.synchronize()
        host_ms = (time.perf_counter_ns() - host_started) / 1e6
        device_ms = begin.elapsed_time(end)
        if offset != output_frames:
            raise RuntimeError(f"decoded {offset} frames, expected {output_frames}")
        if batch_size == 1:
            reference.flush()
            reference_sha256 = digest.hexdigest()

        mse = sum_squared_error / raw_bytes if batch_size != 1 else 0.0
        result = {
            "tile_batch_size": batch_size,
            "device_ms": device_ms,
            "synchronized_host_ms": host_ms,
            "max_hbm_bytes": _maximum_hbm_bytes(),
            "output_sha256": digest.hexdigest(),
            "reference_sha256": reference_sha256,
            "uint8_mismatch_count": mismatch_count,
            "uint8_max_abs": max_abs,
            "uint8_mse": mse,
            "uint8_psnr_db": None if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse)),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del reference

    report = {
        "schema_version": 1,
        "status": "pass",
        "device": torch.hpu.get_device_name(),
        "visible_modules": os.environ.get("HABANA_VISIBLE_MODULES"),
        "hls_module_id": os.environ.get("HLS_MODULE_ID"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "component": str(args.component),
        "load_seconds": load_seconds,
        "latent_shape": [1, 24, args.latent_t, args.latent_height, args.latent_width],
        "decoded_shape": list(raw_shape),
        "spatial_tile_count": vae._decoder_tile_count(normalized_latent),
        "temporal_chunk_count": (args.latent_t + int(model.token_drop)) // int(model.tokens_chunk_size) - 1,
        "decoder_tile_size": int(model.decoder_tile_size),
        "decoder_tile_overlap": int(model.decoder_tile_overlap_min),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
