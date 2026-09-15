#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease one idle Gaudi module and serve a local H3 partition."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import IO, Any

_FLASHGEN_FILENAME = "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
_FLASHGEN_METADATA = {
    "key_format": "minimax-h3-native",
    "qkv_layout": "grouped",
    "lora_rank": "64",
    "lora_alpha": "64",
    "tasks": "t2va",
    "base_schedule": "1.0,0.7,0.4,0.15,0.0",
}
_FASTH3_RELATIVE_PATH = Path("dense-datafree/adapter_model.safetensors")
_FASTH3_BYTES = 1485626152
_FASTH3_SHA256 = "4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12"
_FASTH3_TENSORS = 809
_FASTH3_METADATA = {
    "format": "fastvideo-lora-v2",
    "base_model": "MiniMaxAI/MiniMax-H3",
    "finetuned_model": "FastVideo/FastVideo-FastH3-Dense-4-step-v1",
    "rank": "64",
    "low_rank_tensors": "724",
    "diff_tensors": "85",
    "set_weight_tensors": "0",
}
_LIGHTX2V_TENSORS = 624
_LIGHTX2V_SHAPES = {
    (128, 5376): 208,
    (128, 7168): 52,
    (128, 14336): 52,
    (5376, 128): 104,
    (7168, 128): 156,
    (28672, 128): 52,
}


@dataclass(frozen=True)
class _LightX2VArtifact:
    expected_bytes: int
    expected_sha256: str
    expected_metadata: Mapping[str, str | None]
    task_type: str


_LIGHTX2V_ARTIFACTS = {
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors":
    _LightX2VArtifact(
        expected_bytes=1383677808,
        expected_sha256="1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c",
        expected_metadata={
            "format": "pt",
            "floating_dtype": "bfloat16",
            "alpha": "128",
            "key_format": "minimax-h3-diffusers",
        },
        task_type="fl2va",
    ),
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors":
    _LightX2VArtifact(
        expected_bytes=1383677768,
        expected_sha256="9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
        expected_metadata={
            "format": "pt",
            "floating_dtype": "bfloat16",
            "alpha": "8",
            "key_format": None,
        },
        task_type="ref2va",
    ),
}
_LIGHTX2V_FILENAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
_LIGHTX2V_BYTES = _LIGHTX2V_ARTIFACTS[_LIGHTX2V_FILENAME].expected_bytes
_LIGHTX2V_SHA256 = _LIGHTX2V_ARTIFACTS[_LIGHTX2V_FILENAME].expected_sha256
_LIGHTX2V_METADATA = _LIGHTX2V_ARTIFACTS[_LIGHTX2V_FILENAME].expected_metadata
_DEFAULT_HABANA_MEDIA_BIN = Path("/opt/habanalabs/media/ffmpeg/bin")


def _parse_cpuset(value: str) -> set[int]:
    result: set[int] = set()
    for field in value.strip().split(","):
        if not field:
            continue
        bounds = [int(item) for item in field.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _resolve_media_bin(path: Path) -> Path:
    """Require the ffmpeg tools used by Ref2VA video inputs and MP4 output."""

    path = path.expanduser().resolve()
    missing = [name for name in ("ffmpeg", "ffprobe") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"media tool directory {path} is missing: {', '.join(missing)}")
    return path


def _prepend_media_path(environment: dict[str, str], media_bin: Path) -> None:
    current = environment.get("PATH", "")
    environment["PATH"] = str(media_bin) + (os.pathsep + current if current else "")


def _accelerators() -> list[dict[str, Any]]:
    devices = []
    for accelerator in sorted(Path("/sys/class/accel").glob("accel[0-9]*")):
        device = accelerator / "device"
        try:
            devices.append({
                "module": int((device / "module_id").read_text()),
                "numa": int((device / "numa_node").read_text()),
                "bus": device.resolve().name,
                "node": Path("/dev/accel") / accelerator.name,
            })
        except (FileNotFoundError, ValueError):
            continue
    return devices


def _device_is_idle(device: dict[str, Any]) -> bool:
    owner = subprocess.run(["fuser", str(device["node"])], capture_output=True, text=True)
    if owner.returncode != 1 or owner.stdout.strip() or owner.stderr.strip():
        return False
    status = subprocess.run(
        [
            "hl-smi",
            "-i",
            device["bus"],
            "--query-aip=memory.used,utilization.aip",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory_mib, utilization = (float(field.strip()) for field in status.split(","))
    return memory_mib <= 1024 and utilization == 0


def _try_lease(lock_dir: Path, requested_module: int | None) -> tuple[dict[str, Any], IO[str]] | None:
    for device in _accelerators():
        if requested_module is not None and device["module"] != requested_module:
            continue
        lock_path = lock_dir / f"gaudi-module{device['module']}.lock"
        stream = lock_path.open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _device_is_idle(device):
                return device, stream
        except BlockingIOError:
            pass
        stream.close()
    return None


def _checkpoint_formats(model: Path) -> dict[str, str]:
    formats = {}
    for component in ("transformer", "text_encoder"):
        config_path = model / component / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization_config")
        if quantization is None:
            formats[component] = "bf16"
        elif (quantization.get("quant_method") == "modelopt"
              and quantization.get("quant_algo") == "FP8_PER_CHANNEL_PER_TOKEN"):
            formats[component] = "modelopt_fp8_ptpc"
        else:
            raise ValueError(f"{config_path} has an unsupported quantization_config: {quantization}")
    return formats


def _resolve_partition(model: Path, partition: str | None) -> tuple[Path, str]:
    model = model.expanduser().resolve()
    if partition is not None and (model / partition).is_dir():
        model = model / partition
    inferred = model.name
    if inferred not in ("FL2VA", "Ref2VA"):
        raise ValueError("model must be a local FL2VA/Ref2VA directory or its parent with --partition")
    if partition is not None and inferred != partition:
        raise ValueError(f"requested partition {partition} does not match {model}")
    if not (model / "model_index.json").is_file():
        raise FileNotFoundError(model / "model_index.json")
    _checkpoint_formats(model)
    return model, inferred.lower()


def _resolve_flashgen_lora(path: Path | None, task_type: str) -> Path | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / _FLASHGEN_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name != _FLASHGEN_FILENAME:
        raise ValueError(f"FlashGen preset requires the published filename {_FLASHGEN_FILENAME}")
    if task_type != "fl2va":
        raise ValueError("FlashGen 4-step v1.0 requires the FL2VA partition")
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    mismatches = {
        key: {
            "expected": expected,
            "observed": metadata.get(key)
        }
        for key, expected in _FLASHGEN_METADATA.items() if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"FlashGen LoRA metadata mismatch: {mismatches}")
    return path


def _resolve_fasth3_adapter(path: Path | None, task_type: str) -> Path | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if path.is_dir():
        nested = path / _FASTH3_RELATIVE_PATH
        path = nested if nested.is_file() else path / _FASTH3_RELATIVE_PATH.name
    if not path.is_file():
        raise FileNotFoundError(path)
    if task_type != "fl2va":
        raise ValueError("FastH3 preview v1 is T2VA-only and requires the FL2VA partition")
    if path.stat().st_size != _FASTH3_BYTES:
        raise ValueError(f"FastH3 Dense-DataFree adapter size mismatch: {path.stat().st_size} != {_FASTH3_BYTES}")

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = checkpoint.keys()
        tensor_count = len(keys)
        tensor_dtypes = {checkpoint.get_slice(key).get_dtype() for key in keys}
    mismatches = {
        key: {
            "expected": expected,
            "observed": metadata.get(key)
        }
        for key, expected in _FASTH3_METADATA.items() if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"FastH3 adapter metadata mismatch: {mismatches}")
    if tensor_count != _FASTH3_TENSORS:
        raise ValueError(f"FastH3 Dense-DataFree adapter must contain {_FASTH3_TENSORS} tensors, got {tensor_count}")
    if tensor_dtypes != {"BF16"}:
        raise ValueError(f"FastH3 Dense-DataFree adapter must contain only BF16 tensors, got {sorted(tensor_dtypes)}")
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256 != _FASTH3_SHA256:
        raise ValueError(f"FastH3 Dense-DataFree adapter SHA256 mismatch: {sha256} != {_FASTH3_SHA256}")
    return path


def _resolve_lightx2v_lora(path: Path | None, task_type: str) -> Path | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = [path / filename for filename in _LIGHTX2V_ARTIFACTS if (path / filename).is_file()]
        if len(candidates) != 1:
            raise ValueError(f"{path} contains {len(candidates)} qualified LightX2V four-step artifacts; "
                             "point the option at the exact file")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = _LIGHTX2V_ARTIFACTS.get(path.name)
    if artifact is None:
        raise ValueError(f"unsupported LightX2V four-step artifact: {path.name}")
    if task_type != artifact.task_type:
        raise ValueError(f"{path.name} requires the {artifact.task_type.upper()} partition")
    if path.stat().st_size != artifact.expected_bytes:
        raise ValueError(f"LightX2V Turbo size mismatch: {path.stat().st_size} != {artifact.expected_bytes}")

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = list(checkpoint.keys())
        tensor_dtypes = Counter(checkpoint.get_slice(key).get_dtype() for key in keys)
        tensor_shapes = Counter(tuple(checkpoint.get_slice(key).get_shape()) for key in keys)
    mismatches = {
        key: {
            "expected": expected,
            "observed": metadata.get(key)
        }
        for key, expected in artifact.expected_metadata.items() if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"LightX2V Turbo metadata mismatch: {mismatches}")
    if len(keys) != _LIGHTX2V_TENSORS:
        raise ValueError(f"LightX2V Turbo must contain {_LIGHTX2V_TENSORS} tensors, got {len(keys)}")
    if tensor_dtypes != Counter({"BF16": _LIGHTX2V_TENSORS}):
        raise ValueError(f"LightX2V Turbo must contain only BF16 tensors, got {dict(tensor_dtypes)}")
    if tensor_shapes != Counter(_LIGHTX2V_SHAPES):
        raise ValueError(f"LightX2V Turbo shape contract mismatch: {dict(tensor_shapes)}")
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256 != artifact.expected_sha256:
        raise ValueError(f"LightX2V Turbo SHA256 mismatch: {sha256} != {artifact.expected_sha256}")
    return path


def _command(args: argparse.Namespace, model: Path, task_type: str) -> list[str]:
    offload = {
        "mode": "layer",
        "components": args.offload_component,
        "pin_memory": False,
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--task-type",
        task_type,
        "--num-gpus",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--enforce-eager",
        "--diffusion-offload-config",
        json.dumps(offload, separators=(",", ":")),
        "--diffusion-attention-backend",
        "HPU_SDPA",
    ]
    if args.flashgen_lora is not None:
        command.extend(("--lora-backend", "peft", "--lora-path", str(args.flashgen_lora)))
    if args.fasth3_adapter is not None:
        command.extend(("--lora-path", str(args.fasth3_adapter)))
    if args.lightx2v_lora is not None:
        command.extend(("--lora-backend", "peft", "--lora-path", str(args.lightx2v_lora)))
    if args.online_fp8:
        command.extend(("--diffusion-quantization-config", '{"method":"fp8"}'))
    return command + args.vllm_args


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local ModelScope checkpoint or FL2VA/Ref2VA partition")
    parser.add_argument("--partition", choices=("FL2VA", "Ref2VA"))
    parser.add_argument("--module", type=int, help="physical module ID; otherwise lease the first idle module")
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=Path(os.environ.get("VLLM_GAUDI_LOCK_DIR", "/tmp/1cat-gaudi-module-locks")),
    )
    parser.add_argument("--cpu-affinity", help="taskset-style CPU list; defaults to the selected module's NUMA node")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path(os.environ["VLLM_GAUDI_H3_TMPDIR"]) if os.environ.get("VLLM_GAUDI_H3_TMPDIR") else None,
        help="put scratch files on this filesystem; defaults to VLLM_GAUDI_H3_TMPDIR or the system temp dir",
    )
    parser.add_argument(
        "--media-bin",
        type=Path,
        default=Path(os.environ.get("VLLM_GAUDI_MEDIA_BIN", _DEFAULT_HABANA_MEDIA_BIN)),
        help="directory containing ffmpeg and ffprobe; defaults to Habana's media toolkit",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        "--flashgen-4step-lora",
        "--flashgen-lora",
        dest="flashgen_lora",
        type=Path,
        help="preload the published ModelScope FlashGen four-step T2VA adapter",
    )
    adapter_group.add_argument(
        "--fasth3-4step-adapter",
        "--fasth3-adapter",
        dest="fasth3_adapter",
        type=Path,
        help="fuse the published FastH3 Dense-DataFree student into the official BF16 transformer",
    )
    adapter_group.add_argument(
        "--lightx2v-4step-lora",
        "--lightx2v-lora",
        dest="lightx2v_lora",
        type=Path,
        help="preload a qualified LightX2V FL2V or Ref2V Turbo four-step BF16 LoRA",
    )
    parser.add_argument(
        "--online-fp8",
        action="store_true",
        help="quantize BF16 DiT/text-encoder linears to HPU per-channel/per-token FP8 while loading",
    )
    parser.add_argument(
        "--phase-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stage VAEs on CPU and offload the DiT between denoise and decode (default: enabled)",
    )
    parser.add_argument(
        "--vae-tile-batch-size",
        type=int,
        default=4,
        help="number of independent video-VAE spatial tiles decoded together (default: 4)",
    )
    parser.add_argument(
        "--vae-persist-bf16-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="store video-VAE decoder Linear weights in their HPU autocast dtype (default: enabled)",
    )
    parser.add_argument(
        "--offload-component",
        action="append",
        choices=("text_encoder", "dit"),
        default=None,
        help="layerwise component offload; default: text_encoder",
    )
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        separator = raw_args.index("--")
        own_args, vllm_args = raw_args[:separator], raw_args[separator + 1:]
    else:
        own_args, vllm_args = raw_args, []
    args = parser.parse_args(own_args)
    args.vllm_args = vllm_args
    args.offload_component = args.offload_component or ["text_encoder"]
    return args


def main() -> int:
    args = _parse_args()
    model, task_type = _resolve_partition(args.model, args.partition)
    args.media_bin = _resolve_media_bin(args.media_bin)
    if args.temp_dir is not None:
        args.temp_dir = args.temp_dir.expanduser().resolve()
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        if not args.temp_dir.is_dir():
            raise NotADirectoryError(args.temp_dir)
    formats = _checkpoint_formats(model)
    args.flashgen_lora = _resolve_flashgen_lora(args.flashgen_lora, task_type)
    args.fasth3_adapter = _resolve_fasth3_adapter(args.fasth3_adapter, task_type)
    args.lightx2v_lora = _resolve_lightx2v_lora(args.lightx2v_lora, task_type)
    if args.fasth3_adapter is not None:
        if formats["transformer"] != "bf16":
            raise ValueError("FastH3 requires a BF16 transformer so its published delta can be fused exactly")
        if "dit" in args.offload_component:
            raise ValueError(
                "FastH3 cannot use DiT offload because its load-time fusion must pass through load_weights()")
    if args.lightx2v_lora is not None:
        if any(checkpoint_format != "bf16" for checkpoint_format in formats.values()):
            raise ValueError("LightX2V qualification requires the official BF16 transformer and text encoder")
        if "dit" in args.offload_component:
            raise ValueError("LightX2V dynamic LoRA does not support layerwise DiT offload")
        if args.online_fp8:
            raise ValueError("LightX2V qualification follows the official BF16 base-plus-LoRA workflow")
    if args.online_fp8 and any(checkpoint_format != "bf16" for checkpoint_format in formats.values()):
        raise ValueError("--online-fp8 accepts only BF16 transformer and text-encoder checkpoints")
    if args.vae_tile_batch_size <= 0:
        raise ValueError("--vae-tile-batch-size must be positive")

    args.lock_dir.mkdir(parents=True, exist_ok=True)
    lease = _try_lease(args.lock_dir, args.module)
    while lease is None:
        target = f"module {args.module}" if args.module is not None else "an idle module"
        print(f"Waiting for {target}; existing jobs remain untouched", flush=True)
        time.sleep(args.poll_seconds)
        lease = _try_lease(args.lock_dir, args.module)
    device, lock = lease

    if args.cpu_affinity:
        affinity = _parse_cpuset(args.cpu_affinity)
    else:
        affinity = _parse_cpuset(Path(f"/sys/devices/system/node/node{device['numa']}/cpulist").read_text())
    affinity &= os.sched_getaffinity(0)
    if not affinity:
        lock.close()
        raise RuntimeError("selected CPU affinity is empty")

    environment = dict(os.environ)
    _prepend_media_path(environment, args.media_bin)
    environment.update(
        HABANA_VISIBLE_MODULES=str(device["module"]),
        HLS_MODULE_ID=str(device["module"]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        VLLM_OMNI_VIDEO_SYNC_TIMEOUT=str(environment.get("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", "14400")),
        PT_HPU_LAZY_MODE="0",
        PYTHONUNBUFFERED="1",
        VLLM_GAUDI_H3_PHASE_OFFLOAD="1" if args.phase_offload else "0",
        VLLM_GAUDI_H3_VAE_TILE_BATCH_SIZE=str(args.vae_tile_batch_size),
        VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS="1" if args.vae_persist_bf16_weights else "0",
    )
    if args.temp_dir is not None:
        environment.update(TMPDIR=str(args.temp_dir), TMP=str(args.temp_dir), TEMP=str(args.temp_dir))
    command = _command(args, model, task_type)
    print(
        json.dumps(
            {
                "model": str(model),
                "module": device["module"],
                "bus": device["bus"],
                "numa": device["numa"],
                "cpu_affinity": sorted(affinity),
                "command": command,
                "checkpoint_formats": formats,
                "flashgen_lora": str(args.flashgen_lora) if args.flashgen_lora else None,
                "fasth3_adapter": str(args.fasth3_adapter) if args.fasth3_adapter else None,
                "lightx2v_lora": str(args.lightx2v_lora) if args.lightx2v_lora else None,
                "online_fp8": bool(args.online_fp8),
                "phase_offload": bool(args.phase_offload),
                "vae_tile_batch_size": int(args.vae_tile_batch_size),
                "vae_persist_bf16_weights": bool(args.vae_persist_bf16_weights),
                "temp_dir": str(args.temp_dir) if args.temp_dir else None,
                "media_bin": str(args.media_bin),
                "huggingface_network_disabled": True,
            },
            indent=2,
        ),
        flush=True,
    )
    os.sched_setaffinity(0, affinity)
    process: subprocess.Popen[bytes] | None = None

    def forward(signum, _frame):
        if process is not None:
            os.killpg(process.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    try:
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        return process.wait()
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
