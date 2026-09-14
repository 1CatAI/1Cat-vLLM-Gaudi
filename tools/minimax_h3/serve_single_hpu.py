#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease one idle Gaudi module and serve a local H3 native-FP8 partition."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import IO, Any


def _parse_cpuset(value: str) -> set[int]:
    result: set[int] = set()
    for field in value.strip().split(","):
        if not field:
            continue
        bounds = [int(item) for item in field.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


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
    for component in ("transformer", "text_encoder"):
        config_path = model / component / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization_config") or {}
        if (quantization.get("quant_method") != "modelopt"
                or quantization.get("quant_algo") != "FP8_PER_CHANNEL_PER_TOKEN"):
            raise ValueError(f"{config_path} is not native FP8_PER_CHANNEL_PER_TOKEN")
    return model, inferred.lower()


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
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
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
    )
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
