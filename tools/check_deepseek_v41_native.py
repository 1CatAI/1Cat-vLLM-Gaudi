# SPDX-License-Identifier: Apache-2.0
"""Lease an available Gaudi2 and archive a native V4.1 contract check."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--lock-dir", required=True, type=Path)
    parser.add_argument("--runtime-libdirs", required=True)
    parser.add_argument("--select", default="not meta")
    parser.add_argument("--test-file", default="tests/standalone/deepseek_v41/test_native_moe.py")
    parser.add_argument("--graph-dump", action="store_true")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(Path("/sys/class/accel").glob("accel[0-9]*"))
    chosen, locks = None, []
    for candidate in candidates:
        module = int((candidate / "device/module_id").read_text())
        held = []
        try:
            # Honor all existing project lock families during migration to a
            # shared module lease. A stale file alone does not own a device.
            paths = set(args.lock_dir.glob(f"*module{module}.lock"))
            paths.add(args.lock_dir / f"gaudi-module{module}.lock")
            for path in sorted(paths):
                stream = path.open("a")
                held.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            device = "/dev/accel/" + candidate.name
            owner = subprocess.run(["fuser", device], capture_output=True, text=True)
            if owner.returncode != 1 or owner.stdout.strip() or owner.stderr.strip():
                continue
            bus = (candidate / "device").resolve().name
            status = subprocess.check_output(["hl-smi", "-i", bus,
                "--query-aip=memory.used,utilization.aip", "--format=csv,noheader,nounits"], text=True)
            memory, active = [float(part.strip()) for part in status.strip().split(",")]
            if memory > 1024 or active:
                continue
            chosen, locks = (candidate, module, bus), held
            held = []
            break
        except BlockingIOError:
            continue
        finally:
            for stream in held:
                stream.close()
    if chosen is None:
        raise RuntimeError("No unowned Gaudi2 module is available; no other workload was changed")
    candidate, module, bus = chosen
    numa = int((candidate / "device/numa_node").read_text())
    local_cpus = set()
    if numa >= 0:
        for part in Path(f"/sys/devices/system/node/node{numa}/cpulist").read_text().strip().split(","):
            endpoints = [int(value) for value in part.split("-")]
            local_cpus.update(range(endpoints[0], endpoints[-1] + 1))
    else:
        local_cpus = os.sched_getaffinity(0)
    cpus = sorted(local_cpus & os.sched_getaffinity(0))[:8]
    if not cpus:
        raise RuntimeError("No allowed CPU belongs to the selected device's NUMA node")
    env = dict(os.environ, HABANA_VISIBLE_MODULES=str(module), HLS_MODULE_ID=str(module), DSV41_TEST_HPU="1",
               PT_HPU_LAZY_MODE="0", PT_HPU_ENABLE_EAGER_CACHE="0", RUNTIME_SCALE_PATCHING="0",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="8",
               LD_LIBRARY_PATH=args.runtime_libdirs, HABANA_LOGS=str(args.evidence / "habana_logs"),
               GC_KERNEL_PATH=str(root / "vllm_gaudi/lib/libdeepseek_v4_gaudi2_kernels.so"))
    if args.graph_dump:
        (args.evidence / "graphs").mkdir()
        env.update(GRAPH_VISUALIZATION="1", GRAPH_VISUALIZATION_DIR=str(args.evidence / "graphs"),
                   PT_HPU_GRAPH_DUMP_PREFIX=str(args.evidence / "graphs"),
                   ENABLE_EXPERIMENTAL_FLAGS="true", SRAM_SLICER_GRAPH_VISUALIZATION="1")
    command = [sys.executable, "-m", "pytest", args.test_file,
               "--confcutdir=tests/standalone/deepseek_v41", "-v", "-s", "-x", "-k", args.select,
               "--junitxml=" + str(args.evidence / "results.xml")]
    runtime = {}
    for directory in args.runtime_libdirs.split(":"):
        for name in ("libSynapse.so", "libhcl.so", "libhabana_pytorch_backend.upstream.so",
                     "libhabana_pytorch2_plugin.upstream.so"):
            path = Path(directory) / name
            if path.is_file() and name not in runtime:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                runtime[name] = {"file": str(path.resolve()), "sha256": digest}
    record = {"command": command, "module": module, "bus": bus, "cpus": cpus, "numa": numa,
              "pid": os.getpid(), "pgid": os.getpgrp(), "lock_dir": str(args.lock_dir.resolve()),
              "environment": {name: env[name] for name in env if name.startswith(("HABANA_", "HLS_", "PT_HPU_"))
                               or name in ("LD_LIBRARY_PATH", "GC_KERNEL_PATH", "GRAPH_VISUALIZATION",
                                           "SRAM_SLICER_GRAPH_VISUALIZATION")},
              "runtime": runtime, "started_at": datetime.now(timezone.utc).isoformat()}
    (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.evidence / "native-build.json").write_bytes((root / "vllm_gaudi/lib/deepseek_v4_build.json").read_bytes())
    native_build = json.loads((args.evidence / "native-build.json").read_text())
    for name, expected in native_build["sources"].items():
        source = root / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Native source changed since the build: {name}")
        destination = args.evidence / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name, expected in native_build["binaries"].items():
        source = root / "vllm_gaudi/lib" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Native binary changed since the build: {name}")
        (args.evidence / "lib").mkdir(exist_ok=True)
        shutil.copy2(source, args.evidence / "lib" / name)
    python_sources = [*root.glob("vllm_gaudi/ops/deepseek_v41*.py"),
                      *root.glob("vllm_gaudi/models/deepseek_v41*.py"),
                      *root.glob("tests/standalone/deepseek_v41/*.py")]
    record["python_sources"] = {}
    for source in python_sources:
        name = str(source.relative_to(root))
        destination = args.evidence / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record["python_sources"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        print(f"Native V4.1 check on module {module} ({bus}); output: {args.evidence / 'run.log'}", flush=True)
        with (args.evidence / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        record["exit_code"] = result.returncode
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
        print((args.evidence / "run.log").read_text()[-10000:])
        raise SystemExit(result.returncode)
    finally:
        for stream in locks:
            stream.close()


if __name__ == "__main__":
    main()
