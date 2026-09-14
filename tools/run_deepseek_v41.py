# SPDX-License-Identifier: Apache-2.0
"""Lease four idle Gaudi2 modules and archive one prepared V4.1 invocation."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def retire_process_group(pgid, *, grace_seconds=10, terminate_seconds=10):
    """Retain leases until the exited launcher's owned process group is gone."""
    import psutil

    def members():
        result = []
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if process.info["status"] != psutil.STATUS_ZOMBIE and os.getpgid(process.pid) == pgid:
                    result.append(process.pid)
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue
        return result

    started = time.monotonic()
    record = {"pgid": pgid, "initial_survivors": members(), "signals": []}
    for delay, signum in ((grace_seconds, signal.SIGTERM), (terminate_seconds, signal.SIGKILL)):
        deadline = time.monotonic() + delay
        while members() and time.monotonic() < deadline:
            time.sleep(0.1)
        live = members()
        if not live:
            break
        record["signals"].append({"signal": signum.name, "pids": live})
        for pid in live:
            try:
                # Recheck ownership immediately before signalling a survivor.
                if os.getpgid(pid) == pgid:
                    os.kill(pid, signum)
            except ProcessLookupError:
                pass
    # A driver may take time to retire a killed device context. Do not release
    # the module leases while a live process in this owned group remains.
    while members():
        time.sleep(0.1)
    record["elapsed_ms"] = (time.monotonic() - started) * 1000
    record["forced_cleanup"] = bool(record["signals"])
    return record


def cpuset(value):
    result = set()
    for field in value.strip().split(","):
        ends = [int(item) for item in field.split("-")]
        result.update(range(ends[0], ends[-1] + 1))
    return result


def recipe_source_hashes(source_hashes):
    """Diagnostic/report scripts are archived but are not serving dependencies."""
    return {path: digest for path, digest in source_hashes.items() if path.startswith("vllm_gaudi/")}


def acquire(lock_dir, count, requested_modules=None):
    held, selected = [], []
    try:
        candidates = sorted(Path("/sys/class/accel").glob("accel[0-9]*"))
        if requested_modules is not None:
            requested = {int(module) for module in requested_modules}
            candidates = [candidate for candidate in candidates
                          if int((candidate / "device/module_id").read_text()) in requested]
            # Keep the caller's order so the PP/TP rank-to-module mapping is
            # stable across acquisitions.
            order = {int(module): index for index, module in enumerate(requested_modules)}
            candidates.sort(key=lambda candidate: order[int((candidate / "device/module_id").read_text())])
        for candidate in candidates:
            module = int((candidate / "device/module_id").read_text())
            local = []
            try:
                namespaces = {lock_dir}
                if lock_dir.name in ("locks", "evidence"):
                    namespaces.update(path for name in ("locks", "evidence")
                                      if (path := lock_dir.parent / name).is_dir())
                paths = {path for directory in namespaces for path in directory.glob(f"*module{module}.lock")}
                paths.update(directory / f"gaudi-module{module}.lock" for directory in namespaces)
                for path in sorted(paths):
                    stream = path.open("a")
                    local.append(stream)
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owner = subprocess.run(["fuser", "/dev/accel/" + candidate.name], text=True, capture_output=True)
                if owner.returncode != 1 or owner.stdout.strip() or owner.stderr.strip():
                    continue
                bus = (candidate / "device").resolve().name
                status = subprocess.check_output(["hl-smi", "-i", bus, "--query-aip=memory.used,utilization.aip",
                                                  "--format=csv,noheader,nounits"], text=True)
                memory, active = map(float, status.strip().split(","))
                if memory > 1024 or active:
                    continue
                selected.append({"module": module, "bus": bus,
                                 "numa": int((candidate / "device/numa_node").read_text())})
                held.extend(local)
                local = []
                if len(selected) == count:
                    return selected, held
            except BlockingIOError:
                continue
            finally:
                for stream in local:
                    stream.close()
        return None, []
    finally:
        if len(selected) != count:
            for stream in held:
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--lock-dir", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--devices", type=int, choices=(1, 2, 4), default=4,
                        help="Four for normal serving; fewer only for bounded component diagnostics")
    parser.add_argument("--modules", type=str,
                        help="Optional comma-separated physical module IDs, in rank order")
    parser.add_argument("--recipe-cache-dir", type=Path,
                        help="Optional persistent cache root; source/runtime identities own separate namespaces")
    parser.add_argument("--dump-plans", action="store_true", help="Save preparation graphs for an explicit diagnostic")
    parser.add_argument("--enable-profiler", action="store_true", help="Register profiler control for an explicit trace")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A normal model/check command is required after --")
    args.evidence.mkdir(parents=True, exist_ok=False)
    # The launcher owns a per-run lock namespace.  Create it before probing
    # modules so a fresh temporary lease directory behaves like the existing
    # shared evidence lock directory.
    args.lock_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.runtime_profile.read_text())
    for item in profile.get("additional_libraries", []) + profile.get("configuration_files", []):
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"The pinned runtime library/configuration changed: {path}")
    env = dict(os.environ)
    for key in ("PYTHONPATH", "LD_PRELOAD", "HABANA_PROFILE", "VLLM_PLUGINS"):
        env.pop(key, None)
    env.update(profile["environment"])
    requested_modules = None
    if args.modules:
        requested_modules = tuple(int(value) for value in args.modules.split(",") if value.strip())
        if len(requested_modules) != args.devices or len(set(requested_modules)) != len(requested_modules):
            raise RuntimeError("--modules must contain exactly --devices distinct module IDs")
    selected, locks = acquire(args.lock_dir, args.devices, requested_modules)
    while selected is None:
        target = args.modules if args.modules else f"{args.devices} unowned Gaudi2 modules"
        print(f"Waiting for {target}; existing jobs remain untouched", flush=True)
        time.sleep(30)
        selected, locks = acquire(args.lock_dir, args.devices, requested_modules)
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "command": command,
              "launcher_pid": os.getpid(), "modules": selected, "runtime_profile": str(args.runtime_profile.resolve())}
    try:
        allowed, reserved, mains, helpers = os.sched_getaffinity(0), set(), [], []
        for item in selected:
            available = cpuset(Path(f"/sys/devices/system/node/node{item['numa']}/cpulist").read_text()) & allowed
            physical = [cpu for cpu in sorted(available) if cpu not in reserved]
            if len(physical) < 6:
                raise RuntimeError("Insufficient free CPU affinity for the selected device NUMA node")
            main_cpu = physical[0]
            siblings = cpuset(Path(f"/sys/devices/system/cpu/cpu{main_cpu}/topology/thread_siblings_list").read_text())
            reserved.update(siblings)
            helper = [cpu for cpu in physical if cpu not in reserved][:4]
            for cpu in helper:
                reserved.update(cpuset(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()))
            mains.append(main_cpu)
            helpers.append(",".join(map(str, helper)))
            item.update(main_cpu=main_cpu, helper_cpus=helper)
        env.update(HABANA_VISIBLE_MODULES=",".join(str(item["module"]) for item in selected),
                   HLS_MODULE_ID=str(selected[0]["module"]), HABANA_LOGS=str(args.evidence / "habana_logs"),
                   VLLM_HPU_DSV4_WORKER_CPUS=",".join(map(str, mains)),
                   VLLM_HPU_DSV4_WORKER_HELPER_CPUS=";".join(helpers),
                   DSV41_RUN_EVIDENCE=str(args.evidence),
                   DSV41_RUNTIME_PROFILE=str(args.evidence / "runtime-profile.json"))
        env.pop("VLLM_HPU_TP2_PLAN_DUMP_DIR", None)
        env.pop("VLLM_TORCH_PROFILER_DIR", None)
        if args.dump_plans:
            env["VLLM_HPU_TP2_PLAN_DUMP_DIR"] = str(args.evidence / "plans")
        if args.enable_profiler:
            env["VLLM_TORCH_PROFILER_DIR"] = str(args.evidence / "traces")
        if env.get("GRAPH_VISUALIZATION") == "1":
            env["GRAPH_VISUALIZATION_DIR"] = str(args.evidence / "graphs")
        record["environment"] = {key: value for key, value in env.items()
                                 if key.startswith(("HABANA_", "HLS_", "PT_HPU_", "VLLM_", "HCL_", "HCCL_", "DSV41_"))
                                 or key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "GC_KERNEL_PATH",
                                            "RUNTIME_SCALE_PATCHING")}
        root = Path(__file__).resolve().parents[1]
        record["source_hashes"] = {}
        for glob in ("vllm_gaudi/**/*.py", "tools/*deepseek_v41*.py"):
            for source in root.glob(glob):
                record["source_hashes"][str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
                destination = args.evidence / "source" / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        (args.evidence / "runtime-profile.json").write_bytes(args.runtime_profile.read_bytes())
        (args.evidence / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=root))
        import importlib.util
        engine = Path(importlib.util.find_spec("vllm").origin).resolve().parents[1]
        record["engine_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
        engine_patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=engine)
        (args.evidence / "engine.patch").write_bytes(engine_patch)
        record["engine_patch_sha256"] = hashlib.sha256(engine_patch).hexdigest()
        if args.recipe_cache_dir is not None:
            model_manifests = {}
            for argument in command:
                candidate = Path(argument) / "manifest.json"
                if candidate.is_file():
                    model_manifests[str(candidate.resolve())] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            native_dir = Path(env.get("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR", root / "vllm_gaudi/lib"))
            native_build = native_dir / "deepseek_v4_build.json"
            cache_identity = {"schema": 2, "source": recipe_source_hashes(record["source_hashes"]),
                              "engine": record["engine_commit"],
                              "engine_patch": record["engine_patch_sha256"], "runtime": profile,
                              "command": command, "model_manifests": model_manifests,
                              "native_build": hashlib.sha256(native_build.read_bytes()).hexdigest()}
            fingerprint = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
            cache = args.recipe_cache_dir.resolve() / fingerprint
            cache.mkdir(parents=True, exist_ok=True)
            (args.evidence / "recipe_cache").symlink_to(cache, target_is_directory=True)
            (args.evidence / "recipes").symlink_to(cache, target_is_directory=True)
            cache_rank = "rank0" if args.devices == 1 else "rank{rank}"
            env["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{cache / cache_rank},false,8192,false"
            record["environment"]["PT_HPU_RECIPE_CACHE_CONFIG"] = env["PT_HPU_RECIPE_CACHE_CONFIG"]
            record["recipe_cache_identity"] = fingerprint
            record["recipe_cache_identity_schema"] = 2
        process = None
        launch_cpus = set(mains)
        for item in selected:
            launch_cpus.update(item["helper_cpus"])
        record["initial_process_cpus"] = sorted(launch_cpus)
        os.sched_setaffinity(0, launch_cpus)
        def stop(signum, frame):
            del frame
            if process is not None:
                process.send_signal(signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with (args.evidence / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record.update(pid=process.pid, pgid=process.pid)
            (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"V4.1 process {process.pid}, modules {env['HABANA_VISIBLE_MODULES']}; {args.evidence / 'run.log'}",
                  flush=True)
            record["exit_code"] = process.wait()
        record["parent_exit_code"] = record["exit_code"]
        record["process_group_retirement"] = retire_process_group(process.pid)
        if record["process_group_retirement"]["forced_cleanup"] and record["exit_code"] == 0:
            record["exit_code"] = 1
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
        print((args.evidence / "run.log").read_text()[-14000:])
        return record["exit_code"]
    finally:
        for stream in locks:
            stream.close()


if __name__ == "__main__":
    sys.exit(main())
