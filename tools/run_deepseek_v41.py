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


def cpuset(value):
    result = set()
    for field in value.strip().split(","):
        if not field:
            continue
        ends = [int(item) for item in field.split("-")]
        result.update(range(ends[0], ends[-1] + 1))
    return result


def active_worker_cpus():
    reserved = set()
    keys = (b"VLLM_HPU_DSV4_WORKER_CPUS=", b"VLLM_HPU_DSV4_WORKER_HELPER_CPUS=",
            b"VLLM_HPU_DSV41_ENGINE_CPUS=", b"VLLM_HPU_DSV41_API_CPUS=")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            entries = (proc / "environ").read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        for entry in entries:
            if entry.startswith(keys):
                reserved.update(cpuset(entry.split(b"=", 1)[1].decode().replace(";", ",")))
    for cpu in tuple(reserved):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        reserved.update(cpuset(path.read_text()))
    return reserved


def acquire(lock_dir, count):
    held, selected = [], []
    try:
        for candidate in sorted(Path("/sys/class/accel").glob("accel[0-9]*")):
            module = int((candidate / "device/module_id").read_text())
            local = []
            try:
                paths = set(lock_dir.glob(f"*module{module}.lock")) | {lock_dir / f"gaudi-module{module}.lock"}
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
    parser.add_argument("--seed-recipes", type=Path,
                        help="Copy private recipe files from a finished run with the identical runtime profile")
    parser.add_argument("--devices", type=int, choices=(1, 2, 4), default=4,
                        help="Four for normal serving; fewer only for bounded component diagnostics")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A normal model/check command is required after --")
    args.evidence.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.runtime_profile.read_text())
    env = dict(os.environ)
    for key in ("PYTHONPATH", "LD_PRELOAD", "HABANA_PROFILE", "VLLM_PLUGINS"):
        env.pop(key, None)
    env.update(profile["environment"])
    selected, locks = acquire(args.lock_dir, args.devices)
    while selected is None:
        print(f"Waiting for {args.devices} unowned Gaudi2 modules; existing jobs remain untouched", flush=True)
        time.sleep(30)
        selected, locks = acquire(args.lock_dir, args.devices)
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "command": command,
              "launcher_pid": os.getpid(), "modules": selected, "runtime_profile": str(args.runtime_profile.resolve())}
    try:
        allowed, reserved, mains, helpers = os.sched_getaffinity(0), active_worker_cpus(), [], []
        record["excluded_active_worker_cpus"] = sorted(reserved)
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
        control_cpus = set()
        if env.get("VLLM_HPU_DSV41_ISOLATE_CONTROL", "0") == "1":
            local = cpuset(Path(f"/sys/devices/system/node/node{selected[0]['numa']}/cpulist").read_text()) & allowed
            for role, count in (("engine", 2), ("api", 1)):
                group = []
                for cpu in sorted(local - reserved):
                    if cpu in reserved:
                        continue
                    group.append(cpu)
                    reserved.update(cpuset(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()))
                    if len(group) == count:
                        break
                if len(group) != count:
                    raise RuntimeError("Insufficient free NUMA-local CPUs for the V4.1 control processes")
                env[f"VLLM_HPU_DSV41_{role.upper()}_CPUS"] = ",".join(map(str, group))
                record.setdefault("control_cpus", {})[role] = group
                control_cpus.update(group)
        temporary = args.evidence / "tmp"
        temporary.mkdir(exist_ok=True)
        env.update(TMPDIR=str(temporary.resolve()),
                   HABANA_VISIBLE_MODULES=",".join(str(item["module"]) for item in selected),
                   HLS_MODULE_ID=str(selected[0]["module"]), HABANA_LOGS=str(args.evidence / "habana_logs"),
                   VLLM_HPU_DSV4_WORKER_CPUS=",".join(map(str, mains)),
                   VLLM_HPU_DSV4_WORKER_HELPER_CPUS=";".join(helpers),
                   VLLM_HPU_TP2_PLAN_DUMP_DIR=str(args.evidence / "plans"),
                   VLLM_TORCH_PROFILER_DIR=str(args.evidence / "traces"), DSV41_RUN_EVIDENCE=str(args.evidence))
        env["PT_HPU_RECIPE_CACHE_CONFIG"] = str(args.evidence / "recipes" / "rank{rank}") + ",false,8192,false"
        if args.devices == 1:
            env["PT_HPU_RECIPE_CACHE_CONFIG"] = env["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", "0")
        if env.get("GRAPH_VISUALIZATION") == "1":
            env["GRAPH_VISUALIZATION_DIR"] = str(args.evidence / "graphs")
            env["PT_HPU_GRAPH_DUMP_PREFIX"] = env["GRAPH_VISUALIZATION_DIR"]
            (args.evidence / "graphs").mkdir(exist_ok=True)
        record["environment"] = {key: value for key, value in env.items()
                                 if key.startswith(("HABANA_", "HLS_", "PT_HPU_", "VLLM_", "HCL_", "HCCL_"))
                                 or key in ("LD_LIBRARY_PATH", "GC_KERNEL_PATH", "RUNTIME_SCALE_PATCHING", "TMPDIR")}
        root = Path(__file__).resolve().parents[1]
        record["source_hashes"] = {}
        for glob in ("vllm_gaudi/**/*.py", "tools/*deepseek_v41*.py", "csrc/deepseek_v4/**/*",
                     "csrc/deepseek_v41/**/*", "tools/communication/*.h", "tools/communication/*.cpp",
                     "tests/standalone/deepseek_v41/*.py", "tests/unit_tests/ops/test_deepseek_v41*.py"):
            for source in root.glob(glob):
                if not source.is_file() or source.suffix in (".so", ".o", ".a", ".pyc") or "build" in source.parts:
                    continue
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
        if args.seed_recipes is not None:
            source = args.seed_recipes.resolve()
            previous = json.loads((source / "process.json").read_text())
            if "exit_code" not in previous:
                raise ValueError("Recipe seed must be a completed, immutable run")
            if (source / "runtime-profile.json").read_bytes() != args.runtime_profile.read_bytes():
                raise ValueError("Recipe seed requires the identical locked runtime profile")
            seed = {"source": str(source), "compiled_graphs": str(source / "graphs"), "files": {}}
            for path in sorted((source / "recipes").rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source / "recipes")
                destination = args.evidence / "recipes" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                with path.open("rb") as stream:
                    seed["files"][str(relative)] = hashlib.file_digest(stream, "sha256").hexdigest()
            (args.evidence / "recipe-seed-manifest.json").write_text(json.dumps(seed, indent=2) + "\n")
            record["recipe_seed"] = str(source)
        process = None
        launch_cpus = set(mains)
        launch_cpus.update(control_cpus)
        for item in selected:
            launch_cpus.update(item["helper_cpus"])
        record["initial_process_cpus"] = sorted(launch_cpus)
        os.sched_setaffinity(0, launch_cpus)
        def stop(signum, frame):
            del frame
            if process is not None:
                os.killpg(process.pid, signum)
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
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
        print((args.evidence / "run.log").read_text()[-14000:])
        return record["exit_code"]
    finally:
        for stream in locks:
            stream.close()


if __name__ == "__main__":
    sys.exit(main())
