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


def record_device_release(selected, *, timeout_seconds=10):
    """Inspect owned modules after retirement, while their leases are held."""
    deadline = time.monotonic() + timeout_seconds
    samples = []
    while True:
        current = []
        for item in selected:
            try:
                raw = subprocess.check_output(
                    ["hl-smi", "-i", item["bus"], "--query-aip=memory.used,utilization.aip",
                     "--format=csv,noheader,nounits"], text=True, timeout=3)
                memory, utilization = map(float, raw.strip().split(","))
                current.append(dict(module=item["module"], memory_mib=memory,
                                    utilization=utilization))
            except (subprocess.SubprocessError, ValueError) as error:
                current.append(dict(module=item["module"], error=str(error)))
        samples.append(current)
        idle = all(row.get("memory_mib", float("inf")) <= 1024
                   and row.get("utilization", 1) == 0 for row in current)
        if idle or time.monotonic() >= deadline:
            return dict(healthy_cards_idle=idle, target_memory_mib=768,
                        admission_ceiling_mib=1024, samples=samples,
                        checked_before_unlock=True)
        time.sleep(0.25)


def validate_native_database_path(environment):
    directory = environment.get("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR")
    if not directory:
        return
    root = Path(directory).resolve()
    accepted = {str(root / "libdeepseek_v4_gaudi2_kernels.so"), "/usr/lib/habanalabs/libtpc_kernels.so"}
    if (root / "deepseek_v41_unique_build.json").is_file():
        accepted.add(str(root / "libdeepseek_v41_unique_kernels.so"))
    # Bind additive registration hashes before acquiring cards or weights.
    # Pinning the manifest file alone does not verify its internal binaries.
    manifest = root / "deepseek_v41_unique_build.json"
    if manifest.is_file():
        for name, expected in json.loads(manifest.read_text()).get("binaries", {}).items():
            path = root / name
            if Path(name).name != name or not path.is_file():
                raise ValueError(f"Invalid or missing additive registration: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Additive registration differs from its build manifest: {name}")
    configured = environment.get("GC_KERNEL_PATH")
    if configured and configured not in accepted:
        raise ValueError("GC_KERNEL_PATH must match the database in VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR; "
                         "update both paths when installing a candidate")


def cpuset(value):
    result = set()
    for field in value.strip().split(","):
        ends = [int(item) for item in field.split("-")]
        result.update(range(ends[0], ends[-1] + 1))
    return result


def active_worker_cpus():
    reserved = set()
    keys = (
        b"VLLM_HPU_DSV4_WORKER_CPUS=",
        b"VLLM_HPU_DSV4_WORKER_HELPER_CPUS=",
        b"VLLM_HPU_DSV41_ENGINE_CPUS=",
        b"VLLM_HPU_DSV41_API_CPUS=",
    )
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


def recipe_source_hashes(source_hashes):
    """Diagnostic/report scripts are archived but are not serving dependencies."""
    return {
        path: digest for path, digest in source_hashes.items() if path.startswith(("vllm_gaudi/", "flashinfer_gaudi/"))
    }


def configure_trace_artifacts(environment, evidence, *, dump_plans, enable_profiler, raw_profiler=False):
    """Generate symbols at compilation time, before any profiler acquisition."""
    if dump_plans:
        environment["VLLM_HPU_TP2_PLAN_DUMP_DIR"] = str(evidence / "plans")
        environment["GRAPH_VISUALIZATION"] = "1"
        environment["GRAPH_VISUALIZATION_DIR"] = str(evidence / "graphs")
    if enable_profiler or raw_profiler:
        if raw_profiler:
            environment["VLLM_HPU_DSV41_RAW_TRACE"] = "1"
            environment["VLLM_HPU_DSV41_RAW_SCOPE_ONLY"] = "1"
        environment["VLLM_TORCH_PROFILER_DIR"] = str(evidence / "traces")
        # Legacy symbol exports enable compiler debug instrumentation. Raw SDK
        # capture publishes its own post-graphs with that extra mode disabled.
        environment["ENABLE_PROFILER"] = "false" if raw_profiler else "true"
        # Kineto's HPU source rejects HABANA_PROFILE=0. Enable the SDK in
        # API-controlled mode before importing the bridge; start_disabled
        # keeps startup and speed requests outside the acquisition.
        environment["HABANA_PROFILE"] = "1"
        if not environment.get("HABANA_PROF_CONFIG"):
            config = {
                "Plugins": [
                    {
                        "enable": True,
                        "lib": "libhost_profiler.so",
                        "name": "HostProfiler",
                        "values": {
                            "api_group": {name: {"value": True} for name in ("HCCL", "HLTHUNK", "SYNAPSE")},
                            "start_disabled": {"value": True},
                        },
                    },
                    {
                        "enable": True,
                        "lib": "libhw_trace.so",
                        "name": "HwTrace",
                        "values": {
                            "generalOptions": {
                                "profilePhase": {"value": "profileApi"},
                                "traceBufferSize": {"value": "0x80000000"},
                            },
                            "parseOptions": {"addFuserMetadata": {"value": False}, "showNullDescs": {"value": False}},
                        },
                    },
                ]
            }
            if environment.get("VLLM_HPU_DSV41_RAW_TRACE") == "1":
                environment["HABANA_PROFILE_WRITE_HLTV"] = "1"
                config["GeneralSettings"] = {
                    "values": {
                        "addPid": {"value": True},
                        "outdir": {"value": str(evidence / "raw")},
                        "session": {"value": "v41_tp4"},
                    }
                }
                host = config["Plugins"][0]["values"]
                host["api_group"]["HLTHUNK"]["value"] = False
                host["api_group"]["SCAL"] = {"value": True}
                if environment.get("VLLM_HPU_DSV41_RAW_SCOPE_ONLY") == "1":
                    # Preserve collective API boundaries; the previous capture
                    # already contains full Synapse/SCAL and CPU operator logs.
                    host["api_group"]["SCAL"]["value"] = False
                    host["api_group"]["SYNAPSE"]["value"] = False
                host["performance_mode"] = {"value": True}
                host["output"] = {name: {"value": name == "hltv"} for name in ("hltv", "json", "csv")}
                hardware = config["Plugins"][1]["values"]
                hardware["generalOptions"].update(arch={"value": "gaudi2"}, traceBufferLocation={"value": "host"})
                hardware["parseOptions"].update(
                    skipParse={"value": True},
                    outputPerInvocation={
                        name: {"value": name in ("hltv", "hltvWithHost")}
                        for name in ("binary", "csv", "dbgInfo", "hltv", "hltvWithHost", "json", "text")
                    },
                )
            config_path = evidence / "profiler-config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            environment["HABANA_PROF_CONFIG"] = str(config_path)
    identity = {key: environment.get(key, "0") for key in ("ENABLE_PROFILER", "GRAPH_VISUALIZATION", "HABANA_PROFILE")}
    if enable_profiler or raw_profiler:
        config_identity = json.loads(Path(environment["HABANA_PROF_CONFIG"]).read_text())
        output = config_identity.get("GeneralSettings", {}).get("values", {}).get("outdir")
        if isinstance(output, dict) and isinstance(output.get("value"), str):
            # Capture destinations do not change compiled execution. Keep all
            # profiler policies in the key while allowing archived runs to
            # reuse recipes without overwriting each other's trace files.
            output["value"] = "<capture-output>"
        identity["profiler_config_sha256"] = hashlib.sha256(
            json.dumps(config_identity, sort_keys=True).encode()
        ).hexdigest()
    return identity


def acquire(lock_dir, count, requested_modules=None, secondary_lock_dirs=()):
    # Two selectors can repeatedly split a free four-card set into partial
    # reservations, release it, then collide again after the same retry delay.
    # Serialize only discovery/acquisition, never the lifetime of an active job.
    allocation_root = lock_dir.parent if lock_dir.name in ("locks", "evidence") else lock_dir
    with (allocation_root / "gaudi-device-allocation.lock").open("a") as allocation:
        try:
            fcntl.flock(allocation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None, []
        return _acquire_modules(lock_dir, count, requested_modules, secondary_lock_dirs)


def host_available_gib():
    memory = {line.split(':', 1)[0]: int(line.split(':', 1)[1].split()[0])
              for line in Path('/proc/meminfo').read_text().splitlines()}
    return memory['MemAvailable'] / 1048576


def ipc_scratch_path(requested, identity):
    """Leave room for the engine's UUID inside Linux's Unix-socket limit."""
    if len(os.fsencode(requested)) + 37 <= 107:
        return Path(requested)
    suffix = hashlib.sha256(os.fsencode(identity)).hexdigest()[:8]
    return Path('/opt/optane/dsv41-tmp') / suffix


def acquire_admitted(lock_dir, count, requested_modules, secondary_lock_dirs, minimum_gib, evidence):
    """Leave modules available while a model lacks its host-memory budget."""
    available = host_available_gib()
    selected, locks = None, []
    if available >= minimum_gib:
        selected, locks = acquire(lock_dir, count, requested_modules, secondary_lock_dirs)
        # A competing loader can consume RAM during device discovery. Return
        # its unused leases immediately, rather than wait while owning cards.
        if selected is not None:
            available = host_available_gib()
            if available < minimum_gib:
                for stream in locks:
                    stream.close()
                selected, locks = None, []
    (evidence / 'host-admission.json').write_text(json.dumps(
        dict(launcher_pid=os.getpid(), available_gib=available, required_gib=minimum_gib,
             ready=selected is not None, owns_cards=selected is not None, time_ns=time.time_ns()), indent=2) + '\n')
    return selected, locks


def _acquire_modules(lock_dir, count, requested_modules=None, secondary_lock_dirs=()):
    held, selected = [], []
    try:
        candidates = sorted(Path("/sys/class/accel").glob("accel[0-9]*"))
        if requested_modules is not None:
            requested = {int(module) for module in requested_modules}
            candidates = [
                candidate for candidate in candidates if int((candidate / "device/module_id").read_text()) in requested
            ]
            # Keep the caller's order so the PP/TP rank-to-module mapping is
            # stable across acquisitions.
            order = {int(module): index for index, module in enumerate(requested_modules)}
            candidates.sort(key=lambda candidate: order[int((candidate / "device/module_id").read_text())])
        for candidate in candidates:
            module = int((candidate / "device/module_id").read_text())
            local = []
            try:
                namespaces = {lock_dir, *secondary_lock_dirs}
                if lock_dir.name in ("locks", "evidence"):
                    namespaces.update(
                        path for name in ("locks", "evidence") if (path := lock_dir.parent / name).is_dir()
                    )
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
                status = subprocess.check_output(
                    ["hl-smi", "-i", bus, "--query-aip=memory.used,utilization.aip", "--format=csv,noheader,nounits"],
                    text=True,
                )
                memory, active = map(float, status.strip().split(","))
                if memory > 1024 or active:
                    continue
                selected.append(
                    {"module": module, "bus": bus, "numa": int((candidate / "device/numa_node").read_text())}
                )
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
    parser.add_argument(
        "--secondary-lock-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional shared module-lease namespaces used by other workers",
    )
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument(
        "--engine-source", type=Path, help="Use and fingerprint an isolated normal vLLM engine checkout"
    )
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Run an immutable archived Python source snapshot for a controlled comparison",
    )
    parser.add_argument(
        "--devices",
        type=int,
        choices=(1, 2, 4),
        default=4,
        help="Four for normal serving; fewer only for bounded component diagnostics",
    )
    parser.add_argument("--modules", type=str, help="Optional comma-separated physical module IDs, in rank order")
    parser.add_argument("--cpu-conflict-policy", choices=("wait", "relocate-or-measure"), default="wait",
                        help="Relocate this run within NUMA, or record CPU overlap when no cores are free")
    parser.add_argument("--preferred-cpus", type=str,
                        help="Prefer this CPU pool when available; retain the selected NUMA conflict policy")
    parser.add_argument("--min-host-available-gib", type=float, default=0,
                        help="Recheck available host RAM after acquiring card locks, before model startup")
    parser.add_argument(
        "--recipe-cache-dir",
        type=Path,
        help="Optional persistent cache root; source/runtime identities own separate namespaces",
    )
    parser.add_argument("--dump-plans", action="store_true", help="Save preparation graphs for an explicit diagnostic")
    profiler = parser.add_mutually_exclusive_group()
    profiler.add_argument(
        "--enable-profiler", action="store_true", help="Register profiler control with compiler debug instrumentation"
    )
    profiler.add_argument(
        "--raw-profiler",
        action="store_true",
        help="Register scope-only raw SDK capture with compiler profiling disabled",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.min_host_available_gib < 0:
        parser.error("Host memory admission threshold must be nonnegative")
    # The child runs inside the immutable source snapshot, so environment
    # and artifact paths must not depend on the caller's working directory.
    args.evidence = args.evidence.resolve()
    args.runtime_profile = args.runtime_profile.resolve()
    args.lock_dir = args.lock_dir.resolve()
    args.secondary_lock_dir = [path.resolve() for path in args.secondary_lock_dir]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A normal model/check command is required after --")
    args.evidence.mkdir(parents=True, exist_ok=False)
    # The launcher owns a per-run lock namespace.  Create it before probing
    # modules so a fresh temporary lease directory behaves like the existing
    # shared evidence lock directory.
    args.lock_dir.mkdir(parents=True, exist_ok=True)
    for directory in args.secondary_lock_dir:
        directory.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.runtime_profile.read_text())
    for item in profile.get("additional_libraries", []) + profile.get("configuration_files", []):
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"The pinned runtime library/configuration changed: {path}")
    env = dict(os.environ)
    for key in ("PYTHONPATH", "LD_PRELOAD", "HABANA_PROFILE", "VLLM_PLUGINS"):
        env.pop(key, None)
    env.update(profile["environment"])
    validate_native_database_path(env)
    auto_scratch = None
    if env.get('TMPDIR'):
        requested_scratch = Path(env['TMPDIR'])
        scratch = ipc_scratch_path(requested_scratch, args.evidence)
        scratch.mkdir(parents=True, exist_ok=True)
        if scratch != requested_scratch:
            auto_scratch = scratch
            env['TMPDIR'] = str(scratch)
            profile['environment']['TMPDIR'] = str(scratch)
        (args.evidence / 'ipc-scratch.json').write_text(json.dumps(
            dict(requested=str(requested_scratch), effective=str(scratch),
                 normalized=scratch != requested_scratch, socket_path_limit_bytes=107), indent=2) + '\n')
    if env.get("VLLM_HPU_DSV41_RAW_TRACE", "0") == "1" and not (args.enable_profiler or args.raw_profiler):
        parser.error("Raw trace capture requires --enable-profiler or --raw-profiler")
    requested_modules = None
    if args.modules:
        requested_modules = tuple(int(value) for value in args.modules.split(",") if value.strip())
        if len(requested_modules) != args.devices or len(set(requested_modules)) != len(requested_modules):
            raise RuntimeError("--modules must contain exactly --devices distinct module IDs")
    selected, locks = acquire_admitted(args.lock_dir, args.devices, requested_modules,
                                      args.secondary_lock_dir, args.min_host_available_gib, args.evidence)
    while selected is None:
        target = args.modules if args.modules else f"{args.devices} unowned Gaudi2 modules"
        print(f"Waiting for {target} and host-memory budget; no card leases held", flush=True)
        time.sleep(2 if args.cpu_conflict_policy == "relocate-or-measure" else 30)
        selected, locks = acquire_admitted(args.lock_dir, args.devices, requested_modules,
                                          args.secondary_lock_dir, args.min_host_available_gib, args.evidence)
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "launcher_pid": os.getpid(),
        "modules": selected,
        "runtime_profile": str(args.runtime_profile.resolve()),
    }
    try:
        allowed = os.sched_getaffinity(0)
        if args.cpu_conflict_policy == "relocate-or-measure":
            effective = Path("/sys/fs/cgroup/cpuset.cpus.effective")
            if effective.is_file() and effective.read_text().strip():
                permitted = cpuset(effective.read_text())
                if permitted:
                    os.sched_setaffinity(0, permitted)
                    allowed = permitted
        # Host admission precedes the card lease. A retiring worker can
        # briefly reappear in /proc between those two steps; wait within the
        # owned lease rather than fail and reload the whole serving model.
        # Archive the actual affinity and reservations so a repeated resource
        # failure is actionable without another model launch.
        node_counts = {}
        for item in selected:
            node_counts[item["numa"]] = node_counts.get(item["numa"], 0) + 1
        while True:
            busy = active_worker_cpus()
            free = {node: sorted(cpuset(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
                                 & allowed - busy) for node in node_counts}
            cpu_ready = all(len(free[node]) >= 5 * count for node, count in node_counts.items())
            memory = {line.split(':', 1)[0]: int(line.split(':', 1)[1].split()[0])
                      for line in Path('/proc/meminfo').read_text().splitlines()}
            available_gib = memory['MemAvailable'] / 1048576
            ready = ((cpu_ready or args.cpu_conflict_policy == "relocate-or-measure")
                     and available_gib >= args.min_host_available_gib)
            (args.evidence / "cpu-admission.json").write_text(json.dumps(
                dict(allowed=sorted(allowed), reserved=sorted(busy), nodes=node_counts,
                     free=free, ready=ready, cpu_ready=cpu_ready, available_gib=available_gib,
                     required_gib=args.min_host_available_gib, launcher_pid=os.getpid()), indent=2) + "\n")
            if ready:
                break
            print(f"Waiting after card acquisition: CPU pool ready={cpu_ready}; "
                  f"host RAM {available_gib:.1f}/{args.min_host_available_gib:.1f} GiB", flush=True)
            time.sleep(10)
        reserved, mains, helpers = active_worker_cpus(), [], []
        preferred = cpuset(args.preferred_cpus) if args.preferred_cpus else set()
        record["preferred_cpus"] = sorted(preferred)
        record["excluded_active_worker_cpus"] = sorted(reserved)
        for item in selected:
            available = cpuset(Path(f"/sys/devices/system/node/node{item['numa']}/cpulist").read_text()) & allowed
            physical = [cpu for cpu in sorted(available, key=lambda cpu: (cpu not in preferred, cpu))
                        if cpu not in reserved]
            # One main and four helpers consume five physical cores. Control
            # process isolation, when requested, has its own explicit check.
            overlap = False
            if len(physical) < 5:
                if args.cpu_conflict_policy != "relocate-or-measure":
                    raise RuntimeError("Insufficient free CPU affinity for the selected device NUMA node")
                assigned = set(mains) | {int(cpu) for group in helpers for cpu in group.split(",")}
                physical = sorted(available - assigned)[:5]
                if len(physical) < 5:
                    raise RuntimeError("NUMA node has fewer than five permitted CPU cores")
                overlap = True
                record.setdefault("cpu_overlap_fallback", []).append(dict(module=item["module"], cpus=physical))
            main_cpu = physical[0]
            siblings = cpuset(Path(f"/sys/devices/system/cpu/cpu{main_cpu}/topology/thread_siblings_list").read_text())
            reserved.update(siblings)
            helper = ([cpu for cpu in physical if cpu != main_cpu][:4]
                      if overlap
                      else [cpu for cpu in physical if cpu not in reserved][:4])
            for cpu in helper:
                reserved.update(
                    cpuset(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text())
                )
            mains.append(main_cpu)
            helpers.append(",".join(map(str, helper)))
            item.update(main_cpu=main_cpu, helper_cpus=helper)
        control_cpus = set()
        if env.get("VLLM_HPU_DSV41_ISOLATE_CONTROL", "0") == "1":
            local = cpuset(Path(f"/sys/devices/system/node/node{selected[0]['numa']}/cpulist").read_text()) & allowed
            for role, count in (("engine", 2), ("api", 1)):
                group = []
                for cpu in sorted(local - reserved):
                    group.append(cpu)
                    reserved.update(
                        cpuset(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text())
                    )
                    if len(group) == count:
                        break
                if len(group) != count:
                    if args.cpu_conflict_policy != "relocate-or-measure":
                        raise RuntimeError("Insufficient free NUMA-local CPUs for the V4.1 control processes")
                    worker_cpus = set(mains) | {int(cpu) for value in helpers for cpu in value.split(",")}
                    choices = sorted(local - worker_cpus) or sorted(local)
                    group = choices[:count]
                    if len(group) != count:
                        raise RuntimeError("Insufficient permitted NUMA-local control CPUs")
                    record.setdefault("cpu_overlap_fallback", []).append(dict(role=role, cpus=group))
                env[f"VLLM_HPU_DSV41_{role.upper()}_CPUS"] = ",".join(map(str, group))
                record.setdefault("control_cpus", {})[role] = group
                control_cpus.update(group)
        env.update(
            HABANA_VISIBLE_MODULES=",".join(str(item["module"]) for item in selected),
            HLS_MODULE_ID=str(selected[0]["module"]),
            HABANA_LOGS=str(args.evidence / "habana_logs"),
            VLLM_HPU_DSV4_WORKER_CPUS=",".join(map(str, mains)),
            VLLM_HPU_DSV4_WORKER_HELPER_CPUS=";".join(helpers),
            DSV41_RUN_EVIDENCE=str(args.evidence),
            DSV41_RUNTIME_PROFILE=str(args.evidence / "runtime-profile.json"),
        )
        env.pop("VLLM_HPU_TP2_PLAN_DUMP_DIR", None)
        env.pop("VLLM_TORCH_PROFILER_DIR", None)
        instrumentation = configure_trace_artifacts(
            env,
            args.evidence,
            dump_plans=args.dump_plans,
            enable_profiler=args.enable_profiler,
            raw_profiler=args.raw_profiler,
        )
        if env.get("GRAPH_VISUALIZATION") == "1":
            env["GRAPH_VISUALIZATION_DIR"] = str(args.evidence / "graphs")
        record["environment"] = {
            key: value
            for key, value in env.items()
            if key.startswith(("HABANA_", "HLS_", "PT_HPU_", "VLLM_", "HCL_", "HCCL_", "DSV41_"))
            or key
            in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "GC_KERNEL_PATH",
                "RUNTIME_SCALE_PATCHING",
                "ENABLE_PROFILER",
                "GRAPH_VISUALIZATION",
                "GRAPH_VISUALIZATION_DIR",
                "TMPDIR",
            )
        }
        root = Path(__file__).resolve().parents[1]
        source_root = args.source_snapshot.resolve() if args.source_snapshot else root
        if args.source_snapshot and not (source_root / "vllm_gaudi/__init__.py").is_file():
            raise RuntimeError(f"Archived source snapshot is incomplete: {source_root}")
        record["source_snapshot"] = str(source_root)
        record["source_hashes"] = {}
        for glob in (
            "vllm_gaudi/**/*.py",
            "vllm_gaudi/**/*.txt",
            "flashinfer_gaudi/**/*.py",
            "flashinfer_gaudi/**/*.json",
            "tools/*deepseek_v41*.py",
            "tests/standalone/deepseek_v41/test_engram_staging.py",
        ):
            for source in source_root.glob(glob):
                record["source_hashes"][str(source.relative_to(source_root))] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
                destination = args.evidence / "source" / source.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        # The compiler reads source lines while preparing new shapes. Execute
        # the archived package so concurrent edits cannot change those lines
        # or mix newly imported modules into an already loaded generation.
        execution_root = (args.evidence / "source").resolve()
        paths = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item and Path(item).resolve() != root]
        if args.engine_source:
            engine_source = args.engine_source.resolve()
            if not (engine_source / "vllm/__init__.py").is_file():
                raise RuntimeError("The selected engine checkout is incomplete")
            paths.insert(0, str(engine_source))
        env["PYTHONPATH"] = os.pathsep.join((str(execution_root), *paths))
        record["execution_source_root"] = str(execution_root)
        record["environment"]["PYTHONPATH"] = env["PYTHONPATH"]
        # Serving re-execs against this file. Persist the normalized scratch
        # path as well as exporting it, or the old overlong value returns.
        runtime_bytes = json.dumps(profile, indent=2).encode() + b'\n'
        (args.evidence / "runtime-profile.json").write_bytes(runtime_bytes)
        record['runtime_profile_source_sha256'] = hashlib.sha256(args.runtime_profile.read_bytes()).hexdigest()
        record['effective_runtime_profile_sha256'] = hashlib.sha256(runtime_bytes).hexdigest()
        if args.source_snapshot:
            archive_patch = source_root.parent / "source.patch"
            (args.evidence / "source.patch").write_bytes(archive_patch.read_bytes() if archive_patch.is_file() else b"")
        else:
            (args.evidence / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=root))
        import importlib.util

        engine = (
            args.engine_source.resolve()
            if args.engine_source
            else Path(importlib.util.find_spec("vllm").origin).resolve().parents[1]
        )
        record["engine_source_root"] = str(engine)
        engine_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=engine, text=True, capture_output=True)
        record["engine_commit"] = engine_revision.stdout.strip() if engine_revision.returncode == 0 else None
        engine_patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=engine) if record["engine_commit"] else b""
        # Installed wheels and frozen source snapshots need the same cache
        # identity protection as a Git checkout. Include untracked Python
        # and native engine modules even when a commit is available.
        engine_sources = {}
        for pattern in ("**/*.py", "**/*.so"):
            for source in sorted((engine / "vllm").glob(pattern)):
                with source.open("rb") as stream:
                    engine_sources[str(source.relative_to(engine))] = hashlib.file_digest(stream, "sha256").hexdigest()
        engine_identity = json.dumps(engine_sources, sort_keys=True).encode()
        (args.evidence / "engine-sources.json").write_bytes(engine_identity)
        record["engine_sources_sha256"] = hashlib.sha256(engine_identity).hexdigest()
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
            cache_identity = {
                "schema": 4,
                "source": recipe_source_hashes(record["source_hashes"]),
                "engine": record["engine_commit"],
                "engine_patch": record["engine_patch_sha256"],
                "engine_sources": record["engine_sources_sha256"],
                "runtime": profile,
                "instrumentation": instrumentation,
                "command": command,
                "model_manifests": model_manifests,
                "native_build": hashlib.sha256(native_build.read_bytes()).hexdigest(),
            }
            fingerprint = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
            cache = args.recipe_cache_dir.resolve() / fingerprint
            cache.mkdir(parents=True, exist_ok=True)
            (args.evidence / "recipe_cache").symlink_to(cache, target_is_directory=True)
            (args.evidence / "recipes").symlink_to(cache, target_is_directory=True)
            cache_rank = "rank0" if args.devices == 1 else "rank{rank}"
            env["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{cache / cache_rank},false,8192,false"
            record["environment"]["PT_HPU_RECIPE_CACHE_CONFIG"] = env["PT_HPU_RECIPE_CACHE_CONFIG"]
            record["recipe_cache_identity"] = fingerprint
            record["recipe_cache_identity_schema"] = 4
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
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with (args.evidence / "run.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=execution_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record.update(pid=process.pid, pgid=process.pid)
            (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
            print(
                f"V4.1 process {process.pid}, modules {env['HABANA_VISIBLE_MODULES']}; {args.evidence / 'run.log'}",
                flush=True,
            )
            record["exit_code"] = process.wait()
        record["parent_exit_code"] = record["exit_code"]
        record["process_group_retirement"] = retire_process_group(process.pid)
        if record["process_group_retirement"]["forced_cleanup"] and record["exit_code"] == 0:
            record["exit_code"] = 1
        record["device_release"] = record_device_release(selected)
        while not record["device_release"]["healthy_cards_idle"]:
            # Large native plans can retire their driver allocation after the
            # owned process group has already disappeared. Keep the card lease
            # across that delay; an observation timeout is not permission to
            # unlock a still-allocated module or reset a shared device.
            (args.evidence / "device-release.json").write_text(
                json.dumps(record["device_release"], indent=2) + "\n")
            print("Owned workers retired; retaining leases until device memory is idle", flush=True)
            record["device_release"] = record_device_release(selected)
        (args.evidence / "device-release.json").write_text(
            json.dumps(record["device_release"], indent=2) + "\n")
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.evidence / "process.json").write_text(json.dumps(record, indent=2) + "\n")
        print((args.evidence / "run.log").read_text()[-14000:])
        return record["exit_code"]
    finally:
        for stream in locks:
            stream.close()
        if auto_scratch is not None:
            shutil.rmtree(auto_scratch)


if __name__ == "__main__":
    sys.exit(main())
