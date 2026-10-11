# SPDX-License-Identifier: Apache-2.0
"""Start an installed V4.1 service and maintain its machine CPU allocation."""
import argparse
import errno
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request


def atomic_json(path, value, *, required=True):
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    except OSError as error:
        if required or error.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        temporary.unlink(missing_ok=True)
        return False
    return True


def cpu_set(values):
    return set(map(int, values))


def physical_cores(cpus):
    result = set(cpus)
    for cpu in cpus:
        for field in Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().split(","):
            bounds = field.strip().split("-")
            result.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def isolate_desktop(reserved):
    """Keep same-user background threads off service cores, preserving original masks."""
    protected = physical_cores(reserved)
    available = set(range(os.cpu_count())) - protected
    if not available:
        raise ValueError("No CPUs remain for other user processes")
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1:
        ancestors.add(parent)
        fields = Path(f"/proc/{parent}/stat").read_text().rsplit(") ", 1)[1].split()
        parent = int(fields[1])
    records = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal() or int(process.name) in ancestors:
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            identity = process.joinpath("stat").read_text().rsplit(") ", 1)[1].split()[19]
            for task in process.joinpath("task").iterdir():
                tid = int(task.name)
                original = os.sched_getaffinity(tid)
                if original & protected:
                    selected = original & available or available
                    os.sched_setaffinity(tid, selected)
                    records.append(
                        dict(pid=int(process.name),
                             tid=tid,
                             start_ticks=identity,
                             original=sorted(original),
                             selected=sorted(selected)))
        except (OSError, ValueError):
            continue
    return records


def service_processes(pid):
    table = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            fields = process.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
            name = process.joinpath("cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="replace")
            table[int(process.name)] = (int(fields[1]), name)
        except OSError:
            continue
    owned = {pid}
    while True:
        expanded = owned | {p for p, (parent, _) in table.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    return {p: table[p][1] for p in owned if p in table}


def maintain_affinity(pid, settings):
    allocation = settings["cpu_allocation"]
    processes = service_processes(pid)
    for process, name in processes.items():
        rank = re.search(r"Worker_TP(\d+)", name)
        if rank:
            index = int(rank[1])
            main = {allocation["worker_main"][index]}
            helpers = cpu_set(allocation["worker_helpers"][index])
        elif "EngineCore" in name:
            main = cpu_set(allocation["engine_main"])
            helpers = cpu_set(allocation["control_helpers"])
        elif process == pid or "APIServer" in name:
            main = cpu_set(allocation["api_main"])
            helpers = cpu_set(allocation["control_helpers"])
        else:
            continue
        try:
            for task in Path(f"/proc/{process}/task").iterdir():
                tid = int(task.name)
                desired = main if tid == process else helpers
                if os.sched_getaffinity(tid) != desired:
                    os.sched_setaffinity(tid, desired)
        except (ProcessLookupError, FileNotFoundError):
            continue
    return processes


def validate_raw_trace_profile(profile):
    environment = profile.get("environment", {})
    for key in ("DUMP_PRE_GRAPHS", "DUMP_POST_GRAPHS"):
        if environment.get(key) == "0":
            raise ValueError(f"{key}=0 is a dump directory, not disabled; remove the variable")
    if environment.get("VLLM_HPU_DSV41_RAW_TRACE", "0") != "1":
        return
    if not environment.get("HABANA_PROF_CONFIG") or environment.get("HABANA_PROFILE_WRITE_HLTV") != "1":
        raise ValueError("Raw trace requires HABANA_PROF_CONFIG and HABANA_PROFILE_WRITE_HLTV=1 before loading")
    config = json.loads(Path(environment["HABANA_PROF_CONFIG"]).read_text())
    hardware = [p for p in config.get("Plugins", ()) if p.get("enable") and p.get("name") == "HwTrace"]
    if len(hardware) != 1 or not hardware[0]["values"]["parseOptions"]["skipParse"]["value"]:
        raise ValueError("Raw trace requires HwTrace skipParse=true before loading")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18552)
    parser.add_argument("--log-root", type=Path, help="Machine-local directory for service logs, such as SSD storage")
    parser.add_argument("--settings", type=Path, help="Override the installation's machine-local settings")
    args, extra = parser.parse_known_args()
    root = args.installation.resolve()
    settings_path = args.settings.expanduser().resolve() if args.settings is not None else root / "settings.json"
    settings = json.loads(settings_path.read_text())
    runtime_path = settings.get("runtime_profile", str(root / "runtime.json"))
    if "--runtime-profile" in extra:
        runtime_path = extra[extra.index("--runtime-profile") + 1]
    runtime_path = Path(runtime_path).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = root / runtime_path
    if runtime_path.is_file():
        validate_raw_trace_profile(json.loads(runtime_path.read_text()))
    compiler_temp = settings.get("environment", {}).get("TMPDIR")
    if compiler_temp:
        # A configured tmpfs scratch directory must be recreated after reboot.
        temporary_path = Path(compiler_temp).expanduser()
        compiler_temp = str((temporary_path if temporary_path.is_absolute() else root / temporary_path).resolve())
        from run_deepseek_v41 import ipc_scratch_path

        compiler_temp = str(ipc_scratch_path(compiler_temp, root))
        Path(compiler_temp).mkdir(mode=0o700, parents=True, exist_ok=True)
    allocation = settings["cpu_allocation"]
    reserved = set(allocation["worker_main"] + allocation["engine_main"] + allocation["api_main"] +
                   allocation["control_helpers"])
    for group in allocation["worker_helpers"]:
        reserved.update(group)
    if not reserved <= set(settings["cpus"]):
        raise ValueError("Machine allocation is outside the installation CPU set")
    log_root = args.log_root.expanduser().resolve() if args.log_root is not None else root / "logs"
    log_dir = log_root / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True)
    isolated = isolate_desktop(reserved) if settings.get("isolate_user_processes", False) else []
    atomic_json(log_dir / "background-affinity.json", isolated)
    environment = dict(os.environ)
    if compiler_temp:
        environment["TMPDIR"] = compiler_temp
    for key in list(environment):
        if key.startswith(("VLLM_HPU_DSV", "VLLM_HPU_TP2", "DSV41_",
                           "DUMP_")) or key in ("PYTHONPATH", "LD_PRELOAD", "GRAPH_VISUALIZATION",
                                                "GRAPH_VISUALIZATION_DIR", "PT_HPU_GRAPH_DUMP_PREFIX"):
            environment.pop(key)
    environment["VLLM_ENGINE_READY_TIMEOUT_S"] = "3600"
    environment["PYTHONUNBUFFERED"] = "1"
    # This launcher owns one machine's devices. CPU control collectives must
    # not advertise a transient routable address for local worker peers.
    environment["GLOO_SOCKET_IFNAME"] = settings.get("environment", {}).get("GLOO_SOCKET_IFNAME", "lo")
    if settings.get("api_key_file"):
        environment["VLLM_API_KEY"] = Path(settings["api_key_file"]).read_text().strip()
    command = [
        str(root / "venv/bin/python"), "-m", "vllm_gaudi.entrypoints.deepseek_v41", "--settings",
        str(settings_path), "--host", args.host, "--port",
        str(args.port), *extra
    ]
    with (log_dir / "service.log").open("w") as stream:
        child = subprocess.Popen(command,
                                 cwd=root,
                                 env=environment,
                                 stdout=stream,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record = dict(pid=child.pid,
                      pgid=child.pid,
                      command=command,
                      installation=str(root),
                      started_at=time.time(),
                      cpu_allocation=allocation)
        atomic_json(log_dir / "process.json", record)
        atomic_json(root / "service.json", dict(**record, log_dir=str(log_dir)))

        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        ready = False
        status_write_failed = False
        while child.poll() is None:
            if not ready:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                        ready = response.status == 200
                except OSError:
                    pass
                if ready:
                    record["ready_at"] = time.time()
                    from report_deepseek_v41_startup import summarize

                    summary = summarize(record, (log_dir / "service.log").read_text(errors="replace"))
                    atomic_json(log_dir / "startup-summary.json", summary, required=False)
                    print("Startup cache summary: " + json.dumps(summary["cache_by_rank"]), flush=True)
                    print("Ready; maintaining main/helper CPU isolation", flush=True)
            if ready:
                record["processes"] = maintain_affinity(child.pid, settings)
            status_written = atomic_json(log_dir / "process.json", record, required=False)
            if not status_written and not status_write_failed:
                print("Status metadata storage is full; keeping inference running and retrying", flush=True)
            elif status_written and status_write_failed:
                print("Status metadata writes recovered", flush=True)
            status_write_failed = not status_written
            time.sleep(1)
        record.update(exit_code=child.returncode, finished_at=time.time())
        atomic_json(log_dir / "process.json", record, required=False)
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
