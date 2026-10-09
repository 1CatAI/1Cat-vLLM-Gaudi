# SPDX-License-Identifier: Apache-2.0
"""Frozen TP4 DSpark serving resources, artifact profile and owned card lease."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--recipe-cache-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--modules", default="2,6,7,3", help="Frozen rank-ordered modules for this comparison")
    parser.add_argument("--cpu-mask", default="0-9,20-21,28-37,54-55", help="Frozen CPU pool for this comparison")
    parser.add_argument("--secondary-lock-dir", type=Path, action="append", default=[],
                        help="Honor an additional existing card reservation namespace")
    parser.add_argument("--min-host-available-gib", type=float, default=350,
                        help="Host RAM admission before card locks; default 350 GiB (at least 350 GB)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide the serving command after --")
    workspace = Path(__file__).resolve().parents[1]
    # Match serving02's NUMA main/helper choices, leaving two spare physical
    # CPUs per socket for the launcher's availability check. The shared lease
    # still excludes CPUs owned by other model tasks.
    from run_deepseek_v41 import active_worker_cpus, cpuset

    cpus = cpuset(args.cpu_mask)
    os.sched_setaffinity(0, cpus)
    if args.min_host_available_gib < 0:
        parser.error("Host memory admission threshold must be nonnegative")
    admission = args.evidence.with_name(args.evidence.name + "-host-admission.json")
    modules = [int(value) for value in args.modules.split(',')]
    node_counts = {}
    for device in Path('/sys/class/accel').glob('accel[0-9]*'):
        if int((device / 'device/module_id').read_text()) in modules:
            node = int((device / 'device/numa_node').read_text())
            node_counts[node] = node_counts.get(node, 0) + 1
    if sum(node_counts.values()) != len(modules):
        parser.error("Requested modules are not present")
    while True:
        memory = {line.split(':', 1)[0]: int(line.split(':', 1)[1].split()[0])
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        full = next(line for line in Path('/proc/pressure/memory').read_text().splitlines()
                    if line.startswith('full '))
        pressure = float(next(value.split('=')[1] for value in full.split() if value.startswith('avg10=')))
        available = memory['MemAvailable'] / 1048576
        reserved = active_worker_cpus()
        free_cpus = {node: sorted(cpuset(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text())
                                 & cpus - reserved) for node in node_counts}
        cpu_ready = all(len(free_cpus[node]) >= count * 5 for node, count in node_counts.items())
        admitted = available >= args.min_host_available_gib and pressure <= 1 and cpu_ready
        admission.parent.mkdir(parents=True, exist_ok=True)
        admission.write_text(json.dumps(dict(launcher_pid=os.getpid(), available_gib=available,
                                             required_gib=args.min_host_available_gib,
                                             full_stall_percent_10s=pressure, admitted=admitted,
                                             free_worker_cpus=free_cpus, cpu_ready=cpu_ready,
                                             time_ns=time.time_ns(), owns_cards=False), indent=2) + '\n')
        if admitted:
            break
        print(f"Waiting for host RAM: {available:.1f}/{args.min_host_available_gib:.1f} GiB; "
              f"CPU pool ready={cpu_ready}; foreign tasks and card locks untouched", flush=True)
        time.sleep(30)
    invocation = [sys.executable, str(workspace / "tools/run_deepseek_v41.py"),
                  "--devices", "4", "--modules", args.modules, "--lock-dir", str(workspace.parent / "locks"),
                  "--min-host-available-gib", str(args.min_host_available_gib),
                  "--runtime-profile", str(args.runtime_profile.resolve()),
                  "--source-snapshot", str(args.source_snapshot.resolve()),
                  "--recipe-cache-dir", str(args.recipe_cache_dir.resolve()),
                  "--engine-source", str(workspace.parent / "builds/dsv41-tp4-dspark-main-v1/engine"),
                  "--raw-profiler", str(args.evidence.resolve()), "--", *command]
    # The evidence positional argument starts argparse's REMAINDER. All
    # resource options must precede it, as well as the command separator.
    invocation[2:2] = [item for directory in args.secondary_lock_dir
                       for item in ("--secondary-lock-dir", str(directory.resolve()))]
    raise SystemExit(subprocess.call(invocation, cwd=workspace))


if __name__ == "__main__":
    main()
