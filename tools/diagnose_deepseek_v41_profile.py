# SPDX-License-Identifier: Apache-2.0
"""Attach to owned PP0 workers before a first-image profiling diagnostic."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("request_output", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18161")
    args = parser.parse_args()
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            if requests.get(args.url + "/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    else:
        raise RuntimeError("Diagnostic service did not become ready")
    owner = json.loads((args.run / "process.json").read_text())
    jobs, logs, record = [], [], {"scope": "GDB diagnostic; no performance qualification", "workers": []}
    try:
        for rank in (0, 1):
            pids = set(re.findall(rf"Worker_PP0_TP{rank} pid=(\d+)", (args.run / "run.log").read_text()))
            if len(pids) != 1:
                raise RuntimeError("Ambiguous PP0 diagnostic process")
            pid = int(pids.pop())
            if os.getpgid(pid) != owner["pgid"]:
                raise RuntimeError("Diagnostic worker ownership changed")
            ready = args.run / f"gdb-rank{rank}.ready"
            command = args.run / f"gdb-rank{rank}.commands"
            command.write_text("\n".join((
                "set pagination off",
                "set confirm off",
                "set print thread-events off",
                "handle SIGPIPE nostop noprint pass",
                "handle SIGSEGV stop print pass",
                f"attach {pid}",
                "python",
                f"open({str(ready)!r}, 'w').write('attached')",
                "end",
                "continue",
                "bt 40",
                "info registers",
                "x/32i $pc-64",
                "python",
                "import gdb",
                "frame = gdb.newest_frame()",
                "if 'addToDebugInfoMap' in (frame.name() or ''):",
                "    for cmd in ['x/12gx $rbx', 'x/20gx $r12', "
                "'set $dsv41_owner = *(void **)($rbp-0x510)', 'x/8gx (char *)$dsv41_owner+0x1318']:",
                "        try: gdb.execute(cmd)",
                "        except gdb.error as exc: print(exc)",
                "end",
                "thread apply all bt 7",
                "detach",
                "quit",
            )) + "\n")
            log = (args.run / f"gdb-rank{rank}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(["gdb", "--batch", "-nx", "-x", str(command)],
                                       stdout=log,
                                       stderr=subprocess.STDOUT)
            jobs.append(process)
            record["workers"].append({"rank": rank, "pid": pid, "gdb_pid": process.pid})
        (args.run / "debugger-ownership.json").write_text(json.dumps(record, indent=2) + "\n")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if all((args.run / f"gdb-rank{rank}.ready").is_file() for rank in (0, 1)):
                break
            if any(job.poll() is not None for job in jobs):
                raise RuntimeError("Debugger failed before attachment; see its recorded log")
            time.sleep(0.2)
        else:
            raise RuntimeError("Debugger attachment timed out")
        print("Both owned PP0 workers attached; starting first-image diagnostic", flush=True)
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("request_deepseek_v41.py")),
            str(args.request_output), "--url", args.url, "--image",
            str(args.image), "--prompt", "Describe this image.", "--tokens", "16", "--profile", "--server-process",
            str(args.run / "process.json")
        ])
        record["request_exit_code"] = result.returncode
    finally:
        # A successful request leaves GDB in continue. Interrupt the debugger
        # to record its idle stack and detach; never signal another workload.
        for job in jobs:
            if job.poll() is None:
                record.setdefault("debugger_interrupts", []).append(job.pid)
                job.send_signal(signal.SIGINT)
        for job in jobs:
            try:
                job.wait(timeout=60)
            except subprocess.TimeoutExpired:
                job.terminate()
                job.wait(timeout=10)
        for log in logs:
            log.close()
        record["gdb_exit_codes"] = [job.returncode for job in jobs]
        (args.run / "debugger-ownership.json").write_text(json.dumps(record, indent=2) + "\n")
    return record.get("request_exit_code", 1)


if __name__ == "__main__":
    sys.exit(main())
