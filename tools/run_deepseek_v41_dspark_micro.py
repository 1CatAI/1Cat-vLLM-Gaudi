# SPDX-License-Identifier: Apache-2.0
"""Fixed-resource DSpark producer/consumer micro launch, using the shared lease."""
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
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide the micro command after --")
    workspace = Path(__file__).resolve().parents[1]
    # Use module 3 for one-rank operators. Pair 2/3 is reserved as this campaign's
    # comparison mapping when a two-rank consumer is required; such a test needs
    # its own NUMA-1 CPU lease, rather than sharing another task's CPUs.
    cpus = set(range(6))
    os.sched_setaffinity(0, cpus)
    invocation = [sys.executable, str(workspace / "tools/run_deepseek_v41.py"),
                  "--devices", "1", "--modules", "3", "--lock-dir", str(workspace.parent / "locks"),
                  "--runtime-profile", str(args.runtime_profile.resolve()),
                  "--engine-source", str(workspace.parent / "builds/dsv41-tp4-dspark-main-v1/engine"),
                  str(args.evidence.resolve()), "--", *command]
    code = subprocess.call(invocation, cwd=workspace)
    log = args.evidence / "habana_logs/synapse_runtime.log"
    if code and log.exists() and "computeFD -12" in log.read_text(errors="replace"):
        # Retry only this documented acquire failure after releasing our own
        # artifact cache. Never drop system caches or reset a shared device.
        released = 0
        repair_deadline = time.monotonic() + 10
        for root in (Path("/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1"),
                     Path("/opt/ssd960/builds/dsv41-tp4-dspark-unique-core-v1")):
            for folder, directories, files in os.walk(root, followlinks=False):
                if time.monotonic() >= repair_deadline:
                    break
                directories[:] = [d for d in directories if not Path(folder, d).is_symlink()]
                for name in files:
                    if time.monotonic() >= repair_deadline:
                        break
                    path = Path(folder, name)
                    if path.is_symlink() or not path.is_file():
                        continue
                    try:
                        with path.open("rb") as stream:
                            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        released += 1
                    except OSError:
                        pass
        retry = args.evidence.with_name(args.evidence.name + "-acquire-retry1").resolve()
        (args.evidence / "acquire-repair.json").write_text(json.dumps(
            dict(reason="computeFD -12", released_owned_files=released, retry=str(retry)), indent=2) + "\n")
        invocation[invocation.index(str(args.evidence.resolve()))] = str(retry)
        time.sleep(15)
        code = subprocess.call(invocation, cwd=workspace)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
