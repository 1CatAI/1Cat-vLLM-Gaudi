# SPDX-License-Identifier: Apache-2.0
"""Own immutable CPU table backings independently of model service restarts.

The manifest describes this live owner, not a persistent device binding. Services
must validate checkpoint identities and retain their own file descriptors before
use. No device context, recipe ID, device address or communicator is retained.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-gib", type=int, default=224)
    args = parser.parse_args()
    from vllm_gaudi.ops.deepseek_v41_residency import EngramResidency, table_regions

    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "ready.json"
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def process_start(pid):
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]

    if manifest.exists():
        previous = json.loads(manifest.read_text())
        try:
            alive = (previous.get("boot_id") == boot_id
                     and process_start(previous["pid"]) == previous.get("process_start"))
        except ProcessLookupError:
            alive = False
        except FileNotFoundError:
            alive = False
        if alive:
            raise RuntimeError("Never overwrite a live backing manifest")
        # A dead owner has no retained descriptors. Establish fresh backings;
        # never reuse its file names as though the allocations still existed.
        manifest.unlink()
    owners = table_regions(args.prepared)
    layers = tuple(sorted({owner["layer"] for owner in owners}))
    tables = EngramResidency(owners, args.budget_gib * 1024**3, device_layers=layers)
    started = time.time()
    try:
        tables.start(cancel=stopped)
        if stopped.is_set():
            return
        record = dict(schema=1, pid=os.getpid(), boot_id=boot_id, process_start=process_start(os.getpid()),
                      model=str(args.prepared.resolve()),
                      started_at=started, ready_at=time.time(), bindings=tables.worker_bindings())
        temporary = args.output / "ready.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(manifest)
        (args.output / "preparation.json").write_text(json.dumps(
            dict(elapsed_seconds=record["ready_at"] - started, owners=tables.reports,
                 admission=tables.admission), indent=2) + "\n")
        if address := os.environ.get("NOTIFY_SOCKET"):
            address = "\0" + address[1:] if address.startswith("@") else address
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notification:
                notification.connect(address)
                notification.sendall(b"READY=1")
        print("Immutable CPU table backings ready", flush=True)
        while not stopped.wait(1):
            tables.check()
    finally:
        manifest.unlink(missing_ok=True)
        tables.close()


if __name__ == "__main__":
    main()
