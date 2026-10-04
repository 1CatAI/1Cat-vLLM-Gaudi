# SPDX-License-Identifier: Apache-2.0
"""Wait for leased modules to finish a previous worker's asynchronous teardown."""
import json
from pathlib import Path
import subprocess
import time


def wait_for_released_modules(modules, record, *, timeout_s=300, poll_s=5):
    """Caller must hold module leases. Never reset or terminate another worker."""
    deadline = time.monotonic() + timeout_s
    observations = []
    while True:
        load = subprocess.check_output(
            ['hl-smi', '-Q', 'module_id,memory.used,utilization.aip', '-f', 'csv,noheader'], text=True)
        rows = {int(fields[0]): fields for line in load.splitlines() if (fields := line.split(','))}
        pending = [module for module in modules
                   if int(rows[module][1].split()[0]) > 768 or int(rows[module][2].split()[0]) != 0]
        observations.append(dict(time=time.time(), cards=load, pending=pending))
        Path(record).write_text(json.dumps(observations, indent=2)+'\n')
        if not pending:
            return load
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Leased modules still have live allocations: {pending}; ownership must be checked')
        time.sleep(poll_s)
