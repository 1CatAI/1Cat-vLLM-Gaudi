# SPDX-License-Identifier: Apache-2.0
"""Observe device availability without reserving, resetting or stopping devices."""
from collections import deque
import json
from pathlib import Path
import subprocess
import time


def wait_for_released_modules(modules, record, *, timeout_s=300, poll_s=5):
    """Wait for asynchronous allocation teardown; this function acquires no lock."""
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
            raise RuntimeError(f'Modules still have live allocations: {pending}; ownership must be checked')
        time.sleep(poll_s)


def device_owners(module, *, sysfs=Path('/sys/class/accel'), device_root=Path('/dev/accel')):
    """Check open compute devices even when a worker has not allocated HBM yet."""
    for device in sysfs.glob('accel[0-9]*'):
        if int((device / 'device/module_id').read_text()) != module:
            continue
        result = subprocess.run(['fuser', str(device_root / device.name)], text=True, capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f'Cannot inspect module {module} ownership: {result.stderr.strip()}')
        return [int(pid) for pid in result.stdout.split()]
    raise RuntimeError(f'No compute device found for module {module}')


def wait_for_free_modules(record, *, count=4, modules=None, preferred=(0, 1, 4, 5),
                          timeout_s=None, poll_s=5, owners=device_owners):
    """Select free modules, or wait. Availability is observational, never a lease.

    A zero-utilization loaded model is occupied. Check open compute handles too,
    so workers starting to load weights are not mistaken for available devices.
    The runtime still arbitrates device opens if another launcher races us.
    """
    requested = None if modules is None else tuple(modules)
    if count < 1 or (requested is not None and (len(requested) != count or len(set(requested)) != count)):
        raise ValueError('Requested modules must be distinct and match the worker count')
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    observations = deque(maxlen=120)
    while True:
        load = subprocess.check_output(
            ['hl-smi', '-Q', 'module_id,memory.used,utilization.aip', '-f', 'csv,noheader'], text=True)
        rows = {int(fields[0]): fields for line in load.splitlines() if (fields := line.split(','))}
        if requested is not None and not set(requested).issubset(rows):
            raise ValueError(f'Requested modules are not present: {requested}')
        free, opened = [], {}
        for module, fields in sorted(rows.items()):
            if int(fields[1].split()[0]) > 768 or int(fields[2].split()[0]) != 0:
                continue
            opened[module] = owners(module)
            if not opened[module]:
                free.append(module)
        if requested is not None:
            selected = requested if set(requested).issubset(free) else ()
        elif len(preferred) == count and set(preferred).issubset(free):
            selected = tuple(preferred)
        else:
            selected = tuple(free[:count]) if len(free) >= count else ()
        observations.append(dict(time=time.time(), cards=load, free=free, owners=opened, selected=selected))
        Path(record).write_text(json.dumps(list(observations), indent=2)+'\n')
        if selected:
            return selected, load
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(f'Not enough free modules; available={free}, requested={requested}')
        print(f'Waiting for {count} free modules; currently available={free}', flush=True)
        time.sleep(poll_s)
