# SPDX-License-Identifier: Apache-2.0
"""Observe device availability and optionally lease existing cooperative locks."""
from collections import deque
from contextlib import contextmanager
import fcntl
import json
import os
import signal
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
                          timeout_s=None, poll_s=5, owners=device_owners, eligible=lambda module: True):
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
            if not opened[module] and eligible(module):
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


def wait_for_host_memory(record, *, minimum_gib, poll_s=5):
    """Avoid loading another full chain while the shared host is swapping.

    This is an observational wait, just like device availability; it reserves
    no resource and never stops another process. Load averages include driver
    waiters and are deliberately irrelevant to this check.
    """
    if minimum_gib < 0:
        raise ValueError('Minimum host headroom must be nonnegative')
    history = deque(maxlen=120)
    while True:
        memory = {line.split(':')[0]: int(line.split()[1])
                  for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith(('MemAvailable:', 'SwapTotal:', 'SwapFree:'))}
        available = memory['MemAvailable'] / (1024 ** 2)
        history.append(dict(time=time.time(), available_gib=available,
                            swap_used_gib=(memory['SwapTotal']-memory['SwapFree'])/(1024**2),
                            memory_pressure=Path('/proc/pressure/memory').read_text(),
                            minimum_gib=minimum_gib))
        Path(record).write_text(json.dumps(list(history), indent=2)+'\n')
        if available >= minimum_gib:
            return
        print(f'Waiting for host memory: available={available:.1f} GiB, required={minimum_gib:.1f} GiB',flush=True)
        time.sleep(poll_s)


@contextmanager
def lease_free_modules(record, *, lock_dir, count=4, modules=None, preferred=(0, 1, 4, 5), poll_s=5):
    """Hold all deployed lock aliases until our device work has been retired.

    Do not delete lock files: existing launchers may hold their original inode.
    Locks coordinate cooperating launchers; open-device checks protect against
    independent users that do not participate in this convention.
    """
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    def lock_paths(module):
        return [lock_dir / name for name in
                (f'module-{module}.lock', f'module{module}.lock', f'gaudi-module{module}.lock')]

    def unlocked(module):
        probes = []
        try:
            for path in lock_paths(module):
                probe = path.open('a+')
                probes.append(probe)
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        finally:
            for probe in probes:
                probe.close()

    while True:
        selected, load = wait_for_free_modules(record, count=count, modules=modules,
                                               preferred=preferred, poll_s=poll_s, eligible=unlocked)
        handles = []
        acquired = False
        try:
            for module in sorted(selected):
                for name in (f'module-{module}.lock', f'module{module}.lock', f'gaudi-module{module}.lock'):
                    handle = (lock_dir / name).open('a+')
                    handles.append(handle)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = not any(device_owners(module) for module in selected)
        except BlockingIOError:
            pass
        except BaseException:
            for handle in reversed(handles):
                handle.close()
            raise
        if acquired:
            try:
                yield selected, load
            finally:
                for handle in reversed(handles):
                    handle.close()
            return
        for handle in reversed(handles):
            handle.close()
        time.sleep(poll_s)



def _process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # State, parent, process group, start ticks. Unlike environ, stat does not
    # fault the target's swapped pages back in while reclaim is under pressure.
    return fields[0], int(fields[1]), int(fields[2]), int(fields[19])


def wait_for_owned_process(process, owned, *, record=None, minimum_free_gib=16, finished=None):
    """Track workers across their separate torchrun sessions before orphaning."""
    identity = _process_identity(process.pid)
    if identity is not None:
        owned[process.pid] = identity[3]
    low_memory = 0
    while True:
        rows = {int(path.name): _process_identity(int(path.name))
                for path in Path('/proc').iterdir() if path.name.isdigit()}
        growing = True
        while growing:
            growing = False
            for pid, row in rows.items():
                parent = None if row is None else rows.get(row[1])
                if (row is not None and parent is not None and
                        owned.get(row[1]) == parent[3] and pid not in owned):
                    owned[pid] = row[3]
                    growing = True
        if record is not None:
            Path(record).write_text(json.dumps(owned, indent=2)+'\n')
        status = process.poll()
        if status is not None:
            return status
        if finished is not None and finished():
            return None
        memory = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                      if line.startswith('MemAvailable:')) / 1024**2
        low_memory = low_memory + 1 if memory < minimum_free_gib else 0
        if low_memory >= 2:
            raise RuntimeError(f'Owned benchmark stopped before host exhaustion: {memory:.1f} GiB available')
        try:
            return process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def retire_process_group(process, *, owned=None, grace_s=30):
    """Retire only recorded child identities, including separate worker sessions."""
    owned = {} if owned is None else owned
    identity = _process_identity(process.pid)
    if identity is not None:
        owned.setdefault(process.pid, identity[3])

    def live():
        return [pid for pid, started in owned.items()
                if (row := _process_identity(pid)) is not None and row[3] == started and row[0] != 'Z']

    for pid in live():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        process.poll()
        if not live():
            return
        time.sleep(0.1)
    for pid in live():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
    deadline = time.monotonic() + grace_s
    while live() and time.monotonic() < deadline:
        time.sleep(0.1)
    if remaining := live():
        raise RuntimeError(f'Owned workers still retiring; retain device ownership: {remaining}')
