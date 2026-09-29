# SPDX-License-Identifier: Apache-2.0
"""Keep shared Engram file pages resident for the complete service lifetime.

One owner per TP shard and layer fits a bounded per-process RLIMIT_MEMLOCK.
Owners map the original files read-only; workers map those same physical pages.
This module deliberately does not import torch or initialize an HPU context.
"""
import contextlib
import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import re
import signal
import tempfile
import threading
import time


def table_regions(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    owners = []
    for tp, record in sorted(manifest["engram_host_shards"].items()):
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("Engram host shard manifest changed")
        shard = json.loads(path.read_text())
        if (shard["tp_rank"] != int(tp) or shard["pp_owner"] != 0 or not shard["shared_read_only"]
                or shard["sharding"] != "complete_hash_heads" or shard["model_revision"] != manifest["model_revision"]):
            raise RuntimeError("Engram residency ownership mismatch")
        layers = sorted({name.split(".")[1] for name in shard["tables"]}, key=int)
        for layer in layers:
            regions = []
            for suffix in ("weight", "scale"):
                item = shard["tables"][f"layers.{layer}.engram.embed.{suffix}"]
                length = (item["row_stop"] - item["row_start"]) * item["row_bytes"]
                if length <= 0 or length != item["shard_bytes"]:
                    raise RuntimeError("Invalid Engram shard byte extent")
                regions.append(dict(file=item["file"], offset=item["shard_offset"], length=length))
            owners.append(dict(tp=int(tp), layer=int(layer), regions=regions))
    return owners


def aligned_region(region):
    page = os.sysconf("SC_PAGESIZE")
    offset, length = region["offset"], region["length"]
    if offset < 0 or length <= 0:
        raise ValueError("Invalid Engram mapping extent")
    delta = offset % page
    return offset - delta, ((length + delta + page - 1) // page) * page


def locked_shared_bytes(owners, proc=Path("/proc")):
    """Credit only identical file extents already fully locked by live owners.

    Ordinary page-cache residency is reclaimable and receives no credit. Each
    inode range is counted once even when several processes lock the same pages.
    Our new owners still take their own locks and verify mincore before readiness.
    """
    wanted, covered = {}, {}
    for owner in owners:
        for region in owner["regions"]:
            stat = Path(region["file"]).stat()
            key = os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino
            offset, size = aligned_region(region)
            wanted.setdefault(key, []).append((offset, offset + size))
    header = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) ([0-9a-f]+) ([0-9a-f]+):([0-9a-f]+) (\d+)(?: |$)")
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            # Residency helpers use spawn and small address spaces. Workers'
            # unlocked mappings cannot establish this credit.
            if b"multiprocessing.spawn" not in (process / "cmdline").read_bytes():
                continue
            current, resident, locked = None, 0, 0
            for line in (process / "smaps").read_text().splitlines():
                match = header.match(line)
                if match:
                    start, end, perms, offset, major, minor, inode = match.groups()
                    key = int(major, 16), int(minor, 16), int(inode)
                    current = (key, int(offset, 16),
                               int(end, 16) - int(start, 16)) if (key in wanted and perms.endswith("s")) else None
                    resident = locked = 0
                elif current is not None and line.startswith("Rss:"):
                    resident = int(line.split()[1]) * 1024
                elif current is not None and line.startswith("Locked:"):
                    locked = int(line.split()[1]) * 1024
                elif current is not None and line.startswith("VmFlags:"):
                    key, offset, size = current
                    # Locked is proportionally charged for shared mappings;
                    # VmFlags lo plus full RSS proves this entire VMA is locked.
                    if "lo" in line.split()[1:] and resident >= size and locked > 0:
                        for start, end in wanted[key]:
                            lo, hi = max(start, offset), min(end, offset + size)
                            if lo < hi:
                                covered.setdefault(key, []).append((lo, hi))
                    current = None
        except (OSError, ValueError):
            # An owner may exit during inspection; missing evidence gives no credit.
            continue
    total = 0
    for intervals in covered.values():
        end = -1
        for lo, hi in sorted(intervals):
            total += max(0, hi - max(lo, end))
            end = max(end, hi)
    return total


class LockedMapping:

    def __init__(self, region):
        self.address = None
        self.lib = ctypes.CDLL(None, use_errno=True)
        self.lib.mmap.restype = ctypes.c_void_p
        self.lib.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long
        ]
        for name in ("mlock", "munmap"):
            getattr(self.lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            getattr(self.lib, name).restype = ctypes.c_int
        self.lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self.lib.mincore.restype = ctypes.c_int
        offset, self.length = aligned_region(region)
        fd = os.open(region["file"], os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            if region["offset"] + region["length"] > stat.st_size:
                raise ValueError("Engram mapping exceeds the frozen file")
            self.identity = dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            address = self.lib.mmap(None, self.length, 1, 1, fd, offset)  # PROT_READ, MAP_SHARED
        finally:
            os.close(fd)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "Engram read-only mmap failed")
        self.address = address
        if self.lib.mlock(address, self.length):
            error = ctypes.get_errno()
            self.close()
            raise OSError(error, "Engram mlock failed; residency is not qualified")

    def resident_pages(self):
        page = os.sysconf("SC_PAGESIZE")
        resident = 0
        for start in range(0, self.length, page * 65536):
            length = min(self.length - start, page * 65536)
            status = (ctypes.c_ubyte * (length // page))()
            if self.lib.mincore(self.address + start, length, status):
                raise OSError(ctypes.get_errno(), "Engram mincore failed")
            resident += sum(value & 1 for value in status)
        return resident

    def close(self):
        if self.address is not None:
            self.lib.munmap(self.address, self.length)
            self.address = None


def _owner(connection, owner, parent_pid):
    mappings = []
    try:
        # Release only our mappings if the API owner exits without Python cleanup.
        lib = ctypes.CDLL(None, use_errno=True)
        if lib.prctl(1, signal.SIGTERM, 0, 0, 0) or os.getppid() != parent_pid:  # PR_SET_PDEATHSIG
            raise RuntimeError("Engram residency parent disappeared")
        requested = sum(aligned_region(r)[1] for r in owner["regions"])
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if hard != resource.RLIM_INFINITY and requested > hard:
            raise RuntimeError(f"Engram owner needs {requested} locked bytes; hard limit is {hard}")
        if soft != resource.RLIM_INFINITY and requested > soft:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (requested, hard))
        for region in owner["regions"]:
            mappings.append(LockedMapping(region))
        page = os.sysconf("SC_PAGESIZE")
        resident = sum(mapping.resident_pages() for mapping in mappings)
        if resident * page != requested:
            raise RuntimeError("Locked Engram mappings are not fully resident")
        connection.send(
            dict(ready=True,
                 pid=os.getpid(),
                 tp=owner["tp"],
                 layer=owner["layer"],
                 locked_bytes=requested,
                 resident_pages=resident,
                 files=[mapping.identity for mapping in mappings]))
        # EOF also releases the mappings when the controller vanishes.
        connection.recv()
    except EOFError:
        pass
    except BaseException as error:
        with contextlib.suppress(BrokenPipeError, EOFError):
            connection.send(dict(ready=False, error=repr(error)))
    finally:
        for mapping in mappings:
            mapping.close()
        connection.close()


class EngramResidency:
    """Start before model loading; release after all consumers have stopped."""

    def __init__(self, owners, budget_bytes=224 * 1024**3):
        self.owners, self.budget_bytes = owners, budget_bytes
        self.processes, self.connections, self.reports = [], [], []
        self.stopped = threading.Event()
        self.monitor = None
        self.shared_locked_bytes = 0

    def start(self, *, timeout=600, check_available=True, cancel=None):
        requested = sum(aligned_region(r)[1] for owner in self.owners for r in owner["regions"])
        if requested > self.budget_bytes or not self.owners:
            raise RuntimeError("Engram mappings exceed the configured host budget")
        if check_available:
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(fields["MemAvailable"].split()[0]) * 1024
            self.shared_locked_bytes = locked_shared_bytes(self.owners) if available < self.budget_bytes else 0
            additional = self.budget_bytes - self.shared_locked_bytes
            if available < additional:
                raise RuntimeError(f"Engram host budget unavailable: {available} < {additional} additional bytes "
                                   f"({self.shared_locked_bytes} already locked shared bytes); wait for resources")
        context = multiprocessing.get_context("spawn")
        deadline = time.monotonic() + timeout
        try:
            for owner in self.owners:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Engram residency preparation cancelled")
                parent, child = context.Pipe()
                process = context.Process(target=_owner,
                                          args=(child, owner, os.getpid()),
                                          name=f"engram-resident-tp{owner['tp']}-l{owner['layer']}")
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            for connection in self.connections:
                while not connection.poll(.05):
                    if cancel is not None and cancel.is_set():
                        raise RuntimeError("Engram residency preparation cancelled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Engram residency preparation timed out")
                report = connection.recv()
                self.reports.append(report)
                if not report["ready"]:
                    raise RuntimeError(report["error"])
            self.check()
        except BaseException:
            self.close()
            raise
        return self

    def check(self):
        if self.stopped.is_set() or len(self.reports) != len(self.owners) or any(not process.is_alive()
                                                                                 for process in self.processes):
            raise RuntimeError("Engram residency owner was lost; stop serving")

    def watch(self, on_failure):

        def monitor():
            while not self.stopped.wait(.5):
                try:
                    self.check()
                except RuntimeError:
                    on_failure()
                    return

        self.monitor = threading.Thread(target=monitor, name="engram-residency-monitor", daemon=True)
        self.monitor.start()

    def close(self):
        self.stopped.set()
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        if self.monitor is not None:
            self.monitor.join(timeout=2)


def wait_for_engram_residency(rank, world_size, *, timeout=900):
    """Announce initialized device/collectives before pinning the host table."""
    directory = os.environ.get("DSV41_ENGRAM_DEVICE_GATE")
    if directory is None:
        return
    directory = Path(directory)
    contract = json.loads((directory / "contract.json").read_text())
    if world_size != contract["world_size"] or not 0 <= rank < world_size:
        raise RuntimeError("Engram device/residency world mismatch")
    marker = directory / f"rank{rank}.json"
    with marker.open("x") as stream:
        json.dump(dict(rank=rank, pid=os.getpid()), stream)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (directory / "error.json").exists():
            raise RuntimeError(json.loads((directory / "error.json").read_text())["error"])
        if (directory / "ready.json").exists():
            return
        try:
            os.kill(contract["parent_pid"], 0)
        except ProcessLookupError as error:
            raise RuntimeError("Engram residency controller disappeared") from error
        time.sleep(.05)
    raise TimeoutError("Device initialized but Engram residency is not ready")


class EngramDeviceGate:
    """Initialize HPU DMA contexts before mlock makes table pages unmovable.

    Workers stop at this gate before model allocation. The private directory
    belongs to one service generation; it is not a reusable readiness cache.
    """

    def __init__(self, residency, *, world_size=4, timeout=900, on_ready=None, on_failure=None):
        self.residency = residency
        self.world_size, self.timeout = world_size, timeout
        self.on_ready, self.on_failure = on_ready, on_failure
        self.stop = threading.Event()
        self.directory = tempfile.TemporaryDirectory(prefix="dsv41-engram-device-")
        self.path = Path(self.directory.name)
        (self.path / "contract.json").write_text(json.dumps(dict(world_size=world_size, parent_pid=os.getpid())))
        self.thread = None

    def start(self):
        if "DSV41_ENGRAM_DEVICE_GATE" in os.environ:
            raise RuntimeError("Engram device gate already belongs to a service")
        os.environ["DSV41_ENGRAM_DEVICE_GATE"] = str(self.path)
        self.thread = threading.Thread(target=self._run, name="engram-device-gate", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            deadline = time.monotonic() + self.timeout
            while not self.stop.wait(.05):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Not all HPU workers reached the Engram residency gate")
                records = []
                for rank in range(self.world_size):
                    try:
                        record = json.loads((self.path / f"rank{rank}.json").read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        break
                    if record["rank"] != rank:
                        raise RuntimeError("Engram device gate rank mismatch")
                    os.kill(record["pid"], 0)
                    records.append(record)
                if len(records) == self.world_size:
                    break
            if self.stop.is_set():
                return
            self.residency.start(timeout=max(0, deadline - time.monotonic()), cancel=self.stop)
            report = dict(policy="locked-shared-file-pages-after-device-init",
                          budget_bytes=self.residency.budget_bytes,
                          owners=self.residency.reports,
                          initialized_workers=records)
            if self.on_ready is not None:
                self.on_ready(report)
            temporary = self.path / "ready.tmp"
            temporary.write_text(json.dumps(report))
            temporary.replace(self.path / "ready.json")
            while not self.stop.wait(.5):
                self.residency.check()
        except Exception as error:
            temporary = self.path / "error.tmp"
            temporary.write_text(json.dumps(dict(error=repr(error))))
            temporary.replace(self.path / "error.json")
            if not self.stop.is_set() and self.on_failure is not None:
                self.on_failure()
        finally:
            self.residency.close()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            # Owners are spawned and retired only by the controller thread.
            # Do not race its mlock/connection startup from API cleanup.
            self.thread.join()
        if os.environ.get("DSV41_ENGRAM_DEVICE_GATE") == str(self.path):
            os.environ.pop("DSV41_ENGRAM_DEVICE_GATE")
        self.directory.cleanup()
