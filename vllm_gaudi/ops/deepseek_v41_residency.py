# SPDX-License-Identifier: Apache-2.0
"""Keep shared Engram file pages resident for the complete service lifetime.

One owner per TP shard and layer fits a bounded per-process RLIMIT_MEMLOCK.
Owners map the original files read-only; workers map those same physical pages.
Device-readable tables use one shared memfd backing instead: the driver pins
with FOLL_WRITE, which otherwise creates a private copy of every checkpoint page.
This module deliberately does not import torch or initialize an HPU context.
"""
import contextlib
import ctypes
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import resource
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


def _interval_union(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def locked_file_reuse(owners, *, proc_root=Path("/proc")):
    """Audit source pages already locked by another read-only shared mapping.

    MemAvailable already includes reclaimable file cache, so only fully resident,
    locked mappings earn admission credit. New device memfds are distinct physical
    allocations and receive no credit for their checkpoint's file pages.
    """
    wanted, identities = {}, {}
    for owner in owners:
        if owner.get("shared_memory", False):
            continue
        for region in owner["regions"]:
            stat = os.stat(region["file"])
            if region["offset"] + region["length"] > stat.st_size:
                raise ValueError("Engram mapping exceeds the frozen file")
            key = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
            offset, length = aligned_region(region)
            wanted.setdefault(key, []).append((offset, offset + length))
            identities[region["file"]] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    wanted = {key: _interval_union(extents) for key, extents in wanted.items()}
    header = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ([-rwxps]{4}) ([0-9a-f]+) "
                        r"([0-9a-f]+):([0-9a-f]+) ([0-9]+)(?: |$)")
    credited, records = {}, []
    if wanted:
        for process in proc_root.glob("[0-9]*"):
            try:
                # Avoid expanding smaps for processes with no matching tables.
                matches = [header.match(line) for line in (process / "maps").read_text().splitlines()]
                if not any(match and (int(match[5], 16), int(match[6], 16), int(match[7])) in wanted
                           for match in matches):
                    continue
                smaps = (process / "smaps").read_text()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            current = None
            for line in smaps.splitlines():
                match = header.match(line)
                if match:
                    key = (int(match[5], 16), int(match[6], 16), int(match[7]))
                    current = None
                    if key in wanted and match[3] == "r--s":
                        current = dict(key=key,
                                       offset=int(match[4], 16),
                                       length=int(match[2], 16) - int(match[1], 16),
                                       rss=0)
                elif current and line.startswith("Rss:"):
                    current["rss"] = int(line.split()[1]) * 1024
                elif current and line.startswith("VmFlags:"):
                    if "lo" in line.split()[1:] and current["rss"] == current["length"]:
                        low, high = current["offset"], current["offset"] + current["length"]
                        for start, stop in wanted[current["key"]]:
                            overlap = (max(low, start), min(high, stop))
                            if overlap[1] > overlap[0]:
                                credited.setdefault(current["key"], []).append(overlap)
                                records.append(
                                    dict(pid=int(process.name),
                                         device_inode=current["key"],
                                         offset=overlap[0],
                                         length=overlap[1] - overlap[0]))
                    current = None
    for file, identity in identities.items():
        stat = os.stat(file)
        if identity != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Engram source changed during resident-page audit")
    reused = sum(stop - start for intervals in credited.values() for start, stop in _interval_union(intervals))
    return dict(reused_bytes=reused,
                mappings=records,
                source_identities={
                    file: dict(zip(("device", "inode", "size", "mtime_ns"), identity))
                    for file, identity in identities.items()
                })


def shared_host_region(region):
    """Copy one source extent into the sole shared, device-pinnable backing.

    Only a bounded transfer chunk is temporary. Both CPU gather and the device
    producer must bind this backing, never keep a second resident table. The
    checkpoint is opened read-only and its identity is checked across the copy.
    """
    descriptor = os.memfd_create("dsv41-engram", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with open(region["file"], "rb", buffering=0) as source:
            before = os.fstat(source.fileno())
            if region["offset"] < 0 or region["offset"] + region["length"] > before.st_size:
                raise ValueError("Shared Engram extent exceeds its source")
            os.ftruncate(descriptor, region["length"])
            source.seek(region["offset"])
            remaining, digest = region["length"], hashlib.sha256()
            while remaining:
                data = source.read(min(16 * 1024**2, remaining))
                if not data:
                    raise ValueError("Truncated shared Engram source")
                digest.update(data)
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Short shared Engram write")
                    view = view[written:]
                remaining -= len(data)
            after = os.fstat(source.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino,
                                                                                      after.st_size, after.st_mtime_ns):
                raise RuntimeError("Engram source changed during shared preparation")
        # A writable registration is required by the driver. Seal its extent;
        # the native producer makes its CPU mapping read-only after pinning.
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        target = os.fstat(descriptor)
        binding = dict(file=f"/proc/{os.getpid()}/fd/{descriptor}",
                       offset=0,
                       length=region["length"],
                       source=dict(region),
                       source_identity=dict(device=before.st_dev,
                                            inode=before.st_ino,
                                            size=before.st_size,
                                            mtime_ns=before.st_mtime_ns),
                       device=target.st_dev,
                       inode=target.st_ino,
                       sha256=digest.hexdigest(),
                       backing="shared_memfd")
        return descriptor, binding
    except BaseException:
        os.close(descriptor)
        raise


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
    mappings, descriptors, bindings = [], [], []
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
        regions = owner["regions"]
        if owner.get("shared_memory", False):
            for region in regions:
                descriptor, binding = shared_host_region(region)
                descriptors.append(descriptor)
                bindings.append(binding)
            regions = bindings
        requested = sum(aligned_region(r)[1] for r in regions)
        for region in regions:
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
                 bindings=bindings,
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
        for descriptor in descriptors:
            os.close(descriptor)
        connection.close()


class EngramResidency:
    """Start before model loading; release after all consumers have stopped."""

    def __init__(self, owners, budget_bytes=224 * 1024**3, *, device_layers=()):
        self.owners = [dict(owner, shared_memory=owner["layer"] in device_layers) for owner in owners]
        self.budget_bytes = budget_bytes
        self.processes, self.connections, self.reports = [], [], []
        self.stopped = threading.Event()
        self.monitor = None
        self.admission = None
        self.close_lock = threading.Lock()

    def start(self, *, timeout=600, check_available=True):
        requested = sum(aligned_region(r)[1] for owner in self.owners for r in owner["regions"])
        if requested > self.budget_bytes or not self.owners:
            raise RuntimeError("Engram mappings exceed the configured host budget")
        if check_available:
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(fields["MemAvailable"].split()[0]) * 1024
            audit = locked_file_reuse(self.owners) if available < self.budget_bytes else dict(reused_bytes=0)
            required = max(0, self.budget_bytes - audit["reused_bytes"])
            self.admission = dict(audit,
                                  available_bytes=available,
                                  required_available_bytes=required,
                                  budget_bytes=self.budget_bytes,
                                  mapped_bytes=requested)
            if available < required:
                raise RuntimeError(f"Engram host budget unavailable after locked-page reuse: {available} < {required}; "
                                   f"reused {audit['reused_bytes']} bytes; wait for resources")
        context = multiprocessing.get_context("spawn")
        deadline = time.monotonic() + timeout
        try:
            for owner in self.owners:
                if self.stopped.is_set():
                    raise RuntimeError("Engram residency stopped during preparation")
                parent, child = context.Pipe()
                process = context.Process(target=_owner,
                                          args=(child, owner, os.getpid()),
                                          name=f"engram-resident-tp{owner['tp']}-l{owner['layer']}")
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            for connection in self.connections:
                while not connection.poll(.2):
                    if self.stopped.is_set():
                        raise RuntimeError("Engram residency stopped during preparation")
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

    def worker_bindings(self):
        self.check()
        result = {}
        for report in self.reports:
            if report["bindings"]:
                result.setdefault(str(report["tp"]), {})[str(report["layer"])] = report["bindings"]
        return result

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
        # Startup cancellation and the controller can retire the same lease
        # concurrently. Serialize closing descriptors and joining children.
        with self.close_lock:
            for connection in self.connections:
                connection.close()
            for process in self.processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
            if self.monitor is not None:
                self.monitor.join(timeout=2)


def _publish(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def wait_for_resident_tables(directory, rank, module, *, timeout=900):
    """A worker calls this after device/communication initialization."""
    directory = Path(directory)
    controller = json.loads((directory / "controller.json").read_text())
    if not 0 <= rank < len(controller["modules"]) or str(module) != controller["modules"][rank]:
        raise RuntimeError("Engram startup worker/module ownership mismatch")
    _publish(directory / f"worker-{rank}.json", dict(pid=os.getpid(), module=str(module)))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (directory / "error.json").exists():
            raise RuntimeError(json.loads((directory / "error.json").read_text())["error"])
        if (directory / "ready.json").exists():
            return json.loads((directory / "ready.json").read_text())["bindings"]
        os.kill(controller["pid"], 0)
        time.sleep(.1)
    raise TimeoutError("Engram residency did not complete after device initialization")


class EngramStartup:
    """Initialize device contexts before locking the large host table.

    Gaudi context creation needs physically contiguous host DMA pages. Filling
    and locking a large table first can make that allocation fail even while
    MemAvailable remains high. Workers rendezvous before any model allocation;
    all subsequently bind the same controller-owned table pages.
    """

    def __init__(self, residency, modules, *, report_path=None, timeout=900):
        self.residency = residency
        self.modules = tuple(str(module) for module in modules)
        self.report_path = None if report_path is None else Path(report_path)
        self.timeout = timeout
        self.temporary = tempfile.TemporaryDirectory(prefix="engram-start-")
        self.directory = Path(self.temporary.name)
        self.stopped = threading.Event()
        self.thread = None
        _publish(self.directory / "controller.json", dict(pid=os.getpid(), modules=self.modules))

    def start(self, on_failure):

        def prepare():
            try:
                deadline = time.monotonic() + self.timeout
                while not self.stopped.is_set():
                    workers = [self.directory / f"worker-{rank}.json" for rank in range(len(self.modules))]
                    if all(path.exists() for path in workers):
                        for path, module in zip(workers, self.modules):
                            worker = json.loads(path.read_text())
                            if worker["module"] != module:
                                raise RuntimeError("Engram startup received a foreign device binding")
                            os.kill(worker["pid"], 0)
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Devices did not initialize before Engram preparation")
                    self.stopped.wait(.1)
                if self.stopped.is_set():
                    return
                self.residency.start(timeout=self.timeout)
                if self.stopped.is_set():
                    return
                report = dict(policy="locked-shared-host-pages",
                              budget_bytes=self.residency.budget_bytes,
                              admission=self.residency.admission,
                              owners=self.residency.reports)
                if self.report_path is not None:
                    _publish(self.report_path, report)
                self.residency.watch(on_failure)
                _publish(self.directory / "ready.json", dict(bindings=self.residency.worker_bindings()))
                print("Engram residency ready: " + json.dumps(report), flush=True)
                # PR_SET_PDEATHSIG follows the creating Linux thread, not
                # merely its process. Keep that thread alive with its owners.
                self.stopped.wait()
            except BaseException as error:
                if not self.stopped.is_set():
                    _publish(self.directory / "error.json", dict(error=repr(error)))
            finally:
                if self.stopped.is_set():
                    self.residency.close()

        self.thread = threading.Thread(target=prepare, name="engram-startup", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stopped.set()
        self.residency.close()
        if self.thread is not None:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("Engram preparation did not retire on shutdown")
        self.temporary.cleanup()
