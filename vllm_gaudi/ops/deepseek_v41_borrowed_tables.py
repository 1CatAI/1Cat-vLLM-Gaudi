# SPDX-License-Identifier: Apache-2.0
"""Independently retain validated immutable Engram backing across services."""
import ctypes
import json
import os
from pathlib import Path


def resident_bytes(descriptor, size):
    import numpy as np

    page = os.sysconf("SC_PAGESIZE")
    length = (size + page - 1) // page * page
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    address = libc.mmap(None, length, 1, 1, descriptor, 0)
    if address == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "Cannot inspect shared table residency")
    try:
        total = 0
        for offset in range(0, length, page * 65536):
            count = min(length - offset, page * 65536) // page
            status = (ctypes.c_ubyte * count)()
            if libc.mincore(address + offset, count * page, status):
                raise OSError(ctypes.get_errno(), "Cannot certify shared table residency")
            total += int(np.count_nonzero(np.frombuffer(status, dtype=np.uint8) & 1)) * page
        return min(total, size)
    finally:
        libc.munmap(address, length)


class BorrowedEngramTables:
    """Retain our own descriptors; a keeper can retire without breaking users."""

    def __init__(self, model, manifest):
        from vllm_gaudi.ops.deepseek_v41_host import resident_table_source
        from vllm_gaudi.ops.deepseek_v41_residency import table_regions

        self.descriptors, self.bindings = [], {}
        self.reused_bytes = 0
        supplied = json.loads(Path(manifest).read_text())["bindings"]
        try:
            identities = set()
            for owner in table_regions(model):
                records = supplied[str(owner["tp"])][str(owner["layer"])]
                if len(records) != len(owner["regions"]):
                    raise ValueError("Borrowed table extent count differs from the model")
                retained = []
                for region, record in zip(owner["regions"], records, strict=True):
                    # Validate source and sealed backing before and after opening
                    # our descriptor, so a retiring keeper cannot substitute it.
                    item = dict(file=region["file"], shard_offset=region["offset"], shard_bytes=region["length"])
                    resident_table_source(item, record)
                    fd = os.open(record["file"], os.O_RDONLY | os.O_CLOEXEC)
                    self.descriptors.append(fd)
                    current = dict(record, file=f"/proc/{os.getpid()}/fd/{fd}")
                    resident_table_source(item, current)
                    identity = (current["device"], current["inode"])
                    if identity in identities:
                        raise ValueError("Borrowed table extents must have independent model ownership")
                    identities.add(identity)
                    self.reused_bytes += resident_bytes(fd, region["length"])
                    retained.append(current)
                self.bindings.setdefault(str(owner["tp"]), {})[str(owner["layer"])] = retained
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.descriptors:
            os.close(fd)
        self.descriptors.clear()
