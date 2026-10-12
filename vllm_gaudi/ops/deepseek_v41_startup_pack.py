# SPDX-License-Identifier: Apache-2.0
"""Validated optional CPU helper for the unchanged compressed weight layout."""
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def native_q16_packer(path):
    path = Path(path).resolve()
    record = json.loads(path.with_suffix(".json").read_text())
    if record["schema"] != 1 or record["abi"] != 1 or record["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("CPU startup packing library differs from its build contract")
    library = ctypes.CDLL(str(path))
    library.dsv41_startup_pack_abi.restype = ctypes.c_int
    if library.dsv41_startup_pack_abi() != 1:
        raise RuntimeError("CPU startup packing library ABI mismatch")
    operation = library.dsv41_startup_pack
    operation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    operation.restype = ctypes.c_int

    def pack(q16):
        if q16.dtype != np.dtype("<i2") or q16.ndim != 2 or not q16.flags.c_contiguous:
            raise ValueError("CPU startup pack requires contiguous N128 I16 source")
        blocks, stream = q16.shape
        if blocks % 2 or stream % 4096 or q16.nbytes > 64 << 20:
            raise ValueError("CPU startup pack exceeds its bounded N256 shape contract")
        output = np.empty((blocks // 2, stream * 2), dtype="<i2")
        if operation(q16.ctypes.data, output.ctypes.data, blocks, stream // 32):
            raise RuntimeError("CPU startup packing rejected its validated source")
        return output

    return pack
