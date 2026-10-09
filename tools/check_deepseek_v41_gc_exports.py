# SPDX-License-Identifier: Apache-2.0
"""Verify required additive TPC exports on CPU before taking HPU leases."""
import argparse
import ctypes
import json
import os
from pathlib import Path


class GuidInfo(ctypes.Structure):
    # Matching tpc_kernel_lib_interface.h: name, hash union and properties.
    _fields_ = [("name", ctypes.c_char * 64), ("name_hash", ctypes.c_uint64),
                ("properties", ctypes.c_uint32)]


def verify(library, base_library, required):
    os.environ["VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL"] = str(Path(base_library).resolve(strict=True))
    library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
    query = library.GetKernelGuids
    query.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(GuidInfo)]
    query.restype = ctypes.c_int
    count = ctypes.c_uint32()
    if query(3, ctypes.byref(count), None) != 0 or not 1 <= count.value <= 100000:
        raise RuntimeError("Cannot enumerate the locked Gaudi2 kernel database")
    guids = (GuidInfo * count.value)()
    if query(3, ctypes.byref(count), guids) != 0:
        raise RuntimeError("Cannot read Gaudi2 TPC exports")
    names = {row.name.decode(): index for index, row in enumerate(guids)}
    missing = sorted(set(required) - names.keys())
    if missing:
        raise RuntimeError(f"Required TPC GUIDs absent from registration table: {missing}")
    return dict(guid_count=count.value, required={name: names[name] for name in required}, cpu_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--required-guid", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.library, args.base_library, args.required_guid)))


if __name__ == "__main__":
    main()
