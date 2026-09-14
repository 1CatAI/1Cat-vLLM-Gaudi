# SPDX-License-Identifier: Apache-2.0
"""Check archived recipe debug metadata through the pinned runtime, without a device.

The ctypes layout is the locked Synapse recipe.h v1.2 diagnostic ABI, not a
public runtime interface. This tool never launches a recipe or acquires an HPU.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from collect_deepseek_v41_trace import recipe_symbols


class Node(ct.Structure):
    _fields_ = [("device", ct.c_uint32), ("context", ct.c_uint16), ("full_context", ct.c_uint32),
                ("descriptors", ct.c_uint32), ("blob", ct.c_uint32), ("name", ct.c_void_p),
                ("operation", ct.c_void_p), ("dtype", ct.c_void_p), ("rois", ct.c_uint16),
                ("engines", ct.c_void_p)]


class Debug(ct.Structure):
    _fields_ = [("major", ct.c_uint32), ("minor", ct.c_uint32), ("recipe", ct.c_uint16),
                ("count", ct.c_uint32), ("nodes", ct.POINTER(Node)), ("printf_count", ct.c_uint32),
                ("printf_address", ct.c_void_p), ("printf_section", ct.c_uint64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--virtual-chain", action="store_true",
                        help="Exercise the exact legacy profiler getRecipeDebugInfo slot through its wrapper chain")
    args = parser.parse_args()
    with args.library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != args.sha256 or ct.sizeof(Node) != 64 or ct.sizeof(Debug) != 48:
        raise RuntimeError("This diagnostic requires the exact locked library and recipe debug ABI")
    args.output.mkdir(parents=True, exist_ok=False)
    library = ct.CDLL(str(args.library))
    initialize, destroy = library.synInitialize, library.synDestroy
    deserialize, release = library.synRecipeDeSerialize, library.synRecipeDestroy
    deserialize.argtypes, release.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p], [ct.c_void_p]
    singleton = library._ZN12synSingleton19getInstanceInternalEv
    singleton.restype = ct.c_void_p
    get_debug = library._ZN12synSingleton18getRecipeDebugInfoEP20InternalRecipeHandlePPK12debug_info_t
    get_debug.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.POINTER(Debug))]
    assert initialize() == 0
    instance = singleton()
    if args.virtual_chain:
        outer = library._ZN12synSingleton11getInstanceEv
        outer.restype = ct.c_void_p
        instance = outer()
        table = ct.cast(instance, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
        # libhw_trace 1.16 saveDebugInfoAndProgramDataBlobs invokes vtable +0x2f0.
        get_debug = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.POINTER(Debug)))(table[0x2f0 // 8])
    records = []
    try:
        for path in sorted(args.cache.rglob("*.recipe")):
            print(path, flush=True)
            expected = recipe_symbols(path)
            handle, pointer = ct.c_void_p(), ct.POINTER(Debug)()
            assert deserialize(ct.byref(handle), str(path).encode()) == 0
            try:
                assert get_debug(instance, handle, ct.byref(pointer)) == 0
                actual = pointer.contents
                assert (actual.major, actual.minor, actual.recipe, actual.count) == (
                    expected["major"], expected["minor"], expected["recipe_id"], len(expected["nodes"]))
                for index, reference in enumerate(expected["nodes"]):
                    node = actual.nodes[index]
                    assert (node.device, node.context, node.full_context, node.descriptors, node.blob) == (
                        reference["device_type"], reference["context_id"], reference["full_context_id"],
                        reference["descriptors"], reference["kernel_blob_index"])
                    for address, name in ((node.name, "node"), (node.operation, "kernel"), (node.dtype, "dtype")):
                        assert address and ct.string_at(address).decode() == reference[name]
                    assert node.rois == len(reference["working_engines"])
                    assert list(ct.string_at(node.engines, node.rois)) == reference["working_engines"]
                records.append({"path": str(path), "sha256": expected["sha256"], "nodes": actual.count})
            finally:
                assert release(handle) == 0
        (args.output / "result.json").write_text(json.dumps({"library": str(args.library), "sha256": digest,
            "virtual_chain": args.virtual_chain,
            "status": "deserialized debug metadata matches serialized source", "recipes": records,
            "limitation": "No device launch, live profiler registration or asynchronous lifetime is exercised"},
            indent=2) + "\n")
    finally:
        assert destroy() == 0


if __name__ == "__main__":
    main()
