# SPDX-License-Identifier: Apache-2.0
"""Check tensor/native HBM admission without loading another full checkpoint."""

import argparse
import ctypes
import json
import os
from pathlib import Path


def main():
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-peak-bytes", type=int, default=96558052864)
    parser.add_argument("--native-reserve-gib", type=float, default=4.)
    args = parser.parse_args()
    torch.hpu.set_device(0)
    result = dict(purpose="Memory admission diagnostic; no model, correctness or speed qualification",
                  tensor_target_bytes=args.torch_peak_bytes,
                  native_target_bytes=int(args.native_reserve_gib * (1 << 30)),
                  tensor_pool_percentage=os.environ.get("PT_HPU_POOL_MEM_ACQUIRE_PERC"),
                  status="running",
                  formal_qualification=False)
    chunks, address = [], ctypes.c_uint64()
    library = ctypes.CDLL("libSynapse.so")
    library.synDeviceMalloc.argtypes = [
        ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64)
    ]
    library.synDeviceMalloc.restype = ctypes.c_int
    library.synDeviceFree.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32]
    library.synDeviceFree.restype = ctypes.c_int
    try:
        remaining = args.torch_peak_bytes
        while remaining:
            count = min(remaining, 256 << 20)
            value = torch.empty(count, device="hpu", dtype=torch.uint8)
            value.zero_()  # Materialize storage without reading uninitialized contents.
            chunks.append(value)
            remaining -= count
        torch.hpu.synchronize()
        result["tensor_allocated_bytes"] = torch.hpu.memory_allocated()
        result["tensor_reserved_bytes"] = torch.hpu.memory_reserved()
        result["tensor_pool_total_bytes"] = torch.hpu.mem_get_info()[1]
        if result["tensor_allocated_bytes"] < args.torch_peak_bytes:
            raise RuntimeError("Tensor ballast did not reach the recorded model peak")
        status = library.synDeviceMalloc(0, result["native_target_bytes"], 0, 0, ctypes.byref(address))
        result["native_allocation_status"] = status
        if status != 0:
            raise RuntimeError(f"Native HBM admission failed with synStatus={status}")
        result["status"] = "passed_tensor_peak_plus_native_reserve"
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        if address.value:
            result["native_free_status"] = library.synDeviceFree(0, address.value, 0)
        chunks.clear()
        (Path(os.environ["DSV41_RUN_EVIDENCE"]) /
         "MEMORY_ADMISSION.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
