# SPDX-License-Identifier: Apache-2.0
"""Prove the selected SDK exports HPU kernels before collecting a model trace."""
import json
import os
from pathlib import Path


def main():
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    x = torch.randn(128, 5120, dtype=torch.bfloat16, device="hpu")
    w = torch.randn(384, 5120, dtype=torch.bfloat16, device="hpu")
    op = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2
    result = op(x, w)
    result.cpu()
    import ctypes
    from vllm_gaudi.ops.deepseek_v41_native_trace import _api
    api = _api()
    api.synDeviceGetMemoryInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
                                          ctypes.POINTER(ctypes.c_uint64)]

    required = ctypes.c_uint32()
    assert api.synProfilerQueryRequiredMemory(0, ctypes.byref(required)) == 0

    def memory():
        free, total = ctypes.c_uint64(), ctypes.c_uint64()
        assert api.synDeviceGetMemoryInfo(0, ctypes.byref(free), ctypes.byref(total)) == 0
        return dict(free=free.value, total=total.value, trace_required=required.value,
                    torch_allocated=torch.hpu.memory_allocated())

    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.HPU], record_shapes=True)
    reports = []
    for capture in range(2):
        before = memory()
        profiler.start()
        during = memory()
        for _ in range(2):
            with torch.profiler.record_function("v41::profiler_admission"):
                result = op(x, w)
                result.cpu()
        profiler.stop()
        destination = root / f"profiler-admission-{capture}.trace.json"
        profiler.export_chrome_trace(str(destination))
        trace = json.loads(destination.read_text())
        kernels = [event for event in trace["traceEvents"]
                   if event.get("ph") == "X" and any(kind in (event.get("args") or {}).get("HW event name", "")
                                                     for kind in ("MME", "TPC"))]
        assert kernels, "Profiler exported no HPU compute events"
        assert any(event["name"] == "v41::profiler_admission" for event in trace["traceEvents"])
        reports.append({"hardware_compute_events": len(kernels), "example": kernels[0],
                        "base_time_nanoseconds": trace.get("baseTimeNanoseconds"), "before": before,
                        "during": during, "after": memory(), "trace": str(destination)})
    report = {"status": "passed", "captures": reports}
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
