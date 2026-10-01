# SPDX-License-Identifier: Apache-2.0
"""Validate raw device publication with aligned CPU scopes in two captures."""
import json
import os
from pathlib import Path
import resource
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v41_native_trace import NativeTrace, scope  # noqa: E402

root = Path(os.environ["DSV41_RUN_EVIDENCE"])
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
x = torch.randn(128, 5120, dtype=torch.bfloat16, device="hpu")
w = torch.randn(384, 5120, dtype=torch.bfloat16, device="hpu")
torch.hpu.synchronize()
reports = []
for capture in range(2):
    profiler = NativeTrace(cpu_trace_dir=root / f"capture{capture}",
                           scope_only=os.environ.get("VLLM_HPU_DSV41_RAW_SCOPE_ONLY", "0") == "1")
    profiler.start()
    for step in range(3):
        with scope(f"v41::raw_admission::capture{capture}::step{step}"):
            result = torch.relu(torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(x, w) + step).cpu()
            assert torch.isfinite(result).all()
    started = time.perf_counter()
    profiler.stop()
    reports.append(dict(capture=capture, export_s=time.perf_counter() - started,
                        maximum_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        metadata=profiler.metadata))
    print(json.dumps(reports[-1]), flush=True)
(root / "result.json").write_text(json.dumps(dict(status="passed", captures=reports), indent=2) + "\n")
