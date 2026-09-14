# SPDX-License-Identifier: Apache-2.0
"""Check isolated profiler configuration produces real TPC/MME events."""

import collections
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge  # noqa: E402

torch.hpu.set_device(0)
_load_bridge(Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"]))
output = Path(os.environ["DSV41_RUN_EVIDENCE"])
value = torch.ones((128, 512), device="hpu", dtype=torch.bfloat16)
weight = torch.ones((512, 128), device="hpu", dtype=torch.bfloat16)
torch.hpu.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                            record_shapes=True) as profiler:
    for step in range(2):
        result = torch.relu(value @ weight + step)
    torch.hpu.synchronize()
profiler.export_chrome_trace(str(output / "trace.json"))
events = json.loads((output / "trace.json").read_text())["traceEvents"]
kinds = collections.Counter((x.get("args") or {}).get("HW event name", "") for x in events)
measured = {
    name: count
    for name, count in kinds.items()
    if "TPC_SPU_START_TO_SPU_HALT" in name or "MMEH_WB" in name or "DBG_DMA_TRC_WR_DATA_LAST" in name
}
(output / "event-counts.json").write_text(json.dumps(dict(events=len(events), hardware=measured), indent=2) + "\n")
print(json.dumps(dict(events=len(events), hardware=measured)), flush=True)
assert measured, "The configuration did not collect hardware execution events"
