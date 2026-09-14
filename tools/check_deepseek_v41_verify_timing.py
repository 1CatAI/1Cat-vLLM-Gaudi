# SPDX-License-Identifier: Apache-2.0
"""Check timing marker ordering and host completion without loading weights."""

import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_verify_timing import VerifyPhaseTiming  # noqa: E402

torch.hpu.set_device(0)
bridge = _load_bridge(Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"]))
# This standalone diagnostic only exercises events; no communicator is used.
torch._vllm_gaudi_tp2_fused_ar_norm_runtime = (bridge, None, None)
timing = VerifyPhaseTiming(Path(os.environ["DSV41_RUN_EVIDENCE"]) / "verify-phases", 0)
timing.calibrate()
value = torch.arange(8, dtype=torch.int64, device="hpu")
host = torch.empty(8, dtype=torch.int64, device="cpu").pin_memory("hpu")
torch.hpu.synchronize()
stream = torch.hpu.Stream()
with torch.hpu.stream(stream):
    timing.begin("marker-check", 1, 6, 5)
    value.add_(10)
    timing.device("compute_done")
    host.copy_(value, non_blocking=True)
    timing.device("d2h_done")
    event = torch.hpu.Event()
    event.record(stream)
    timing.synchronize(event)
    timing.finish(1, 1)
timing.flush()
assert host.tolist() == list(range(10, 18))
row = timing.records[0]
assert all(x["status"] == "complete" for x in row["device"].values()), row
stamps = [x["device_timestamp_ns"] for x in row["device"].values()]
assert stamps == sorted(stamps), row
cal = timing.calibration
last_low = stamps[-1] + cal["offset_low_ns"]
first_high = stamps[0] + cal["offset_high_ns"]
assert last_low <= row["host"]["event_wait_done_ns"]
assert first_high >= row["host"]["round_start"]
print(json.dumps({
    "passed": True,
    "calibration_width_ms": (cal["offset_high_ns"] - cal["offset_low_ns"]) / 1e6,
    "record": row
}),
      flush=True)
