# SPDX-License-Identifier: Apache-2.0
"""Opt-in verification diagnostics; never used to qualify default latency."""

import json
from pathlib import Path
import time

import torch


class VerifyPhaseTiming:
    def __init__(self, directory, rank, *, max_records=64):
        self.directory, self.rank = Path(directory), rank
        self.bridge = getattr(torch, "_vllm_gaudi_tp2_fused_ar_norm_runtime")[0]
        if not hasattr(self.bridge, "verify_phase_marker"):
            raise RuntimeError("Verify timing requires the diagnostic bridge build")
        self.max_records = max_records
        self.active = None
        self.records = []
        self.calibrations = []
        self.pending_markers = []

    def calibrate(self):
        # Before serving only. Preserve bounds instead of assuming device and
        # host clocks have the same epoch, even on one physical host.
        for _ in range(5):
            before = time.perf_counter_ns()
            event = self.bridge.verify_phase_marker()
            stamp = event.snapshot(wait=True)
            after = time.perf_counter_ns()
            if stamp["status"] != "complete":
                raise RuntimeError(f"Verify clock calibration failed: {stamp}")
            device = stamp["device_timestamp_ns"]
            self.calibrations.append(dict(host_before_ns=before, host_after_ns=after,
                                          offset_low_ns=before - device, offset_high_ns=after - device,
                                          marker=stamp))
        self.calibration = min(self.calibrations, key=lambda x: x["offset_high_ns"] - x["offset_low_ns"])

    def begin(self, request_id, generation, count, proposed):
        if self.active is not None:
            raise RuntimeError("Previous verify diagnostic has not been consumed")
        if count != 6 or proposed != 5 or len(self.records) >= self.max_records:
            return
        self.active = dict(request_id=request_id, generation=generation, count=count,
                           proposed=proposed, host={}, markers={})
        # Diagnostic only: bound Kineto's wall-clock epoch against the
        # monotonic completion markers without synchronizing the device.
        anchor = f"v41::clock_anchor::rank{self.rank}::generation{generation}"
        before = time.perf_counter_ns()
        with torch.profiler.record_function(anchor):
            pass
        after = time.perf_counter_ns()
        self.active["trace_anchor"] = dict(name=anchor, host_before_ns=before,
                                           host_after_ns=after)
        self.host("round_start")
        self.device("round_start")

    def host(self, name, timestamp=None):
        if self.active is not None:
            self.active["host"][name] = time.perf_counter_ns() if timestamp is None else timestamp

    def device(self, name):
        if self.active is not None:
            self.active["markers"][name] = self.bridge.verify_phase_marker()

    def synchronize(self, event):
        if self.active is None:
            event.synchronize()
            return
        # Exact original HPUEvent::synchronize order, with separate host
        # timestamps for hardware waiting and the subsequent callback drain.
        self.active["host"].update(self.bridge.verify_synchronize_phases(event.hpu_event))

    def finish(self, committed, output_count):
        if self.active is None:
            return
        self.host("transaction_consumed")
        row, self.active = self.active, None
        row["committed"], row["output_count"] = committed, output_count
        row["device"] = {}
        for name, event in row.pop("markers").items():
            stamp = event.snapshot()
            row["device"][name] = stamp
            if stamp["status"] != "complete":
                # Keep the handle alive until a later nonblocking snapshot.
                # Never wait just to finish diagnostic serialization.
                self.pending_markers.append((row, name, event))
        self.records.append(row)

    def flush(self):
        pending = []
        for row, name, event in self.pending_markers:
            stamp = event.snapshot()
            row["device"][name] = stamp
            if stamp["status"] != "complete":
                pending.append((row, name, event))
        self.pending_markers = pending
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"rank{self.rank}-verify-phases.json"
        path.write_text(json.dumps(dict(
            schema=1, rank=self.rank, units="ns", clock="CLOCK_MONOTONIC",
            purpose="Diagnostic markers; not an uninstrumented performance qualification",
            device_scope="physical compute/current stream, no inserted waits",
            calibrations=self.calibrations, calibration=self.calibration,
            records=self.records), indent=2) + "\n")
