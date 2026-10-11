# SPDX-License-Identifier: Apache-2.0
"""Bounded four-rank Prefill timeline when Synapse HwTrace cannot capture.

This diagnostic uses current-stream HPU events and host monotonic timestamps.
It reports model spans, not individual hardware kernels or utilization counters.
It is disabled unless an evidence directory is explicitly supplied.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import torch

_active = None
_completed = 0


def begin(request_id, generation, tokens, pp_rank, tp_rank):
    global _active
    if os.environ.get("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE", "0") in ("", "0") or tokens < 8192:
        return False
    prefixes = tuple(prefix.strip()
                     for prefix in os.environ.get("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_REQUEST_PREFIX", "").split(",")
                     if prefix.strip())
    if prefixes and not request_id.startswith(prefixes):
        return False
    if _active is not None:
        raise RuntimeError("A Prefill event trace already owns this worker")
    limit = int(os.environ.get("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_LIMIT", "1"))
    if _completed >= limit:
        return False
    host_only = os.environ.get("VLLM_HPU_DSV41_PREFILL_HOST_ONLY") == "1"
    anchor = None if host_only else torch.hpu.Event(enable_timing=True)
    anchor_host_before = time.perf_counter_ns()
    if anchor is not None:
        anchor.record()
    anchor_host_after = time.perf_counter_ns()
    _active = dict(request_id=request_id,
                   generation=generation,
                   tokens=tokens,
                   pp_rank=pp_rank,
                   tp_rank=tp_rank,
                   host_only=host_only,
                   anchor=anchor,
                   anchor_host_before_ns=anchor_host_before,
                   anchor_host_after_ns=anchor_host_after,
                   labels=[
                       label.strip()
                       for label in os.environ.get("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_LABELS", "").split(",")
                       if label.strip()
                   ],
                   spans=[],
                   started_ns=anchor_host_before)
    return True


@contextmanager
def span(name, layer=None, rows=None):
    trace = _active
    if trace is None or (trace["labels"] and name not in trace["labels"]):
        yield
        return
    if trace.get("host_only"):
        item = dict(name=name,
                    layer=layer,
                    rows=rows,
                    start=None,
                    end=None,
                    host_start_ns=time.perf_counter_ns(),
                    thread_cpu_start_ns=time.thread_time_ns(),
                    process_cpu_start_ns=time.process_time_ns())
        trace["spans"].append(item)
        try:
            yield
        finally:
            item.update(host_end_ns=time.perf_counter_ns(),
                        thread_cpu_end_ns=time.thread_time_ns(),
                        process_cpu_end_ns=time.process_time_ns())
        return
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    item = dict(name=name, layer=layer, rows=rows, start=start, end=end, host_start_ns=host_start)
    trace["spans"].append(item)
    try:
        yield
    finally:
        end.record()
        item["host_end_ns"] = time.perf_counter_ns()


def finish():
    global _active, _completed
    trace, _active = _active, None
    if trace is None:
        return
    anchor = trace.pop("anchor")
    rows = trace.pop("spans")
    if trace.get("host_only"):
        # Native replay owns a finite pool of device events. Host scopes must
        # not borrow from it, including for the outer measurement enclosure.
        trace["device_ms"] = None
    else:
        end = torch.hpu.Event(enable_timing=True)
        end.record()
        end.synchronize()
        trace["device_ms"] = anchor.elapsed_time(end)
    trace["finished_ns"] = time.perf_counter_ns()
    trace["schema"] = 1
    trace["clock_contract"] = ("HPU event offsets share one rank-local anchor; host timestamps use "
                               "CLOCK_MONOTONIC. Cross-rank device clocks are not assumed synchronized.")
    trace["scope"] = ("Diagnostic current-stream Prefill spans; not hardware kernels or a "
                      "non-profiled throughput qualification")
    if trace.get("host_only"):
        trace["clock_contract"] = "Host timestamps use CLOCK_MONOTONIC; no device events or device clock samples."
        trace["scope"] = ("Host-only scopes with main-thread/process CPU clocks; no device timing or synchronization. "
                          "Host waits are not hardware-kernel or bandwidth measurements.")
    # Read counters after the timed enclosure; ordinary requests never enter
    # this diagnostic. The worker resets the high-water mark after readiness.
    trace["memory"] = dict(allocated_bytes=torch.hpu.memory_allocated(),
                           peak_allocated_bytes=torch.hpu.max_memory_allocated(),
                           scope="since_last_ready_including_request_warmup; not request-isolated")
    trace["spans"] = []
    for item in rows:
        start = item.pop("start")
        stop = item.pop("end")
        if start is not None:
            item["device_start_ms"] = anchor.elapsed_time(start)
            item["device_end_ms"] = anchor.elapsed_time(stop)
            item["device_duration_ms"] = item["device_end_ms"] - item["device_start_ms"]
        trace["spans"].append(item)
    destination = os.environ["VLLM_HPU_DSV41_PREFILL_EVENT_TRACE"]
    kind = "prefill-host-trace" if trace.get("host_only") else "prefill-event-trace"
    directory = Path(os.environ["DSV41_RUN_EVIDENCE"]) / kind if destination == "1" else Path(destination)
    path = (directory / f"prefill-events-pp{trace['pp_rank']}-tp{trace['tp_rank']}-g{trace['generation']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
    _completed += 1


def abort():
    global _active
    _active = None
