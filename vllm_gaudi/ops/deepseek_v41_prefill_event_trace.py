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


def enabled(tokens):
    if not os.environ.get("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE") or tokens < 8192:
        return False
    limit = int(os.environ.get("VLLM_HPU_DSV41_PREFILL_EVENT_TRACE_LIMIT", "1"))
    return _completed < limit


def begin(request_id, generation, tokens, pp_rank, tp_rank):
    global _active
    if not enabled(tokens):
        return False
    if _active is not None:
        raise RuntimeError("A Prefill event trace already owns this worker")
    anchor = torch.hpu.Event(enable_timing=True)
    anchor_host_before = time.perf_counter_ns()
    anchor.record()
    anchor_host_after = time.perf_counter_ns()
    _active = dict(request_id=request_id,
                   generation=generation,
                   tokens=tokens,
                   pp_rank=pp_rank,
                   tp_rank=tp_rank,
                   anchor=anchor,
                   anchor_host_before_ns=anchor_host_before,
                   anchor_host_after_ns=anchor_host_after,
                   spans=[],
                   started_ns=anchor_host_before)
    return True


@contextmanager
def span(name, layer=None, rows=None):
    trace = _active
    if trace is None:
        yield
        return
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    item = dict(name=name, layer=layer, rows=rows, start=start, end=end,
                host_start_ns=host_start)
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
    end = torch.hpu.Event(enable_timing=True)
    end.record()
    end.synchronize()
    anchor = trace.pop("anchor")
    rows = trace.pop("spans")
    trace["device_ms"] = anchor.elapsed_time(end)
    trace["finished_ns"] = time.perf_counter_ns()
    trace["schema"] = 1
    trace["clock_contract"] = ("HPU event offsets share one rank-local anchor; host timestamps use "
                               "CLOCK_MONOTONIC. Cross-rank device clocks are not assumed synchronized.")
    trace["scope"] = ("Diagnostic current-stream Prefill spans; not hardware kernels or a "
                      "non-profiled throughput qualification")
    trace["spans"] = []
    for item in rows:
        start = item.pop("start")
        stop = item.pop("end")
        item["device_start_ms"] = anchor.elapsed_time(start)
        item["device_end_ms"] = anchor.elapsed_time(stop)
        item["device_duration_ms"] = item["device_end_ms"] - item["device_start_ms"]
        trace["spans"].append(item)
    destination = os.environ["VLLM_HPU_DSV41_PREFILL_EVENT_TRACE"]
    directory = (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "prefill-event-trace"
                 if destination == "1" else Path(destination))
    path = (directory /
            f"prefill-events-pp{trace['pp_rank']}-tp{trace['tp_rank']}-g{trace['generation']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
    _completed += 1


def abort():
    global _active
    _active = None
