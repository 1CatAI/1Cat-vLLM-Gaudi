# SPDX-License-Identifier: Apache-2.0
"""Account for device overlap and observed host work in identical windows."""
import bisect
import collections
import gzip
import json


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(spans):
    return sum(end - start for start, end in merged(spans))


def clipped(start, end, windows, ends):
    index = bisect.bisect_right(ends, start)
    while index < len(windows) and windows[index][0] < end:
        low, high = windows[index]
        begin, stop = max(start, low), min(end, high)
        if begin < stop:
            yield begin, stop
        index += 1


def device_partition(engines, windows, scale):
    events = collections.defaultdict(list)
    for bit, name in enumerate(("TPC", "MME", "DMA", "NIC")):
        for start, end in merged(engines.get(name, [])):
            events[start].append((bit, 1))
            events[end].append((bit, -1))
    counts, totals = [0] * 4, collections.Counter()
    previous = None
    for stamp, changes in sorted(events.items()):
        mask = sum(1 << bit for bit, count in enumerate(counts) if count)
        if previous is not None and mask:
            totals[mask] += stamp - previous
        for bit, delta in changes:
            counts[bit] += delta
        previous = stamp
    period = sum(end - start for start, end in windows)
    totals[0] = period - sum(totals.values())
    assert totals[0] >= -1e-5
    return {("+".join(name for bit, name in enumerate(("TPC", "MME", "DMA", "NIC")) if mask & (1 << bit))
             or "no_recorded_device_activity"): value / scale for mask, value in sorted(totals.items())}


def host_accounting(path, windows, compute, scale, *, export_intervals=False):
    ends = [end for _, end in windows]
    categories, operations = collections.defaultdict(list), collections.defaultdict(list)
    counts = collections.Counter()
    with gzip.open(path / "host.jsonl.gz", "rt") as stream:
        for line in stream:
            start, length, pid, tid, kind, name, *extra = json.loads(line)
            spans = list(clipped(start, start + length, windows, ends))
            if not spans:
                continue
            lower = name.lower()
            if "synchroniz" in lower or "wait" in lower:
                category = "observed_synchronization_or_wait"
            elif "hccl" in lower:
                category = "HCCL_host_API"
            elif "enqueue" in lower or "launch" in lower:
                category = "recipe_or_device_submission"
            elif "compile" in lower and kind != "user_annotation":
                category = "compiler_or_compiled_entry"
            elif kind == "cpu_op":
                category = "CPU_operator_dispatch"
            elif kind == "user_annotation":
                category = "semantic_scope"
            else:
                category = "other_host_runtime"
            categories[category].extend(spans)
            operations[(category, name)].extend(spans)
            counts[(category, name)] += 1
    # Collapse lane overlap once. Re-sorting millions of identical device
    # intervals for each host category obscures small captures with parse cost.
    compute = merged(compute)
    compute_us = sum(end - start for start, end in compute)
    summary = {}
    for category, spans in categories.items():
        spans = merged(spans)
        active = sum(end - start for start, end in spans)
        overlap = active + compute_us - duration(spans + compute)
        summary[category] = dict(activity_ms=active / scale, overlap_with_compute_ms=overlap / scale,
                                 outside_compute_ms=(active - overlap) / scale)
    details = [dict(category=category, name=name, observed_host_events=counts[(category, name)],
                    activity_ms=duration(spans) / scale) for (category, name), spans in operations.items()]
    details.sort(key=lambda row: -row["activity_ms"])
    result = dict(categories=summary, operations=details,
                  interpretation="Host intervals may nest and overlap across threads and categories. "
                                 "Observed wait or HCCL API time is not device communication latency; "
                                 "outside-compute activity does not establish its causal critical-path contribution.")
    if export_intervals:
        result["activity_intervals_us"] = {name: merged(spans) for name, spans in categories.items()}
    (path / "host-breakdown.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return summary
