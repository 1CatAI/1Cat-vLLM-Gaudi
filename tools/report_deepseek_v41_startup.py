# SPDX-License-Identifier: Apache-2.0
"""Summarize one installed service startup from its process record and log."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re


def summarize(record, text):
    start = record["started_at"]
    year = datetime.fromtimestamp(start).year
    phases = {}
    weights = {}
    cache = {}
    for line in text.splitlines():
        stamp = re.search(r"(?:INFO|WARNING|ERROR) (\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if stamp is None:
            continue
        when = datetime.strptime(f"{year}-{stamp[1]}", "%Y-%m-%d %H:%M:%S").timestamp()
        if "prepared weights loaded" in line:
            phases["weights_ready"] = max(phases.get("weights_ready", start), when)
        for count in (1, 6):
            if f"C{count} native warmup searches:" in line:
                phases.setdefault(f"c{count}_start", when)
        match = re.search(r"weight startup: rank=(\d+) total_seconds=([\d.]+) families=(.*?) counters=(.*?) reader",
                          line)
        if match:
            weights[match[1]] = dict(total_seconds=float(match[2]),
                                     families=json.loads(match[3]),
                                     counters=json.loads(match[4]),
                                     counters_overlap=True)
        rank = re.search(r"Worker_TP(\d+)", line)
        if rank:
            counters = cache.setdefault(rank[1], Counter())
            for marker, name in (("lowered backend restored:", "backend_hits"), ("lowered backend published:",
                                                                                 "backend_rebuilt"),
                                 ("lowered backend not cacheable:",
                                  "backend_uncacheable"), ("lowered backend rejected:", "backend_rejected"),
                                 ("guarded frontend restored:", "frontend_hits"), ("frontend artifact rejected;",
                                                                                   "frontend_rejected"),
                                 ("frontend guard rejected:", "same_geometry_guard_rejected")):
                if marker in line:
                    counters[name] += 1
    durations = {}
    previous = start
    for marker, name in (("weights_ready", "initialization_and_weights"),
                         ("c1_start", "prefill_and_protocol_preparation"), ("c6_start", "c1_native_preparation")):
        if marker not in phases:
            break
        durations[name] = phases[marker] - previous
        previous = phases[marker]
    ready = record.get("ready_at")
    if ready is not None:
        durations["remaining_preparation"] = ready - previous
    return dict(status="ready" if ready is not None else "incomplete",
                started_at=start,
                ready_at=ready,
                elapsed_seconds=ready - start if ready is not None else None,
                stages_seconds=durations,
                phase_timestamps=phases,
                weights=weights,
                cache_by_rank=cache,
                qualified_startup=ready is not None and ready - start <= 600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    log = args.log or args.record.parent / "service.log"
    result = summarize(record, log.read_text(errors="replace"))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
