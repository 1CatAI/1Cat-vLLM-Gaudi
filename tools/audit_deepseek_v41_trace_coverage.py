# SPDX-License-Identifier: Apache-2.0
"""Check retained hardware bounds against request clocks and target annotations.

This detects missing acquisition prefixes; compatible bounds alone do not prove
that every hardware packet is present. Physical-call reconstruction remains a
separate check against the serialized recipe contracts and execution counters.
"""

import argparse
import gzip
import json
from pathlib import Path


def audit(analysis, requests):
    ranks, missing = [], False
    for rank in range(4):
        directory = analysis / f"rank{rank}"
        inventory = json.loads((directory / "inventory.json").read_text())
        base = inventory["base_time_nanoseconds"]
        if base is None:
            raise ValueError("The trace has no host-clock origin")
        first, last = inventory["first_us"], inventory["last_us"]
        phases = []
        with gzip.open(directory / "host.jsonl.gz", "rt") as stream:
            for line in stream:
                ts, dur, _, _, _, name = json.loads(line)
                if name.startswith("v41::target::"):
                    phases.append((ts, ts + dur, name))
        observations = []
        for request in requests:
            record = json.loads((request / "request.json").read_text())
            timing = json.loads((request / "timing.json").read_text())
            if not record["profile"] or timing["error"] or timing["ttft_ms"] is None:
                raise ValueError("Coverage requires successful, explicitly profiled request records")
            start = (record["request_start_unix_ns"] - base) / 1000
            end = start + timing["request_ms"] * 1000
            target = [phase for phase in phases if start <= phase[0] <= end]
            before = [phase for phase in target if phase[1] < first]
            after = [phase for phase in target if phase[0] > last]
            absent = first > start + timing["ttft_ms"] * 1000 or last < start
            incomplete = absent or bool(before) or bool(after)
            missing |= incomplete
            observations.append({"request": str(request.resolve()), "target_host_phases": len(target),
                "target_phases_before_hardware": len(before), "target_phases_after_hardware": len(after),
                "first_output_precedes_first_hardware": first > start + timing["ttft_ms"] * 1000,
                "hardware_first_from_request_start_ms": (first - start) / 1000,
                "hardware_last_from_request_end_ms": (last - end) / 1000,
                "request_ms": timing["request_ms"], "coverage_gap": incomplete})
        ranks.append({"rank": rank, "trace_sha256": inventory["trace_sha256"],
                      "hardware_window_ms": (last - first) / 1000, "requests": observations})
    return {"status": "incomplete hardware coverage" if missing else "request/target bounds consistent",
            "report_units": "ms", "ranks": ranks,
            "limitation": "Bounds are necessary but insufficient evidence of complete packet/call coverage. "
                          "Missing host annotations cannot establish per-target coverage."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("requests", type=Path, nargs="+")
    args = parser.parse_args()
    result = audit(args.analysis, args.requests)
    (args.analysis / "coverage-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
