# SPDX-License-Identifier: Apache-2.0
"""Compare a new full-model candidate with preserved, unprofiled parent timings.

This produces an experiment decision, never changes production defaults. The
checks file records evidenced contracts; missing checks cannot qualify a run.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

REQUIRED_CHECKS = (
    "source_and_runtime",
    "changing_inputs_and_routing",
    "layers_and_collectives",
    "state_updates",
    "numerics",
    "dataflow",
    "memory_and_generation",
    "no_fallback",
)


def measurement(record):
    rows = record.get("results", [])
    if record.get("profile") or not record.get("qualified_measurement") or len(rows) != 3:
        raise ValueError("Require three complete, qualified, unprofiled requests")
    for row in rows:
        samples = row.get("steady_intervals_ms", [])
        if (row.get("error") or row.get("token_count") != 192 or row.get("intervals") != 181
                or row.get("discarded_intervals") != 10 or row.get("coalesced_token_events") != 0
                or row.get("timing_valid") is not True or len(samples) != 181
                or not all(math.isfinite(value) and value > 0 for value in samples)
                or not math.isclose(statistics.mean(samples), row["steady_ms"], rel_tol=1e-9)):
            raise ValueError("Invalid ITL samples; raw measurements must be retained")
    return rows


def decide(parent, candidate, checks, target_ms, minimum_gain_ms, structural_change):
    before, after = measurement(parent), measurement(candidate)
    parent_mean = statistics.mean(row["steady_ms"] for row in before)
    candidate_mean = statistics.mean(row["steady_ms"] for row in after)
    gain = parent_mean - candidate_mean
    required = (*REQUIRED_CHECKS, "trace_mechanism") if structural_change else REQUIRED_CHECKS
    failed = [key for key in required if checks.get(key) is False]
    unknown = [key for key in required if checks.get(key) is not True and key not in failed]
    speed_pass = all(row["steady_ms"] < target_ms for row in after)
    if failed:
        decision = "archive_contract_failure"
    elif not speed_pass and gain < minimum_gain_ms:
        decision = "archive_insufficient_gain"
    elif unknown:
        decision = "pending_contract_evidence"
    elif speed_pass:
        decision = "start_full_quality_and_lifecycle_qualification"
    else:
        decision = "retain_as_next_iteration_parent"
    return {
        "decision":
        decision,
        "parent_mean_ms":
        parent_mean,
        "candidate_mean_ms":
        candidate_mean,
        "net_gain_ms":
        gain,
        "relative_latency_reduction":
        gain / parent_mean,
        "target_ms":
        target_ms,
        "minimum_gain_ms":
        minimum_gain_ms,
        "all_rounds_below_target":
        speed_pass,
        "failed_checks":
        failed,
        "unknown_checks":
        unknown,
        "rounds": [{
            key: row.get(key)
            for key in ("steady_ms", "p50_ms", "p90_ms", "p99_ms", "ttft_ms", "request_ms", "intervals",
                        "client_cpu_ms", "server_cpu_ms", "hpu_memory")
        } for row in after],
        "production_qualified":
        False
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path, help="Preserved performance/result.json")
    parser.add_argument("candidate", type=Path, help="New performance/result.json")
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-ms", type=float, default=15.)
    parser.add_argument("--minimum-gain-ms", type=float, default=.5)
    parser.add_argument("--structural-change", action="store_true")
    args = parser.parse_args()
    if args.parent.resolve() == args.candidate.resolve():
        parser.error("Parent and candidate must be distinct preserved measurements")
    if not math.isfinite(args.target_ms) or args.target_ms <= 0 or args.minimum_gain_ms < 0:
        parser.error("Require positive finite target and nonnegative minimum gain")
    records = {
        name: json.loads(path.read_text())
        for name, path in (("parent", args.parent), ("candidate", args.candidate), ("checks", args.checks))
    }
    result = decide(**records,
                    target_ms=args.target_ms,
                    minimum_gain_ms=args.minimum_gain_ms,
                    structural_change=args.structural_change)
    result["sources"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
        for name, path in (("parent", args.parent), ("candidate", args.candidate), ("checks", args.checks))
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "rounds")}, indent=2))


if __name__ == "__main__":
    main()
