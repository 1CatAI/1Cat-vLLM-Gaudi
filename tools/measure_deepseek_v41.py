# SPDX-License-Identifier: Apache-2.0
"""Measure new V4.1 candidates through real, unprofiled streaming requests."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def quantile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    low = int(location)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--server-process", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18161")
    parser.add_argument("--prompt", default="What is 17 plus 28? Explain the calculation.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--discard-intervals", type=int, default=10)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 <= args.discard_intervals < args.tokens - 1:
        parser.error("The protocol must leave at least one complete token interval per round")
    args.output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("request_deepseek_v41.py")
    record = {"server_process": str(args.server_process.resolve()),
              "protocol": {"rounds": args.rounds, "tokens": args.tokens,
                           "discard_first_intervals": args.discard_intervals, "profiler": False},
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rounds": [],
              "units": "ms", "note": "API token intervals include zeros when DSpark emits a prefix in one chunk. "
              "Chunk intervals are also retained; hardware time is not inferred from API timing."}
    for index in range(args.rounds):
        directory = args.output / f"round{index + 1}"
        command = [sys.executable, str(runner), str(directory), "--url", args.url,
                   "--tokens", str(args.tokens), "--prompt", args.prompt,
                   "--server-process", str(args.server_process.resolve())]
        with (args.output / f"round{index + 1}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            record["failed_round"] = index + 1
            (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError(f"Candidate request failed; original output remains in {directory}")
        timing = json.loads((directory / "timing.json").read_text())
        if timing["error"] is not None or timing["token_count"] != args.tokens:
            raise RuntimeError("Incomplete request cannot qualify an ITL round")
        intervals = timing["api_itl_ms"][args.discard_intervals:]
        chunks = timing["chunk_arrival_ms"]
        row = {"round": index + 1, "request_ms": timing["request_ms"], "ttft_ms": timing["ttft_ms"],
               "retained_intervals": len(intervals), "mean_itl_ms": statistics.mean(intervals),
               "p50_itl_ms": quantile(intervals, .5), "p90_itl_ms": quantile(intervals, .9),
               "p99_itl_ms": quantile(intervals, .99), "max_itl_ms": max(intervals),
               "zero_intervals": sum(value == 0 for value in intervals), "output_chunks": len(chunks),
               "chunk_intervals_ms": [right - left for left, right in zip(chunks, chunks[1:])],
               "token_ids_sha256": hashlib.sha256((directory / "token_ids.json").read_bytes()).hexdigest()}
        record["rounds"].append(row)
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if not isinstance(value, list)}), flush=True)


if __name__ == "__main__":
    main()
