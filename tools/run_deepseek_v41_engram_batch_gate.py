# SPDX-License-Identifier: Apache-2.0
"""Fixed four-card producer/consumer gate for common device Engram batching."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(run):
    ranks = [json.loads((run / f"engram-chain-rank{rank}.json").read_text()) for rank in range(4)]
    if any(rank["status"] != "component_passed" or len(rank["checks"]) != 3 for rank in ranks):
        raise ValueError("All ranks must finish three real-input correctness checks and native A/B")
    saved = []
    for iteration in range(3):
        medians = [max(next(row["median_device_ms"] for row in rank["rounds"]
                            if row["iteration"] == iteration and row["arm"] == arm)
                       for rank in ranks) for arm in (0, 1)]
        saved.append(medians[0] - medians[1])
    result = dict(candidate="common batched device Engram", feature="VLLM_HPU_DSV41_DSPARK_BATCH_ENGRAM",
                  paired_slowest_rank_saved_ms=saved, projected_ms_per_round=statistics.median(saved),
                  micro_passed=all(value > 0 for value in saved) and statistics.median(saved) >= 0.3,
                  actual_cases=3, ranks=4, native_commands=[rank["native_commands"] for rank in ranks],
                  producer_launches_per_round=[12, 2], default_enabled=False, end_to_end_tested=False,
                  credited_end_to_end_ms=0,
                  limitation="Two actual Engram layers; component projection, not complete-round latency")
    (run / "component-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace.parent
    storage = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving")
    if not args.summarize_only:
        command = [sys.executable, str(workspace / "tools/run_deepseek_v41.py"),
                   "--devices", "4", "--modules", "2,6,7,3", "--preferred-cpus", "10-19,38-47",
                   "--cpu-conflict-policy", "relocate-or-measure", "--min-host-available-gib", "350",
                   "--lock-dir", str(root / "locks"),
                   "--secondary-lock-dir", "/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks",
                   "--runtime-profile", str(args.runtime_profile.resolve()),
                   "--engine-source", str(root / "builds/dsv41-tp4-dspark-main-v1/engine"),
                   "--recipe-cache-dir", str(storage / "engram-batch-406/chain-cache"),
                   str(args.run.resolve()), "--", sys.executable, "-m", "torch.distributed.run",
                   "--standalone", "--nproc-per-node", "4", "tools/check_deepseek_v41_engram_batch_chain.py",
                   "--prepared", "/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2",
                   "--fixtures", "/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1/"
                   "production-c6-request-fixtures-02-with-history"]
        subprocess.run(command, cwd=workspace, check=True)
    print(json.dumps(summarize(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
