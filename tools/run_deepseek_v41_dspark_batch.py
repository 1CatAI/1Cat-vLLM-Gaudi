# SPDX-License-Identifier: Apache-2.0
"""One accumulated DSpark serving comparison and its accompanying decode trace."""
import argparse
import csv
import io
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def select_batch(ledger, baseline):
    candidates = [c for c in ledger if c.get("micro_passed") and c.get("real_inputs_passed")
                  and c.get("baseline_id") == baseline["request_id"] and not c.get("end_to_end_tested")]
    candidates = sorted(candidates, key=lambda c: c["saved_ms_per_round"], reverse=True)[:5]
    switches = {c["switch"] for c in candidates}
    if len(switches) != len(candidates):
        raise ValueError("Alternative implementations cannot be accumulated under the same switch")
    expected = sum(c["saved_ms_per_round"] for c in candidates)
    if expected < 1.0 and len(candidates) < 3:
        raise ValueError("Wait for >=1 ms/round or 3-5 qualified candidates")
    if any(c["saved_ms_per_round"] < 0.3 for c in candidates):
        raise ValueError("Below-threshold candidates do not enter the batch")
    return candidates, expected


def summarize_rounds(records, request_id, output_tokens):
    matching = {r["request_id"] for r in records
                if r["request_id"] == request_id or r["request_id"].startswith(request_id + "-")}
    if len(matching) != 1:
        raise ValueError("SSE request must resolve to exactly one engine request")
    engine_id = matching.pop()
    rows = [r for r in records if r["request_id"] == engine_id and r.get("proposed_count") == 5
            and r.get("target_count") == 6 and r.get("ring_released")]
    if not rows:
        raise ValueError("No completed C6 rounds for the scored request")
    durations = [(r["end_ns"] - r["start_ns"]) / 1e6 for r in rows]
    histogram = {str(n): sum(r["committed"] - 1 == n for r in rows) for n in range(6)}
    return dict(mean_round_ms=statistics.fmean(durations), median_round_ms=statistics.median(durations),
                rounds=len(rows), useful_tokens_per_round=(output_tokens - 1) / len(rows),
                accepted_count_histogram=histogram,
                acceptance_rate=sum(r["committed"] - 1 for r in rows) / (5 * len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--service-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18579")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Permit other-card model load; do not qualify a formal TPOT")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    ledger = json.loads(args.ledger.read_text())
    candidates, expected = select_batch(ledger.get("pending", []) if isinstance(ledger, dict) else ledger, baseline)
    request = json.loads(args.request.read_text())
    if (request.get("temperature"), request.get("top_p"), request.get("seed")) != (1.0, 0.95, 42):
        raise ValueError("Batch qualification requires official sampling and seed42")
    if request.get("ignore_eos") or request.get("stop"):
        raise ValueError("Batch qualification requires natural EOS")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(candidates=candidates, expected_saved_ms_per_round=expected,
                baseline_id=baseline["request_id"], baseline_mean_round_ms=baseline["mean_round_ms"],
                request_sha256=hashlib.sha256(args.request.read_bytes()).hexdigest(),
                enabled_switches={c["switch"]: "1" for c in candidates})
    (args.output / "batch-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        return
    process = json.loads((args.service_run / "process.json").read_text())
    if process.get("exit_code") is not None:
        raise ValueError("Service has exited")
    for switch in plan["enabled_switches"]:
        if process["environment"].get(switch) != "1":
            raise ValueError(f"Frozen service does not enable {switch}")
    workspace = Path(__file__).resolve().parents[1]
    # Keep the scored request unprofiled; capture once in the companion request
    # using the identical frozen service and sample, outside the scored metric.
    for name, trace in (("formal", False), ("trace", True)):
        load = subprocess.check_output(["hl-smi", "-Q", "index,module_id,memory.used,utilization.aip",
                                        "-f", "csv"], text=True)
        (args.output / (name + "-load.csv")).write_text(load)
        owned_modules = {item["module"] for item in process["modules"]}
        other_model_load = [row for row in csv.reader(io.StringIO(load)) if len(row) >= 3
                            and row[1].strip().isdigit() and int(row[1]) not in owned_modules
                            and float(row[2].strip().split()[0]) > 1024]
        if not trace and other_model_load and not args.diagnostic:
            raise RuntimeError("Other-card models are active; coordinate exclusivity before formal TPOT")
        command = [sys.executable, "tools/qualify_deepseek_v41_request.py", str(args.request.resolve()),
                   str((args.output / name).resolve()), "--url", args.url]
        if trace:
            command += ["--profile", "decode", "--decode-trace-skip-tokens", "128", "--decode-trace-tokens", "64"]
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not trace:
            result = json.loads((args.output / name / "result.json").read_text())
            rounds = json.loads((args.service_run / "round-timing/rank0.json").read_text())["records"]
            response = (args.output / name / "response.sse").open()
            with response:
                request_id = next(json.loads(line[6:])["id"] for line in response
                                  if line.startswith("data: {") and "\"id\"" in line)
            scored = summarize_rounds(rounds, request_id, result["usage"]["completion_tokens"])
            actual = baseline["mean_round_ms"] - scored["mean_round_ms"]
            scored.update(saved_ms_per_round=actual, expected_saved_ms_per_round=expected,
                          half_prediction_met=actual >= expected / 2,
                          tpot_ms=result["decode_ms_per_token"], prefill_tokens_per_s=result["prefill_tokens_per_s"],
                          final_formal_qualification=not args.diagnostic and not other_model_load,
                          default_promotion_pending_output_quality=True)
            (args.output / "batch-result.json").write_text(json.dumps(scored, indent=2) + "\n")


if __name__ == "__main__":
    main()
