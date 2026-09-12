# SPDX-License-Identifier: Apache-2.0
"""Wait for the owned ordinary C1 service, then measure performance and trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18162")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-C1")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--target-ms", type=float, default=24.)
    parser.add_argument("--trace-tokens", type=int, default=64,
                        help="Independent profiler request length; three timed requests remain 192 tokens")
    args = parser.parse_args()
    if args.trace and not 12 <= args.trace_tokens <= 192:
        parser.error("Trace request must retain a complete window after discarding ten intervals")
    run = args.run.resolve()
    process = json.loads((run / "process.json").read_text())
    source = Path(__file__).read_bytes()
    (run / "measurement-controller.py").write_bytes(source)
    status = {"controller_pid": os.getpid(), "api_pid": process["pid"],
              "source_sha256": hashlib.sha256(source).hexdigest(), "phase": "waiting_for_service"}

    def save():
        (run / "measurement-control.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)

    def invoke(name, extra):
        script = Path(__file__).with_name("benchmark_deepseek_v41_c1.py")
        with (run / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, str(script), str(run / name), "--url", args.url,
                                     "--model", args.model, "--process-record", str(run / "process.json"),
                                     *extra], stdout=log, stderr=subprocess.STDOUT)
        print((run / f"{name}.log").read_text()[-8000:], flush=True)
        result.check_returncode()

    save()
    try:
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            os.kill(process["pid"], 0)
            current = json.loads((run / "process.json").read_text())
            if "exit_code" in current:
                raise RuntimeError("Owned service exited before readiness")
            try:
                response = requests.get(args.url + "/health", timeout=2)
                if response.status_code == 200:
                    models = requests.get(args.url + "/v1/models", timeout=5).json()
                    if not any(model["id"] == args.model for model in models["data"]):
                        raise RuntimeError("Service model identity differs from this candidate")
                    (run / "service-ready.json").write_text(json.dumps({
                        "ready_at_unix": time.time(), "models": models, "api_pid": process["pid"]},
                        indent=2) + "\n")
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("Owned candidate did not become healthy before startup deadline")
        status["phase"] = "unprofiled_requests"
        save()
        invoke("performance", ["--target-ms", str(args.target_ms)])
        performance = json.loads((run / "performance/result.json").read_text())
        status["all_three_below_24_ms"] = performance["all_three_below_24_ms"]
        status["target_ms"] = performance["target_ms"]
        status["all_three_below_target_ms"] = performance["all_three_below_target_ms"]
        # Timing coalescence can invalidate ITL without invalidating the saved
        # hardware execution. A failed or incomplete request cannot advance.
        if any(row["error"] or row["token_count"] != 192 for row in performance["results"]):
            raise RuntimeError("Performance requests did not all complete; raw errors retained")
        if args.trace:
            status["phase"] = "four_rank_trace"
            save()
            invoke("trace-request", ["--profile", "--tokens", str(args.trace_tokens),
                                     "--rounds", "1", "--warmup-tokens", "0"])
        status["phase"] = "measured_pending_analysis"
        save()
    except Exception as error:
        status.update(phase="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
