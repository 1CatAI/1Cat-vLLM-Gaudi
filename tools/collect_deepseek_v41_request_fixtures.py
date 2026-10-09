# SPDX-License-Identifier: Apache-2.0
"""Collect three real sampled C6 boundaries; never score this diagnostic."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-run", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--launcher-pid", type=int)
    parser.add_argument("--url", default="http://127.0.0.1:18579")
    args = parser.parse_args()
    owner = args.service_run.with_name(args.service_run.name + "-fixture-client.json")
    report = dict(pid=os.getpid(), scope="input collection only; no speed qualification", status="waiting")
    owner.write_text(json.dumps(report, indent=2) + "\n")
    deadline = None
    while deadline is None or time.monotonic() < deadline:
        if args.launcher_pid is not None and not Path(f"/proc/{args.launcher_pid}").exists():
            raise RuntimeError("Owned fixture launcher exited before readiness; do not wait indefinitely")
        manifest = args.service_run / "process.json"
        if manifest.exists():
            process = json.loads(manifest.read_text())
            if process.get("exit_code") is not None:
                raise RuntimeError("Fixture service failed before readiness; no measurement or gain")
            if process["recipe_cache_identity"] != args.cache_key:
                raise RuntimeError("Seeded recipe identity differs from the actual launch")
            if deadline is None:
                deadline = time.monotonic() + 7200
            try:
                with urllib.request.urlopen(args.url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
        time.sleep(15)
    else:
        raise RuntimeError("Full warmup did not finish before the fixture deadline")
    request = json.loads(args.request.read_text())
    request["max_tokens"] = 32
    request.pop("max_completion_tokens", None)
    path = args.service_run / "fixture-request.json"
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    output = args.service_run / "unscored-input-request"
    command = [sys.executable, "tools/qualify_deepseek_v41_request.py", str(path), str(output), "--url", args.url]
    with (args.service_run / "fixture-request-client.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    # Length termination is expected here. It never establishes natural EOS
    # quality or a complete-round gain; the only criterion is real inputs.
    missing = [str(args.fixtures / f"rank{rank}" / f"c6-{index}.pt")
               for rank in range(4) for index in range(3)
               if not (args.fixtures / f"rank{rank}" / f"c6-{index}.pt").is_file()]
    report.update(status="captured" if not missing else "failed", request_client_exit=result.returncode,
                  missing=missing, performance_qualified=False)
    owner.write_text(json.dumps(report, indent=2) + "\n")
    # Retire exactly the owned service after the fixture request has drained.
    process = json.loads((args.service_run / "process.json").read_text())
    pid = process["pid"]
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode()
    if "vllm_gaudi.entrypoints.deepseek_v41" not in cmdline:
        raise RuntimeError("Fixture server ownership changed; do not terminate it")
    os.killpg(process["pgid"], signal.SIGTERM)
    if missing:
        raise RuntimeError("The request did not capture three actual C6 boundaries on every rank")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
