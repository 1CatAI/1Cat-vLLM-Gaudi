# SPDX-License-Identifier: Apache-2.0
"""Qualify only the source candidate against saved reference outputs."""

import argparse
import json
from pathlib import Path
import statistics
import time
import traceback

import requests


def run_request(url, model, tokens, timeout=240, drop_intervals=10, prompt="Hello"):
    if tokens < drop_intervals + 2:
        raise ValueError("Request is too short for the selected interval warmup")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    started = time.perf_counter_ns()
    events = []
    with requests.post(f"{url}/v1/completions", json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            body = json.loads(line[6:])
            for choice in body.get("choices", []):
                events.append(
                    {
                        "received_ns": time.perf_counter_ns(),
                        "text": choice.get("text", ""),
                        "token_ids": choice.get("token_ids"),
                        "finish_reason": choice.get("finish_reason"),
                        "request_id": body.get("id"),
                    }
                )
    arrivals = [e for e in events if e["token_ids"]]
    if len(arrivals) != tokens or any(len(e["token_ids"]) != 1 for e in arrivals):
        raise RuntimeError(f"Expected {tokens} separate token arrivals, got {len(arrivals)}")
    intervals = [(right["received_ns"] - left["received_ns"]) / 1e6 for left, right in zip(arrivals, arrivals[1:])]
    return {
        "payload": payload,
        "started_ns": started,
        "events": events,
        "text": "".join(e["text"] for e in events),
        "token_ids": [e["token_ids"][0] for e in arrivals],
        "after10_ms": statistics.mean(intervals[10:]) if len(intervals) > 10 else None,
        "steady_ms": statistics.mean(intervals[drop_intervals:]),
        "warmup_intervals_dropped": drop_intervals,
        "last64_ms": statistics.mean(intervals[-64:]),
        "itl_ms": intervals,
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--quality-references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "baseline_rerun": False, "runs": []}
    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    try:
        reference = json.loads(args.reference.read_text())
        deadline = time.monotonic() + args.ready_timeout
        session = requests.Session()
        session.trust_env = False
        while time.monotonic() < deadline:
            try:
                if session.get(args.url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        else:
            raise TimeoutError("Candidate service did not become ready")
        warm = run_request(args.url, args.model, 32, timeout=1200)
        (args.output / "warm.json").write_text(json.dumps(warm) + "\n")
        if warm["token_ids"] != reference["token_ids"][:32]:
            raise RuntimeError("Candidate warmup differs from the saved token reference")
        for index in range(3):
            row = run_request(args.url, args.model, 192, timeout=600)
            row["tokens_exact"] = row["token_ids"] == reference["token_ids"]
            (args.output / f"candidate-{index}.json").write_text(json.dumps(row) + "\n")
            summary["runs"].append({key: row[key] for key in ("steady_ms", "last64_ms", "tokens_exact")})
            save()
            print(json.dumps(summary["runs"][-1]), flush=True)
            if not row["tokens_exact"]:
                raise RuntimeError("Candidate output differs from the saved token reference")
        summary["steady_ms"] = statistics.mean(row["steady_ms"] for row in summary["runs"])
        summary["tokens_per_second"] = 1000 / summary["steady_ms"]
        summary["reference_ms"] = reference["steady_ms"]
        summary["quality"] = []
        if args.quality_references:
            references = sorted(args.quality_references.glob("reference-quality-*.json"))
            if not references:
                raise ValueError("No saved quality cohort found")
            for index, path in enumerate(references):
                saved = json.loads(path.read_text())
                row = run_request(args.url, args.model, len(saved["token_ids"]), timeout=600,
                                  prompt=saved["payload"]["prompt"])
                row["tokens_exact"] = row["token_ids"] == saved["token_ids"]
                (args.output / f"quality-{index}.json").write_text(json.dumps(row) + "\n")
                summary["quality"].append({"reference": path.name, "tokens_exact": row["tokens_exact"]})
                save()
                if not row["tokens_exact"]:
                    raise RuntimeError("Candidate quality output differs from its saved reference")
        summary["status"] = "qualified_token_cohort"
    except Exception:
        summary.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
