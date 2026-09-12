# SPDX-License-Identifier: Apache-2.0
"""Measure real C1 completions using the preserved V4 request boundary."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import requests


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def process_cpu(record_path):
    if record_path is None:
        return {}
    group = json.loads(record_path.read_text())["pgid"]
    result = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if os.getpgid(int(proc.name)) != group:
                continue
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            result[proc.name] = {"cpu_ms": (int(fields[11]) + int(fields[12])) * 1000 / os.sysconf("SC_CLK_TCK"),
                                 "start_ticks": fields[19], "name": (proc / "comm").read_text().strip()}
        except (OSError, ProcessLookupError):
            continue
    return result


def score_arrivals(events, expected, discard=10):
    arrivals, ids, coalesced = [], [], 0
    for event in events:
        tokens = event["token_ids"]
        ids.extend(tokens)
        if len(tokens) > 1:
            coalesced += 1
        arrivals.extend([event["received_ns"] if len(tokens) == 1 else None] * len(tokens))
    valid = len(ids) == expected and coalesced == 0 and len(ids) > discard + 1
    result = {"token_count": len(ids), "coalesced_token_events": coalesced, "discarded_intervals": discard,
              "timing_valid": valid, "intervals": 0}
    if valid:
        intervals = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:])][discard:]
        result.update(intervals=len(intervals), steady_ms=sum(intervals) / len(intervals),
                      p50_ms=percentile(intervals, 0.5), p90_ms=percentile(intervals, 0.9),
                      p99_ms=percentile(intervals, 0.99), steady_intervals_ms=intervals)
    return result, ids


def complete(session, args, output, tokens):
    payload = {"model": args.model, "prompt": "Hello", "max_tokens": tokens,
               "temperature": 0, "ignore_eos": True, "stream": True, "return_token_ids": True,
               "stream_options": {"include_usage": True}}
    cpu_before = process_cpu(args.process_record)
    client_cpu = time.process_time_ns()
    started = time.perf_counter_ns()
    raw, events, pieces, usage, error = [], [], [], None, None
    try:
        with session.post(args.url + "/v1/completions", json=payload, stream=True,
                          timeout=(10, 1800)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                received = time.perf_counter_ns()
                if not line.startswith(b"data: "):
                    continue
                body = line[6:].decode("utf-8")
                raw.append({"received_ns": received, "data": body})
                if body == "[DONE]":
                    break
                item = json.loads(body)
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    pieces.append(choice.get("text") or "")
                    if choice.get("token_ids"):
                        events.append({"received_ns": received, "token_ids": choice["token_ids"],
                                       "text": choice.get("text"), "finish_reason": choice.get("finish_reason")})
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        error = repr(exc)
    finished = time.perf_counter_ns()
    cpu_after = process_cpu(args.process_record)
    score, ids = score_arrivals(events, tokens)
    score.update(request_ms=(finished - started) / 1e6, error=error,
                 ttft_ms=(events[0]["received_ns"] - started) / 1e6 if events else None,
                 client_cpu_ms=(time.process_time_ns() - client_cpu) / 1e6, usage=usage)
    score["server_cpu_ms"] = {pid: {"name": value["name"],
                                    "cpu_ms": value["cpu_ms"] - cpu_before[pid]["cpu_ms"]}
                              for pid, value in cpu_after.items() if pid in cpu_before
                              and value["start_ticks"] == cpu_before[pid]["start_ticks"]}
    if error:
        score["timing_valid"] = False
    output.mkdir()
    (output / "request.json").write_text(json.dumps({"payload": payload, "started_ns": started,
                                                    "finished_ns": finished}, indent=2) + "\n")
    (output / "stream.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in raw))
    (output / "events.json").write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n")
    (output / "token_ids.json").write_text(json.dumps(ids) + "\n")
    (output / "output.txt").write_text("".join(pieces))
    (output / "timing.json").write_text(json.dumps(score, indent=2) + "\n")
    print(json.dumps({"run": output.name, **{k: v for k, v in score.items()
                                            if k not in ("steady_intervals_ms", "server_cpu_ms")}}), flush=True)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18162")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-C1")
    parser.add_argument("--process-record", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--target-ms", type=float, default=24., help="Strict per-round steady ITL target")
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--profile", action="store_true", help="Profiling-only acquisition, not speed qualification")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output / "harness.py").write_bytes(source)
    (args.output / "protocol.json").write_text(json.dumps({
        "harness_sha256": hashlib.sha256(source).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "timer": "Client monotonic per-token SSE arrival; coalesced token events invalidate ITL",
        "comparison": "V4 23.755395 ms is a cross-model target; no prior valid V4.1 C1 measurement"
    }, indent=2) + "\n")
    session = requests.Session()
    results = []
    if args.warmup_tokens:
        warm = complete(session, args, args.output / "warmup", args.warmup_tokens)
        if warm["error"] or warm["token_count"] != args.warmup_tokens:
            raise RuntimeError("The real warmup request did not complete; raw result retained")
    if args.profile:
        session.post(args.url + "/start_profile", timeout=(10, 1800)).raise_for_status()
    try:
        for index in range(args.rounds):
            results.append(complete(session, args, args.output / f"round-{index}", args.tokens))
    finally:
        stop_error = None
        if args.profile:
            try:
                session.post(args.url + "/stop_profile", timeout=(10, 1800)).raise_for_status()
            except requests.RequestException as error:
                stop_error = str(error)
        session.close()
        valid = (not args.profile and args.rounds == 3 and args.tokens == 192 and len(results) == 3
                 and all(row["timing_valid"] and row["intervals"] == 181 for row in results))
        record = {"finished_at": datetime.now(timezone.utc).isoformat(), "profile": args.profile,
                  "profile_stop_error": stop_error,
                  "qualified_measurement": valid, "target_ms": args.target_ms,
                  "all_three_below_target_ms": valid and all(row["steady_ms"] < args.target_ms for row in results),
                  "all_three_below_24_ms": valid and all(
                      row["steady_ms"] < 24 for row in results), "results": results}
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        if stop_error:
            raise RuntimeError(f"Profiler export failed; request results were retained: {stop_error}")


if __name__ == "__main__":
    main()
