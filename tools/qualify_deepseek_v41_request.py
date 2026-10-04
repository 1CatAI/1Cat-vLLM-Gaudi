# SPDX-License-Identifier: Apache-2.0
"""Replay an exact saved chat request to EOS, with engine timings and optional phase capture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import requests


METRICS = ("request_prefill_time_seconds", "request_decode_time_seconds", "request_queue_time_seconds",
           "request_inference_time_seconds", "time_to_first_token_seconds", "e2e_request_latency_seconds")


def histogram_values(body):
    values = {}
    for line in body.splitlines():
        match = re.fullmatch(r'(vllm:[\w:]+)(?:\{.*\})?\s+([-+\d.eE]+)(?:\s+\d+)?', line)
        if match:
            key, value = match.groups()
            values[key] = values.get(key, 0.) + float(value)
    return values


def streaming_intervals(events):
    """Only individual token arrivals establish client inter-token latency."""
    if len(events) < 2 or any(event["count"] != 1 for event in events):
        return {"valid": False, "reason": "coalesced token chunks or fewer than two arrivals"}
    values = [(right["arrival_s"] - left["arrival_s"]) * 1000
              for left, right in zip(events, events[1:])]

    def summarize(samples):
        ordered = sorted(samples)

        def quantile(fraction):
            at = (len(ordered) - 1) * fraction
            low = int(at)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (at - low)

        return dict(count=len(samples), mean_ms=statistics.mean(samples), p50_ms=quantile(.5),
                    p95_ms=quantile(.95), p99_ms=quantile(.99))

    return {"valid": True, "all": summarize(values),
            "after_first_10_intervals": summarize(values[10:]) if len(values) > 10 else None,
            "samples_ms": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18444")
    parser.add_argument("--api-key-file", type=Path, help="Authenticate without putting a key on the command line")
    parser.add_argument("--profile", choices=("prefill", "decode"))
    parser.add_argument("--profile-start-before-request", action="store_true",
                        help="Initialize decode capture while idle; includes prefill in the raw capture")
    parser.add_argument("--decode-trace-tokens", type=int, default=128)
    parser.add_argument("--decode-trace-skip-tokens", type=int, default=0,
                        help="Let initial decode compilation finish before opening the requested capture")
    parser.add_argument("--expected-prompt-tokens", type=int, default=16384)
    parser.add_argument("--eos-token-id", type=int, default=1)
    parser.add_argument("--prefill-only", action="store_true",
                        help="Measure a max_tokens=1 request; does not qualify natural EOS or semantics")
    args = parser.parse_args()
    if args.profile_start_before_request and args.profile != "decode":
        parser.error("--profile-start-before-request requires --profile decode")
    body = json.loads(args.request.read_text())
    if args.prefill_only and (body.get("max_tokens") != 1 or args.profile == "decode"):
        raise ValueError("Prefill-only timing requires max_tokens=1 and excludes decode capture")
    if body.get("ignore_eos") or not body.get("stream") or not body.get("return_token_ids"):
        raise ValueError("Qualification requires natural EOS, streaming and returned token IDs")
    if body.get("stop") or body.get("stop_token_ids"):
        raise ValueError("This cohort measures EOS; custom stop conditions would confound it")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    session = requests.Session()
    if args.api_key_file:
        session.headers["Authorization"] = "Bearer " + args.api_key_file.read_text().strip()
    profile_records = []

    def profile(action):
        stamp = {"action": action, "start_ns": time.time_ns()}
        response = session.post(args.url + "/" + action + "_profile", timeout=(10, 1800))
        response.raise_for_status()
        stamp.update(end_ns=time.time_ns(), status=response.status_code)
        profile_records.append(stamp)
        (args.output / "profile.json").write_text(json.dumps(profile_records, indent=2) + "\n")

    def metrics(label):
        response = session.get(args.url + "/metrics", timeout=30)
        response.raise_for_status()
        (args.output / f"metrics-{label}.txt").write_text(response.text)
        return histogram_values(response.text)

    before = metrics("before")
    if args.profile == "prefill" or args.profile_start_before_request:
        profile("start")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "qualification": "prefill_only" if args.prefill_only else "natural_eos",
              "profile": args.profile, "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
              "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stream_chunk_size": 65536,
              "decode_trace_skip_tokens": args.decode_trace_skip_tokens,
              "decode_trace_tokens": args.decode_trace_tokens,
              "profile_start_before_request": args.profile_start_before_request,
              "timer": "engine histograms are scheduling-to-first-token and first-to-last-token; SSE is client arrival"}
    events, ids, content, reasoning, usage = [], [], [], [], None
    returned_prompt = None
    finish_reason = stop_reason = error = None
    controller = ThreadPoolExecutor(max_workers=1)
    pending = None
    jobs = []
    stop_submitted = False
    start_submitted = args.profile == "prefill" or args.profile_start_before_request
    start_ns, started = time.time_ns(), time.perf_counter()
    report["request_start_ns"] = start_ns
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    first_token_s = last_token_s = first_content_s = response_end_s = None
    saw_done = False
    try:
        with session.post(args.url + "/v1/chat/completions", json=body, stream=True,
                          timeout=(30, 7200)) as response, (args.output / "response.sse").open("wb") as raw:
            response.raise_for_status()
            # urllib3 preserves HTTP chunk boundaries. A byte-at-a-time
            # iterator needlessly delays the initial 16K prompt-ID event and
            # makes subsequent token arrivals appear as a buffered burst.
            for line in response.iter_lines(chunk_size=65536):
                elapsed = time.perf_counter() - started
                raw.write(line + b"\n")
                if not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    saw_done = True
                    response_end_s = time.perf_counter() - started
                    break
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("prompt_token_ids") is not None:
                    returned_prompt = item["prompt_token_ids"]
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    if delta.get("content") and first_content_s is None:
                        first_content_s = elapsed
                    reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                    current_ids = choice.get("token_ids") or []
                    if current_ids:
                        ids.extend(current_ids)
                        first_token_s = elapsed if first_token_s is None else first_token_s
                        last_token_s = elapsed
                        events.append({"arrival_s": elapsed, "count": len(current_ids), "total": len(ids)})
                        if len(ids) % 32 == 0:
                            raw.flush()
                        if args.profile and not start_submitted and len(ids) >= args.decode_trace_skip_tokens + 1:
                            pending = controller.submit(profile, "start")
                            jobs.append(pending)
                            start_submitted = True
                        stop_now = (args.profile == "prefill" or
                                    args.profile == "decode" and
                                    len(ids) >= args.decode_trace_skip_tokens + args.decode_trace_tokens + 1)
                        if args.profile and stop_now and not stop_submitted:
                            # One controller preserves start-before-stop order;
                            # streaming continues during profiler export.
                            pending = controller.submit(profile, "stop")
                            jobs.append(pending)
                            stop_submitted = True
                    if choice.get("finish_reason"):
                        finish_reason, stop_reason = choice["finish_reason"], choice.get("stop_reason")
        if args.profile and start_submitted and not stop_submitted:
            pending = controller.submit(profile, "stop")
            jobs.append(pending)
            stop_submitted = True
        for job in jobs:
            job.result()
        after = metrics("after")
        # Engine statistics export periodically. Wait for this B1 request's
        # completion sample, without sending any additional generation.
        deadline = time.monotonic() + 30
        key = "vllm:request_decode_time_seconds_count"
        while after.get(key, 0) - before.get(key, 0) < 1 and time.monotonic() < deadline:
            time.sleep(1)
            after = metrics("after")
        durations = {}
        for name in METRICS:
            prefix = "vllm:" + name
            count = after.get(prefix + "_count", 0) - before.get(prefix + "_count", 0)
            durations[name] = {"samples": count,
                               "seconds": after.get(prefix + "_sum", 0) - before.get(prefix + "_sum", 0)}
        report.update(usage=usage, finish_reason=finish_reason, stop_reason=stop_reason, tokens=len(ids),
                      client_ttft_s=first_token_s, client_last_token_s=last_token_s,
                      client_first_content_s=first_content_s, client_total_s=response_end_s,
                      client_inter_token_latency=streaming_intervals(events),
                      engine=durations, request_start_ns=start_ns)
        assert saw_done, "stream ended without the completion sentinel"
        prompt_count = usage["prompt_tokens"] if usage else len(returned_prompt or [])
        assert prompt_count == args.expected_prompt_tokens, ("prompt length", prompt_count)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if usage else 0
        assert not cached, ("prefix reuse invalidates the prefill timing", cached)
        expected_ids_path = args.request.with_name(args.request.name.replace(".request.json", ".token_ids.json"))
        if expected_ids_path != args.request and expected_ids_path.exists() and returned_prompt is not None:
            assert returned_prompt == json.loads(expected_ids_path.read_text()), (
                "server prompt IDs differ from frozen input")

        assert usage and usage["completion_tokens"] == len(ids), ("token accounting", usage, len(ids))
        if args.prefill_only:
            assert len(ids) == 1 and finish_reason in ("length", "stop"), (len(ids), finish_reason)
        else:
            assert finish_reason == "stop", ("did not finish naturally", finish_reason, stop_reason)
            assert stop_reason in (None, args.eos_token_id), ("unexpected stop", stop_reason)
        # vLLM may remove the EOS itself from streamed token IDs. With no
        # custom stop condition, finish=stop plus the configured EOS suffices;
        # distinguish that proof from directly observing EOS in returned IDs.
        report["eos_proof"] = (None
                               if args.prefill_only else "returned EOS token" if ids and ids[-1] == args.eos_token_id
                               else "engine stop with ignore_eos=false and no custom stop conditions")
        measured = (("request_prefill_time_seconds", ) if args.prefill_only else
                    ("request_prefill_time_seconds", "request_decode_time_seconds"))
        for name in measured:
            assert durations[name]["samples"] == 1, ("non-isolated or absent engine timing", name, durations[name])
        prefill_s = durations["request_prefill_time_seconds"]["seconds"]
        decode_s = durations["request_decode_time_seconds"]["seconds"]
        report.update(prefill_tokens_per_s=prompt_count / prefill_s,
                      decode_tokens_per_s=(len(ids) - 1) / decode_s if len(ids) > 1 and decode_s > 0 else None,
                      decode_ms_per_token=1000 * decode_s / (len(ids) - 1) if len(ids) > 1 else None,
                      status="passed")
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=repr(exc), finish_reason=finish_reason, stop_reason=stop_reason,
                      usage=usage, tokens=len(ids))
    finally:
        if args.profile and start_submitted and not stop_submitted:
            try:
                controller.submit(profile, "stop").result()
            except Exception as exc:
                report["profile_stop_error"] = repr(exc)
        controller.shutdown(wait=True)
        report["client_total_s"] = response_end_s
        report["harness_total_s"] = time.perf_counter() - started
        report["sse_events"] = events
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (args.output / "output.txt").write_text("".join(content))
        (args.output / "reasoning.txt").write_text("".join(reasoning))
        (args.output / "token_ids.json").write_text(json.dumps(ids) + "\n")
        (args.output / "prompt_token_ids.json").write_text(json.dumps(returned_prompt) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "sse_events"}, ensure_ascii=False), flush=True)
        session.close()
    if error:
        raise error


if __name__ == "__main__":
    main()
