# SPDX-License-Identifier: Apache-2.0
"""Archive one real text/image request and its observable streaming timing."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import time

import requests


def process_snapshot(process_record):
    """Read host resource counters without synchronizing an HPU worker."""
    if process_record is None:
        return None
    import os
    import psutil
    record = json.loads(process_record.read_text())
    try:
        parent = psutil.Process(record["pid"])
        if os.getpgid(parent.pid) != record["pgid"]:
            raise RuntimeError("The recorded server process group is no longer current")
        processes = [parent, *parent.children(recursive=True)]
    except (OSError, psutil.Error) as exc:
        raise RuntimeError("The recorded server process is unavailable") from exc
    rows = []
    for process in processes:
        try:
            cpu, memory = process.cpu_times(), process.memory_info()
            rows.append({
                "pid": process.pid,
                "name": process.name(),
                "created": process.create_time(),
                "user_ms": cpu.user * 1000,
                "system_ms": cpu.system * 1000,
                "rss_bytes": memory.rss
            })
        except psutil.NoSuchProcess:
            continue
    return {"sampled_at": datetime.now(timezone.utc).isoformat(), "processes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18161")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--prompt", default="计算 17 + 28。先给出结果，再解释计算过程。")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile", action="store_true")
    profiling.add_argument("--external-profile",
                           action="store_true",
                           help="Record that the caller owns an already-active multi-request acquisition")
    parser.add_argument("--stop-at-eos", action="store_true", help="Honor EOS for generation quality cases")
    parser.add_argument("--thinking-mode",
                        choices=("default", "chat", "thinking"),
                        default="default",
                        help="Use the frozen upstream prompt encoder's chat/thinking contract")
    parser.add_argument("--server-process", type=Path, help="Launcher process.json for per-process host CPU counters")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    content = [{"type": "text", "text": args.prompt}]
    image_info = None
    if args.image is not None:
        data = args.image.read_bytes()
        mime = mimetypes.guess_type(args.image.name)[0]
        if mime is None or not mime.startswith("image/"):
            raise ValueError("The image must have a supported image extension")
        image_info = {"path": str(args.image.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
        content.insert(0, {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
            }
        })
    body = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": content
        }],
        "temperature": 0,
        "max_tokens": args.tokens,
        "ignore_eos": not args.stop_at_eos,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }
    if args.thinking_mode != "default":
        body["chat_template_kwargs"] = {"enable_thinking": args.thinking_mode == "thinking"}
    request_record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "model": args.model,
        "prompt": args.prompt,
        "image": image_info,
        "max_tokens": args.tokens,
        "temperature": 0,
        "ignore_eos": not args.stop_at_eos,
        "thinking_mode": args.thinking_mode,
        "profile": args.profile or args.external_profile,
        "profiler_control": "external" if args.external_profile else "request_tool",
        "clock": "perf_counter_ns",
        "request_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    (args.output / "request.json").write_text(json.dumps(request_record, indent=2, ensure_ascii=False) + "\n")
    session = requests.Session()
    if args.profile:
        try:
            session.post(args.url + "/start_profile", timeout=(10, 1800)).raise_for_status()
        except Exception as exc:
            (args.output / "profile.json").write_text(json.dumps({"status": "start_failed", "error": repr(exc)}) + "\n")
            session.close()
            raise
    resource_before = process_snapshot(args.server_process)
    started = time.perf_counter_ns()
    request_record.update(request_start_perf_counter_ns=started, request_start_unix_ns=time.time_ns())
    (args.output / "request.json").write_text(json.dumps(request_record, indent=2, ensure_ascii=False) + "\n")
    token_times, token_ids, pieces, reasoning, chunks = [], [], [], [], []
    usage, error = None, None
    try:
        with session.post(args.url + "/v1/chat/completions", json=body, stream=True,
                          timeout=(10, 1800)) as response, (args.output / "stream.jsonl").open("w") as raw:
            if not response.ok:
                (args.output / "http-error.txt").write_text(response.text)
                response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                received = time.perf_counter_ns()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].decode("utf-8")
                raw.write(json.dumps({"elapsed_ms": (received - started) / 1e6, "data": payload}) + "\n")
                raw.flush()
                if payload == "[DONE]":
                    break
                item = json.loads(payload)
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("usage") is not None:
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    reason = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    ids = choice.get("token_ids") or []
                    if text or reason or ids:
                        chunks.append((received - started) / 1e6)
                    pieces.append(text)
                    reasoning.append(reason)
                    token_ids.extend(ids)
                    token_times.extend([(received - started) / 1e6] * len(ids))
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        finished = time.perf_counter_ns()
        (args.output / "output.txt").write_text("".join(pieces))
        (args.output / "reasoning.txt").write_text("".join(reasoning))
        (args.output / "token_ids.json").write_text(json.dumps(token_ids) + "\n")
        report = {
            "request_ms": (finished - started) / 1e6,
            "ttft_ms":
            chunks[0] if chunks else None,
            "usage":
            usage,
            "token_count":
            len(token_ids),
            "chunk_count":
            len(chunks),
            "chunk_arrival_ms":
            chunks,
            "token_arrival_ms":
            token_times,
            "error":
            error,
            "timing_note":
            "DSpark may emit multiple tokens per chunk; arrivals measure the API, "
            "not hardware token boundaries."
        }
        if len(token_times) > 1:
            report["mean_api_itl_ms"] = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
            report["api_itl_ms"] = [b - a for a, b in zip(token_times, token_times[1:])]
        (args.output / "timing.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if resource_before is not None:
            try:
                resource_after = process_snapshot(args.server_process)
                before = {(row["pid"], row["created"]): row for row in resource_before["processes"]}
                deltas = []
                for row in resource_after["processes"]:
                    previous = before.get((row["pid"], row["created"]))
                    if previous is not None:
                        deltas.append({
                            "pid": row["pid"],
                            "name": row["name"],
                            "user_ms": row["user_ms"] - previous["user_ms"],
                            "system_ms": row["system_ms"] - previous["system_ms"]
                        })
                resources = {
                    "before":
                    resource_before,
                    "after":
                    resource_after,
                    "cpu_delta":
                    deltas,
                    "note":
                    "Host CPU time summed across process threads; snapshots bracket the API request. "
                    "RSS snapshots are not peak memory and shared mappings must not be summed."
                }
            except (OSError, RuntimeError) as exc:
                resources = {"before": resource_before, "after_error": repr(exc)}
            (args.output / "host-resources.json").write_text(json.dumps(resources, indent=2) + "\n")
        if args.profile:
            try:
                session.post(args.url + "/stop_profile", timeout=(10, 1800)).raise_for_status()
                (args.output / "profile.json").write_text(json.dumps({"status": "stopped"}) + "\n")
            except Exception as exc:
                (args.output / "profile.json"
                 ).write_text(json.dumps({
                     "status": "stop_failed",
                     "error": repr(exc),
                     "request_error": error
                 }) + "\n")
                if error is None:
                    session.close()
                    raise
        session.close()
        print(json.dumps({
            key: value
            for key, value in report.items() if not isinstance(value, list)
        },
                         ensure_ascii=False),
              flush=True)


if __name__ == "__main__":
    main()
