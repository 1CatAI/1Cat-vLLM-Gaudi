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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18161")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--prompt", default="计算 17 + 28。先给出结果，再解释计算过程。")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
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
        content.insert(0, {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
    body = {"model": args.model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": args.tokens, "ignore_eos": True,
            "return_token_ids": True, "stream": True, "stream_options": {"include_usage": True}}
    request_record = {"started_at": datetime.now(timezone.utc).isoformat(), "url": args.url,
                      "model": args.model, "prompt": args.prompt, "image": image_info,
                      "max_tokens": args.tokens, "temperature": 0, "ignore_eos": True,
                      "profile": args.profile, "clock": "perf_counter_ns"}
    (args.output / "request.json").write_text(json.dumps(request_record, indent=2, ensure_ascii=False) + "\n")
    session = requests.Session()
    if args.profile:
        session.post(args.url + "/start_profile", timeout=(10, 1800)).raise_for_status()
    started = time.perf_counter_ns()
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
        report = {"request_ms": (finished - started) / 1e6, "ttft_ms": chunks[0] if chunks else None,
                  "usage": usage, "token_count": len(token_ids), "chunk_count": len(chunks),
                  "chunk_arrival_ms": chunks, "token_arrival_ms": token_times, "error": error,
                  "timing_note": "DSpark may emit multiple tokens per chunk; arrivals measure the API, not hardware token boundaries."}
        if len(token_times) > 1:
            report["mean_api_itl_ms"] = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
            report["api_itl_ms"] = [b - a for a, b in zip(token_times, token_times[1:])]
        (args.output / "timing.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if args.profile:
            session.post(args.url + "/stop_profile", timeout=(10, 1800)).raise_for_status()
        session.close()
        print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
