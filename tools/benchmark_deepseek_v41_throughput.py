# SPDX-License-Identifier: Apache-2.0
"""Frozen-corpus HTTP throughput and common-residency measurements.

Input JSON requires source, revision, tokenizer_sha256, same_paper_corpus,
and prefill/decode lists of {id, token_ids}. It never silently truncates a
prompt or substitutes ignore_eos. Results do not establish hardware parity.
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

TARGETS = {
    1: (4676.39, 87.60, 86.39),
    2: (4989.25, 109.94, 102.41),
    4: (5265.43, 188.71, 148.50),
    8: (5099.23, 334.18, 254.45),
    16: (5171.36, 608.23, 422.51),
    32: (5254.98, 1044.32, 680.61),
    64: (5250.05, 1334.24, 848.14),
}
DECODE_SCALING_TARGETS = {1: 80., 2: 160., 4: 250., 8: 450., 16: 800., 32: 1300.}


def validate_corpus(corpus, protocol="paper"):
    for name in ("source", "revision", "tokenizer_sha256", "same_paper_corpus"):
        if name not in corpus:
            raise ValueError(f"Missing corpus provenance: {name}")
    kinds = (("decode", (2048, )), ) if protocol == "cached-decode" else (("prefill", (8191, 8192)), ("decode", (511,
                                                                                                                 512)))
    minimum = 32 if protocol == "cached-decode" else 64
    for kind, lengths in kinds:
        prompts = corpus.get(kind, [])
        if len(prompts) < minimum:
            raise ValueError(f"{kind} needs at least {minimum} frozen prompts")
        for row in prompts:
            ids = row["token_ids"]
            if len(ids) not in lengths or not all(isinstance(i, int) and i >= 0 for i in ids):
                raise ValueError(f"Invalid {kind} token IDs for {row['id']}")


def prefix_metrics_snapshot(document, model):
    """Read upstream counters without changing the serving process."""
    from prometheus_client.parser import text_string_to_metric_families
    values, buckets, configs = {}, {}, {}
    engines = set()
    for family in text_string_to_metric_families(document):
        for sample in family.samples:
            if sample.name == "vllm:cache_config_info" and "engine" in sample.labels:
                configs[sample.labels["engine"]] = dict(sample.labels)
                continue
            if sample.labels.get("model_name") != model or "engine" not in sample.labels:
                continue
            engines.add(sample.labels.get("engine"))
            if sample.name == "vllm:request_prefill_kv_computed_tokens_bucket":
                buckets[float(sample.labels["le"])] = sample.value
            elif sample.name == "vllm:request_success_total":
                values["finished_" + sample.labels["finished_reason"]] = sample.value
            elif sample.name in {
                    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                    "vllm:request_prefill_kv_computed_tokens_count", "vllm:request_prefill_kv_computed_tokens_sum",
                    "vllm:num_preemptions_total", "vllm:num_requests_running", "vllm:num_requests_waiting"
            }:
                values[sample.name.removeprefix("vllm:")] = sample.value
    if len(engines) != 1 or not buckets:
        raise ValueError("Cache proof requires one isolated engine with upstream cache metrics")
    config = configs.get(next(iter(engines)))
    if config is None:
        raise ValueError("Cache proof requires the matching engine cache configuration")
    return dict(values=values, buckets=buckets, config=config)


def prefix_metrics_proof(before, after, requests, minimum):
    """Prove every request hit, using the per-request uncached-token histogram.

    A high aggregate hit count alone is insufficient. Equal page-aligned
    prompts plus the histogram bound rule out one miss hidden by another hit.
    Missing usage values stay missing; no per-request cache counts are invented.
    """
    result = dict(valid=False, source="upstream-metrics", request_count=len(requests))
    try:
        a, b = before["values"], after["values"]
        config = before["config"]
        if config != after["config"] or config["enable_prefix_caching"] != "True":
            raise ValueError("Cache configuration changed or caching is disabled")
        block = int(config["block_size"])
        lengths = {r["input_tokens"] for r in requests}
        if len(lengths) != 1 or block <= 0:
            raise ValueError("Metrics proof requires equal page-aligned prompt lengths")
        length = lengths.pop()
        if length % block or not 0 < minimum <= length:
            raise ValueError("Prompt length or cache threshold is not supported")
        delta = {key: b[key] - value for key, value in a.items() if not key.startswith("num_requests_")}
        if any(not math.isfinite(v) or v < 0 for v in delta.values()):
            raise ValueError("Metric reset or non-finite counter")
        n = len(requests)
        if any(a[key] != 0 or b[key] != 0 for key in ("num_requests_running", "num_requests_waiting")):
            raise ValueError("Wave boundaries are not drained")
        if delta["prefix_cache_queries_total"] != n * length or delta["request_prefill_kv_computed_tokens_count"] != n:
            raise ValueError("Metrics include missing or unrelated requests")
        finished = sum(value for key, value in delta.items() if key.startswith("finished_"))
        if finished != n or any(delta.get("finished_" + reason, 0) for reason in ("abort", "error", "repetition")):
            raise ValueError("Wave did not finish exactly the expected successful requests")
        if delta["num_preemptions_total"]:
            raise ValueError("Preempted wave needs request-level cache accounting")
        if (delta["prefix_cache_hits_total"] + delta["request_prefill_kv_computed_tokens_sum"] != n * length
                or delta["prefix_cache_hits_total"] < n * minimum):
            raise ValueError("Cached and computed prompt work does not reconcile")
        # Cached prefixes consist of complete blocks. Therefore uncached
        # work is a block multiple for these exact page-aligned inputs.
        next_invalid = ((length - minimum) // block + 1) * block
        limits = [
            limit for limit in before["buckets"]
            if limit in after["buckets"] and math.isfinite(limit) and limit < next_invalid
        ]
        if not limits:
            raise ValueError("Histogram is too coarse to prove the required cache bound")
        limit = max(limits)
        if after["buckets"][limit] - before["buckets"][limit] != n:
            raise ValueError("At least one request recomputed too much of its prompt")
        result.update(valid=True,
                      delta=delta,
                      histogram_upper_bound=limit,
                      cached_tokens_lower_bound_per_request=length - int(limit // block) * block)
    except (KeyError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    return result


async def read_prefix_metrics(client, args, path):
    response = await client.get(args.url.rstrip("/") + "/metrics")
    response.raise_for_status()
    path.write_text(response.text)
    return prefix_metrics_snapshot(response.text, args.model)


def score_wave(requests, min_cached_tokens=None, cache_proof=None):
    """All timestamps share the client's monotonic clock, in ns.

    Residency excludes the token at the opening boundary: throughput counts
    completed intervals (left-open, right-closed), not an extra first token.
    Coalesced events retain counts for HTTP throughput but invalidate ITL and
    the common-window qualification, instead of inventing zero intervals.
    """
    complete = all(not r.get("error") and r["done"] and r["events"] for r in requests)
    started = min(r["started_ns"] for r in requests)
    finished = max(r["finished_ns"] for r in requests)
    seconds = (finished - started) / 1e9
    tokens = sum(len(e["token_ids"]) for r in requests for e in r["events"])
    coalesced = sum(len(e["token_ids"]) > 1 for r in requests for e in r["events"])
    accounted = all(
        r.get("usage") and r["usage"]["completion_tokens"] == sum(len(e["token_ids"]) for e in r["events"])
        for r in requests)
    cached = [((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") for r in requests]
    cache_valid = (min_cached_tokens is None or all(
        type(count) is int and min_cached_tokens <= count <= r["input_tokens"] for r, count in zip(requests, cached)))
    if min_cached_tokens is not None and cache_proof is not None:
        cache_valid = bool(cache_proof["valid"])
        # Explicit contradictory response data cannot be overridden by totals.
        cache_valid &= all(count is None or (type(count) is int and min_cached_tokens <= count <= r["input_tokens"])
                           for r, count in zip(requests, cached))
    # Prefill ends at the last first token, not the final HTTP/DONE event.
    # Retain both boundaries so network drain is never called prefill work.
    prefill_seconds = ((max(r["events"][0]["received_ns"] for r in requests) - started) / 1e9 if complete else None)
    result = dict(complete=complete,
                  token_accounting_valid=accounted,
                  coalesced_events=coalesced,
                  cached_tokens=cached,
                  cache_proof=cache_proof,
                  cache_hit_valid=cache_valid,
                  http_seconds=seconds,
                  output_tokens=tokens,
                  prefill_seconds=prefill_seconds,
                  http_output_tokens_s=tokens / seconds if seconds > 0 else None,
                  input_tokens=sum(r["input_tokens"] for r in requests),
                  common_seconds=0.,
                  common_tokens=0,
                  resident_valid=False,
                  ttft_ms=[],
                  request_ms=[],
                  itl_ms=[])
    for r in requests:
        events = r["events"]
        result["request_ms"].append((r["finished_ns"] - r["started_ns"]) / 1e6)
        if events:
            result["ttft_ms"].append((events[0]["received_ns"] - r["started_ns"]) / 1e6)
        if all(len(e["token_ids"]) == 1 for e in events):
            result["itl_ms"].extend((b["received_ns"] - a["received_ns"]) / 1e6 for a, b in zip(events, events[1:]))
    if not complete or coalesced or not accounted or not cache_valid:
        return result
    left = max(r["events"][0]["received_ns"] for r in requests)
    right = min(r["events"][-1]["received_ns"] for r in requests)
    if right > left:
        result.update(common_seconds=(right - left) / 1e9,
                      common_tokens=sum(left < e["received_ns"] <= right for r in requests for e in r["events"]),
                      resident_valid=True)
    return result


def score_round(waves, concurrency, kind, repeat, common_required):
    valid = bool(waves) and all(w["complete"] and w["token_accounting_valid"] for w in waves)
    http_seconds = sum(w["http_seconds"] for w in waves)
    common_seconds = sum(w["common_seconds"] for w in waves if w["resident_valid"])
    prefill_seconds = sum(w["prefill_seconds"] or 0 for w in waves)
    if kind == "prefill":
        valid &= prefill_seconds > 0 and all(w["output_tokens"] == concurrency for w in waves)
    else:
        valid &= common_seconds >= common_required and all(w["resident_valid"] for w in waves)
    return dict(concurrency=concurrency,
                kind=kind,
                round=repeat,
                valid=valid,
                http_seconds=http_seconds,
                prefill_seconds=prefill_seconds,
                output_tokens=sum(w["output_tokens"] for w in waves),
                common_seconds=common_seconds,
                input_tokens_s=(sum(w["input_tokens"]
                                    for w in waves) / prefill_seconds if kind == "prefill" and valid else None),
                http_output_tokens_s=(sum(w["output_tokens"]
                                          for w in waves) / http_seconds if http_seconds > 0 else None),
                resident_output_tokens_s=(sum(w["common_tokens"]
                                              for w in waves) / common_seconds if common_seconds else None))


def compare_rounds(results, concurrency, protocol="paper"):
    row = dict(concurrency=concurrency)
    if protocol == "cached-decode":
        rounds = [r for r in results if r["concurrency"] == concurrency and r["kind"] == "decode"]
        values = [r["resident_output_tokens_s"] for r in rounds if r["resident_output_tokens_s"] is not None]
        qualified = len(rounds) == len(values) == 3 and all(r["valid"] for r in rounds)
        median = statistics.median(values) if values else None
        target = DECODE_SCALING_TARGETS[concurrency]
        row["resident_output_tokens_s"] = dict(
            median=median,
            qualified=qualified,
            target=target,
            exceeded=bool(qualified and median >= target),
            scaling_efficiency_vs_80=(median / (80 * concurrency) if median is not None else None))
        return row
    for kind, metric, target in (("prefill", "input_tokens_s", 0), ("decode", "resident_output_tokens_s", 1),
                                 ("decode", "http_output_tokens_s", 2)):
        rounds = [r for r in results if r["concurrency"] == concurrency and r["kind"] == kind]
        qualified = len(rounds) == 3 and all(r["valid"] for r in rounds)
        values = [r[metric] for r in rounds if r[metric] is not None]
        qualified &= len(values) == 3
        median = statistics.median(values) if values else None
        result = dict(median=median,
                      qualified=qualified,
                      target=TARGETS[concurrency][target],
                      exceeded=bool(qualified and median > TARGETS[concurrency][target]))
        if metric == "http_output_tokens_s":
            seconds = sum(r["http_seconds"] for r in rounds)
            pooled = sum(r["output_tokens"] for r in rounds) / seconds if seconds > 0 else None
            result.update(pooled=pooled, pooled_exceeded=bool(qualified and pooled > TARGETS[concurrency][target]))
            result["exceeded"] &= result["pooled_exceeded"]
        row[metric] = result
    return row


async def request(client, args, prompt, kind, directory):
    directory.mkdir()
    payload = dict(model=args.model,
                   prompt=prompt["token_ids"],
                   temperature=0,
                   max_tokens=1 if kind == "prefill" else 2048,
                   ignore_eos=False,
                   stream=True,
                   return_token_ids=True,
                   stream_options={"include_usage": True})
    (directory / "request.json").write_text(json.dumps(payload))
    record = dict(prompt_id=prompt["id"],
                  input_tokens=len(prompt["token_ids"]),
                  started_ns=time.perf_counter_ns(),
                  events=[],
                  usage=None,
                  error=None,
                  done=False)
    with (directory / "stream.jsonl").open("w") as raw:
        try:
            async with client.stream("POST", args.url.rstrip("/") + "/v1/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    received = time.perf_counter_ns()
                    if not line.startswith("data: "):
                        continue
                    body = line[6:]
                    raw.write(json.dumps(dict(received_ns=received, data=body), ensure_ascii=False) + "\n")
                    if body == "[DONE]":
                        record["done"] = True
                        break
                    item = json.loads(body)
                    if item.get("error"):
                        raise RuntimeError(item["error"])
                    if item.get("usage"):
                        record["usage"] = item["usage"]
                    for choice in item.get("choices", []):
                        if choice.get("token_ids"):
                            record["events"].append(dict(received_ns=received, **choice))
        except Exception as exc:
            record["error"] = repr(exc)
    record["finished_ns"] = time.perf_counter_ns()
    (directory / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


async def measure(args, corpus):
    import httpx
    results = []
    async with httpx.AsyncClient(timeout=args.timeout,
                                 trust_env=False,
                                 limits=httpx.Limits(max_connections=64, max_keepalive_connections=64)) as client:
        if args.protocol == "cached-decode" and not args.reuse_warm_prefixes:
            # Warm exact frozen prompts via normal requests. Do not equate
            # warming with a cache hit: every measured request must report
            # its own cached-token count in usage, or qualification fails.
            warm = args.output / "prefix-warmup"
            warm.mkdir()
            for index, prompt in enumerate(corpus["decode"]):
                record = await request(client, args, prompt, "prefill", warm / f"request{index}")
                if not record["done"] or record["error"]:
                    raise RuntimeError("Prefix warmup failed; original response retained")
        for concurrency in args.concurrency:
            for kind in args.kind:
                for repeat in range(args.rounds):
                    parent = args.output / f"c{concurrency}-{kind}-round{repeat}"
                    parent.mkdir()
                    waves, common_seconds = [], 0.
                    for wave in range(args.max_waves):
                        rows = corpus[kind]
                        before = (await read_prefix_metrics(client, args, parent / f"wave{wave}-metrics-before.txt")
                                  if args.cache_proof == "metrics" else None)
                        records = await asyncio.gather(*(request(client, args, rows[(wave * concurrency + slot) %
                                                                                    len(rows)], kind, parent /
                                                                 f"wave{wave}-request{slot}")
                                                         for slot in range(concurrency)))
                        proof = None
                        if before is not None:
                            after = await read_prefix_metrics(client, args, parent / f"wave{wave}-metrics-after.txt")
                            proof = prefix_metrics_proof(before, after, records, args.min_cached_tokens)
                        score = score_wave(records,
                                           args.min_cached_tokens if args.protocol == "cached-decode" else None, proof)
                        waves.append(score)
                        common_seconds += score["common_seconds"] if score["resident_valid"] else 0
                        (parent / "waves.json").write_text(json.dumps(waves, indent=2))
                        if not score["complete"] or not score["token_accounting_valid"] or not score["cache_hit_valid"]:
                            break
                        if kind == "prefill" or common_seconds >= args.common_seconds:
                            break
                    result = score_round(waves, concurrency, kind, repeat, args.common_seconds)
                    results.append(result)
                    (args.output / "rounds.json").write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)
                    if not result["valid"]:
                        raise RuntimeError("Request/accounting failure retained; stopping invalid campaign")
    table = [compare_rounds(results, concurrency, args.protocol) for concurrency in args.concurrency]
    (args.output / "comparison.json").write_text(json.dumps(table, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-C1")
    parser.add_argument("--protocol", choices=("paper", "cached-decode"), default="paper")
    parser.add_argument("--cache-proof",
                        choices=("usage", "metrics"),
                        default="usage",
                        help="Metrics require an isolated engine and reconcile every wave's actual prefill work")
    parser.add_argument("--reuse-warm-prefixes",
                        action="store_true",
                        help="Resume on the same warm service; every measured request must still prove a cache hit")
    parser.add_argument("--min-cached-tokens",
                        type=int,
                        default=1920,
                        help="For 2K input, allow the final 128-token page to be recomputed")
    parser.add_argument("--concurrency", type=int, nargs="+", choices=TARGETS)
    parser.add_argument("--kind", nargs="+", choices=("prefill", "decode"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--common-seconds", type=float, default=30.)
    parser.add_argument("--max-waves", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=1800.)
    args = parser.parse_args()
    if args.protocol == "cached-decode":
        if args.kind not in (None, ["decode"]):
            parser.error("cached-decode measures decode only")
        if not 1 <= args.min_cached_tokens <= 2048:
            parser.error("cached-decode requires positive cached tokens within the 2K prompt")
        args.kind = ["decode"]
        args.concurrency = args.concurrency or list(DECODE_SCALING_TARGETS)
        if any(c not in DECODE_SCALING_TARGETS for c in args.concurrency):
            parser.error("cached-decode targets concurrency 1/2/4/8/16/32")
    else:
        if args.reuse_warm_prefixes:
            parser.error("Warm-prefix reuse applies only to cached-decode")
        if args.cache_proof != "usage":
            parser.error("Metrics cache proof applies only to cached-decode")
        args.kind = args.kind or ["prefill", "decode"]
        args.concurrency = args.concurrency or list(TARGETS)
    corpus = json.loads(args.corpus.read_text())
    validate_corpus(corpus, args.protocol)
    if args.common_seconds < 30 or args.rounds < 1 or args.max_waves < 1:
        parser.error("Require >=30 seconds common residency and positive rounds/waves")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "corpus.json").write_bytes(args.corpus.read_bytes())
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "protocol.json").write_text(
        json.dumps(dict(
            parameters={
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            },
            corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            comparison=("V4.1 Gaudi2 TP2xPP2 cached 2K pure decode scaling; C1 anchor 80 tokens/s"
                        if args.protocol == "cached-decode" else
                        "V4.1 Gaudi2 TP2xPP2 versus V4 MI250 TP8; deployment comparison, not hardware parity"),
            timer=("monotonic client SSE; prefill=start..max(first); residency=max(first)..min(last), (left,right]; "
                   "HTTP includes admission/drain; report both round median and pooled complete waves"),
            protocol_version=3,
            phase="explicit candidate run; first-shape/cold runs must be archived separately",
        ),
                   indent=2))
    asyncio.run(measure(args, corpus))


if __name__ == "__main__":
    main()
