# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Join four worker monotonic clocks with final scheduler consumption."""

import argparse
import json
from pathlib import Path
from statistics import mean, median


def stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    index = (len(ordered) - 1) * .95
    low = int(index)
    p95 = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return dict(count=len(values), mean_ms=mean(values), median_ms=median(values), p95_ms=p95, max_ms=max(values))


def indexed(document):
    if document["clock"] != "perf_counter_ns":
        raise ValueError("Round clocks must use the same host monotonic timebase")
    result = {}
    for row in document["records"]:
        key = row["request_id"], row["generation"]
        if key in result:
            raise ValueError(f"Duplicate round {key}")
        result[key] = row
    return result


def reconcile(workers, engine):
    if set(workers) != set(range(4)):
        raise ValueError("Full-round qualification requires all four worker ledgers")
    ranks = {rank: indexed(doc) for rank, doc in workers.items()}
    consumed = indexed(engine)
    keys = set(consumed)
    if any(set(rows) != keys for rows in ranks.values()):
        raise ValueError("Worker and scheduler generations are incomplete or different")
    result = []
    for key in sorted(keys, key=lambda key: min(ranks[r][key]["start_ns"] for r in ranks)):
        rows = [ranks[r][key] for r in range(4)]
        record = consumed[key]
        for field in ("target_count", "proposed_count", "committed", "output_count", "output"):
            if any(row[field] != rows[0][field] for row in rows[1:]):
                raise ValueError(f"Rank disagreement for {key}: {field}")
        if record["target_count"] != rows[0]["target_count"]:
            raise ValueError(f"Scheduler C-shape disagreement for {key}")
        if any(row["end_ns"] < row["start_ns"] for row in rows):
            raise ValueError(f"Invalid worker interval for {key}")
        if any(not row.get("ring_released") for row in rows[2:]):
            raise ValueError(f"Missing PP1 post-release completion for {key}")
        start = min(row["start_ns"] for row in rows)
        end = max(record["scheduler_consumed_ns"], *(row["end_ns"] for row in rows))
        if record["scheduler_consumed_ns"] < start:
            raise ValueError(f"Scheduler completion precedes input preparation for {key}")
        result.append(
            dict(request_id=key[0],
                 generation=key[1],
                 target_count=rows[0]["target_count"],
                 proposed_count=rows[0]["proposed_count"],
                 committed=rows[0]["committed"],
                 output_count=rows[0]["output_count"],
                 start_ns=start,
                 end_ns=end,
                 full_round_ms=(end - start) / 1e6,
                 scheduler_consumed_ns=record["scheduler_consumed_ns"],
                 workers={
                     str(rank): row
                     for rank, row in enumerate(rows)
                 }))
    return result


def summarize(rows, discard):
    summaries = {}
    for request_id in dict.fromkeys(row["request_id"] for row in rows):
        request = [row for row in rows if row["request_id"] == request_id]
        complete = [row for row in request if row["target_count"] == 6 and row["proposed_count"] == 5]
        steady = complete[discard:]
        c1 = [row for row in request if row["target_count"] == 1]
        tail = [row for row in request if 1 < row["target_count"] <= 6 and row not in complete]
        prefill = [row for row in request if row["target_count"] > 6]
        proposed = sum(row["proposed_count"] for row in steady)
        accepted = sum(row["committed"] - 1 for row in steady)
        summaries[request_id] = dict(steady_c6=stats([row["full_round_ms"] for row in steady]),
                                     discarded_c6=min(discard, len(complete)),
                                     total_c6=len(complete),
                                     c1=stats([row["full_round_ms"] for row in c1]),
                                     partial_verify=stats([row["full_round_ms"] for row in tail]),
                                     prefill=stats([row["full_round_ms"] for row in prefill]),
                                     accepted_drafts=accepted,
                                     proposed_drafts=proposed,
                                     acceptance_rate=accepted / proposed if proposed else None,
                                     full_round_speed_pass=bool(steady)
                                     and mean(row["full_round_ms"] for row in steady) < 50)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--discard", type=int, default=10)
    args = parser.parse_args()
    if args.discard < 0:
        raise ValueError("Discard count cannot be negative")
    directory = args.run / "round-timing"
    workers = {rank: json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(4)}
    engines = list(directory.glob("engine-*.json"))
    if len(engines) != 1:
        raise ValueError("Expected one EngineCore timing ledger")
    rows = reconcile(workers, json.loads(engines[0].read_text()))
    summaries = summarize(rows, args.discard)
    result = dict(units="ms",
                  boundary="earliest worker input preparation through latest worker commit or scheduler consume",
                  discard=args.discard,
                  requests=summaries,
                  rounds=rows)
    (args.run / "full-rounds.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# DSpark 完整轮次", "", "四个 rank 输入准备最早时刻至所有 worker 提交及 scheduler 消费的最晚时刻。",
        f"同主机 perf_counter_ns；无新增设备同步。每请求丢弃前 {args.discard} 个完整 C6，C1/尾部/prefill 单列。", "",
        "| 请求 | 稳态 C6 数 | 平均 ms | P95 ms | 最大 ms | Draft 接受率 | <50 ms |", "|---|---:|---:|---:|---:|---:|---|"
    ]
    for request_id, summary in summaries.items():
        timing = summary["steady_c6"]
        if not timing["count"]:
            lines.append(f"| {request_id} | 0 | — | — | — | — | 未验收 |")
            continue
        lines.append(f"| {request_id} | {timing['count']} | {timing['mean_ms']:.6f} | {timing['p95_ms']:.6f} | "
                     f"{timing['max_ms']:.6f} | {summary['acceptance_rate']:.2%} | "
                     f"{'通过速度门' if summary['full_round_speed_pass'] else '未达标'} |")
    (args.run / "FULL_ROUND_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
