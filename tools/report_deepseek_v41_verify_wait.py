# SPDX-License-Identifier: Apache-2.0
"""Reconcile PP0 verify consumption using same-transaction host/device markers."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


HOST_PHASES = (
    ("前置处理", "consume_start", "validator_submit_start"),
    ("校验图及固定缓冲提交", "validator_submit_start", "validator_submit_done"),
    ("D2H copy 调用及提交", "validator_submit_done", "d2h_submit_done"),
    ("完成事件记录／pipeline 提交等待", "d2h_submit_done", "event_wait_start_ns"),
    ("设备完成事件等待", "event_wait_start_ns", "event_wait_done_ns"),
    ("host callback 排空", "event_wait_done_ns", "host_callbacks_done_ns"),
    ("CPU 列表读取及校验", "host_callbacks_done_ns", "consume_done"),
)
CRITICAL_PHASES = (
    "PP0 target 尚未完成：剩余执行及内部依赖",
    "PP0 target 完成至 PP1 target 开始（交接及输入依赖）",
    "PP1 target 执行区间（含捕获／提交间隙）",
    "PP1 target 完成至验证前缀／commit 源就绪",
    "源就绪后的 PP 交付（含提交和调度）",
    "PP0 收到结果至校验开始",
    "PP0 校验区间（含提交间隙）",
    "校验完成至 D2H 完成（含提交间隙）",
    "D2H 完成至 CPU 消费结束",
)


def stats(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * .95
    low = int(index)
    p95 = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return {"count": len(values), "mean_ms": mean(values), "p95_ms": p95, "max_ms": max(values)}


def marker_host(record, document, name):
    stamp = record["device"][name]
    if stamp["status"] != "complete":
        raise ValueError(f"Incomplete marker {name}: {stamp}")
    calibration = document["calibration"]
    offset = (calibration["offset_low_ns"] + calibration["offset_high_ns"]) / 2
    return stamp["device_timestamp_ns"] + offset


def split_record(left, right, left_doc, right_doc):
    host = left["host"]
    start, end = host["consume_start"], host["consume_done"]
    host_ms = {}
    for label, begin, finish in HOST_PHASES:
        if host[finish] < host[begin]:
            raise ValueError(f"Host ordering error: {label}")
        host_ms[label] = (host[finish] - host[begin]) / 1e6
    total = (end - start) / 1e6
    assert abs(sum(host_ms.values()) - total) < 1e-8
    points = [marker_host(left, left_doc, "stage_target_done"),
              marker_host(right, right_doc, "stage_model_start"),
              marker_host(right, right_doc, "stage_target_done"),
              marker_host(right, right_doc, "commit_source_ready"),
              marker_host(left, left_doc, "commit_transfer_done"),
              marker_host(left, left_doc, "validator_start"),
              marker_host(left, left_doc, "validator_done"),
              marker_host(left, left_doc, "d2h_done")]
    # Clock alignment only yields a bounded cross-device completion time.
    uncertainty = sum((doc["calibration"]["offset_high_ns"] -
                       doc["calibration"]["offset_low_ns"]) / 2 for doc in (left_doc, right_doc))
    # Clipping intersects each milestone interval with the observed PP0 wait;
    # never count complete PP1 compute again as an extra additive latency.
    previous = start
    spans, violations = [], []
    for point in points + [end]:
        clipped = min(end, max(start, point))
        if clipped < previous - uncertainty:
            violations.append({"previous_ns": previous, "point_ns": point})
        clipped = max(previous, clipped)
        spans.append((clipped - previous) / 1e6)
        previous = clipped
    critical = dict(zip(CRITICAL_PHASES, spans, strict=True))
    assert abs(sum(critical.values()) - total) < 1e-8
    return {"request_id": left["request_id"], "generation": left["generation"],
            "committed": left["committed"], "consume_ms": total,
            "verify_transaction_ms": (host["transaction_consumed"] - host["verify_start"]) / 1e6,
            "engram_host_ms": (host["engram_complete_done"] - host["engram_complete_start"]) / 1e6,
            "host_ms": host_ms, "completion_path_ms": critical,
            "clock_pair_uncertainty_ms": uncertainty / 1e6,
            "ordering_violations": violations,
            "pp1_target_device_interval_ms": (marker_host(right, right_doc, "stage_target_done") -
                                               marker_host(right, right_doc, "stage_model_start")) / 1e6,
            "pp1_prefix_device_interval_ms": (marker_host(right, right_doc, "prefix_done") -
                                               marker_host(right, right_doc, "prefix_start")) / 1e6}


def table(lines, summary, key, total):
    lines.extend(["", "| 拆分项 | 平均 ms/次 | 占比 |", "|---|---:|---:|"])
    for label, values in summary[key].items():
        value = values["mean_ms"]
        lines.append(f"| {label} | {value:.6f} | {value / total * 100:.3f}% |")
    lines.append(f"| 合计 | **{total:.6f}** | **100%** |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--discard", type=int, default=10,
                        help="Drop the first N paired C6 transactions of this diagnostic request")
    args = parser.parse_args()
    docs = {rank: json.loads((args.run / "verify-phases" / f"rank{rank}-verify-phases.json").read_text())
            for rank in range(4)}
    all_rows, summaries = [], {}
    lines = ["# C6 PP0 verify 等待归因", "",
             "本次为带原生时间点的诊断采集，不是无插桩性能验收；不能与历史139.6–140.0 ms直接相减。",
             "host 表与设备完成链表是同一窗口的两种视角，不能相加。完成链是里程碑间隔，含排队/提交；不是纯 kernel 活动。",
             "PP 源就绪后的交付仍含提交、调度和通信，不能当作纯链路传输。"]
    for left_rank, right_rank in ((0, 2), (1, 3)):
        right_by_key = {(row["request_id"], row["generation"]): row for row in docs[right_rank]["records"]}
        paired = []
        for left in docs[left_rank]["records"]:
            right = right_by_key.get((left["request_id"], left["generation"]))
            if right is None:
                raise ValueError("Missing matching PP1 generation")
            row = split_record(left, right, docs[left_rank], docs[right_rank])
            row["rank"] = left_rank
            paired.append(row)
        rows = paired[args.discard:]
        if not rows:
            raise ValueError("No steady C6 transactions after discard")
        summary = {"captured": len(paired), "discarded": args.discard,
                   "consume": stats([row["consume_ms"] for row in rows]),
                   "verify_transaction": stats([row["verify_transaction_ms"] for row in rows]),
                   "engram_host": stats([row["engram_host_ms"] for row in rows]),
                   "clock_pair_uncertainty_ms": max(row["clock_pair_uncertainty_ms"] for row in rows),
                   "ordering_violations": sum(len(row["ordering_violations"]) for row in rows)}
        for key in ("host_ms", "completion_path_ms"):
            summary[key] = {label: stats([row[key][label] for row in rows]) for label in rows[0][key]}
        summaries[left_rank] = summary
        all_rows.extend(paired)
        total = summary["consume"]["mean_ms"]
        lines.extend(["", f"## PP0 TP{left_rank}（配对 PP1 TP{left_rank}）", "",
                      f"采集 {len(paired)} 个 C6，丢弃前 {args.discard} 个，统计 {len(rows)} 个。",
                      f"PP0 consume 平均 {total:.6f} ms，P95 {summary['consume']['p95_ms']:.6f} ms，"
                      f"最大 {summary['consume']['max_ms']:.6f} ms。",
                      f"跨卡完成时间对齐误差界 ±{summary['clock_pair_uncertainty_ms']:.6f} ms；"
                      f"里程碑顺序异常 {summary['ordering_violations']}。", "", "CPU 调用拆分："])
        table(lines, summary, "host_ms", total)
        if summary["ordering_violations"]:
            lines.extend(["", "完成链存在超出时钟误差的顺序异常，不可据此归因；原始数据保留。"])
        else:
            lines.extend(["", "同一 consume 窗口的设备完成链："])
            table(lines, summary, "completion_path_ms", total)
        lines.extend(["", f"窗口外的 Engram host completion 平均 {summary['engram_host']['mean_ms']:.6f} ms。",
                      f"包含 Engram 的完整 verify 事务平均 {summary['verify_transaction']['mean_ms']:.6f} ms。"])
    output = args.run / "wait-breakdown"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    (output / "transactions.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    with (output / "transactions.csv").open("w") as stream:
        writer = csv.writer(stream)
        labels = [entry[0] for entry in HOST_PHASES]
        writer.writerow(["rank", "request_id", "generation", "consume_ms", *labels])
        for row in all_rows:
            writer.writerow([row["rank"], row["request_id"], row["generation"], row["consume_ms"],
                             *(row["host_ms"][label] for label in labels)])
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
