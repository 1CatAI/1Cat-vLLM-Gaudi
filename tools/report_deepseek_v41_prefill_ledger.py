# SPDX-License-Identifier: Apache-2.0
"""Close a four-rank prefill window without adding overlapping kernel/host time."""
import argparse
import gzip
import json
from pathlib import Path

from deepseek_v41_trace_accounting import clipped, duration
from report_deepseek_v41_latency_ledger import sweep

STAGES = ("Attention提交入口内", "MoE提交入口内")
FALLBACK = "其余主机阶段（含mHC、层间回收、首token消费）"


def kernel_group(category):
    if category in ("路由专家", "共享专家", "Router"):
        return "MoE（Router/路由专家/共享专家）"
    if category in ("mHC", "Engram"):
        return "mHC/Engram"
    # Preserve unknown/fused categories; never guess ownership from shapes.
    return category


def prefill_scope(name):
    if name.startswith("v41::prefill::attention::"):
        return STAGES[0]
    if name.startswith("v41::prefill::moe::"):
        return STAGES[1]
    return None


def report(root, output, kernel_link="../kernel-report/index.html", expected_prompt_tokens=None):
    records, proofs, host_reports = [], [], []
    prompt_tokens = None
    for rank in range(4):
        path = root / f"rank{rank}"
        with gzip.open(path / "activity-intervals.json.gz", "rt") as stream:
            record = json.load(stream)
        proof = json.loads((path / "device-windows.json").read_text())
        if record["rank"] != rank or proof["phase"] != "prefill" or proof["unit"] != "request":
            raise ValueError("Four matching prefill request records are required")
        if len(record["windows_us"]) != 1 or record["windows_us"] != proof["windows_us"]:
            raise ValueError("Activity and coverage proof use different windows")
        count = proof["coverage_proof"]["prompt_tokens"]
        if not isinstance(count, int) or count <= 0:
            raise ValueError("A positive complete prompt-token count is required")
        if expected_prompt_tokens is not None and count != expected_prompt_tokens:
            raise ValueError(f"Expected complete {expected_prompt_tokens}-token prefill")
        if prompt_tokens is not None and count != prompt_tokens:
            raise ValueError("Four ranks must cover the same complete prompt")
        prompt_tokens = count
        records.append(record)
        proofs.append(proof)
        host_reports.append(json.loads((path / "host-breakdown.json").read_text()))
    windows = records[0]["windows_us"]
    base = records[0]["base_time_nanoseconds"]
    if base is None:
        raise ValueError("A shared-clock origin is required for four-rank alignment")
    device, host, observed_host = [], [], []
    ends = [end for _, end in windows]
    for rank, record in enumerate(records):
        if record["base_time_nanoseconds"] is None:
            raise ValueError("Missing rank clock origin")
        shift = (record["base_time_nanoseconds"] - base) / 1000
        for group in record["groups"]:
            category = kernel_group(group["category"])
            device.extend((low + shift, high + shift, ("device", rank, group["engine"], category))
                          for low, high in group["intervals_us"])
        compute = [(a, b) for low, high, key in device if key[1] == rank and key[2] in ("TPC", "MME")
                   for a, b in clipped(low, high, windows, ends)]
        compute_time = duration(compute)
        categories = {}
        intervals = host_reports[rank].get("activity_intervals_us")
        if intervals is None:
            raise ValueError("Host interval export is required to align the API table")
        for name, values in intervals.items():
            spans = [(a, b) for low, high in values for a, b in clipped(low + shift, high + shift, windows, ends)]
            active = duration(spans)
            overlap = active + compute_time - duration(spans + compute)
            categories[name] = dict(activity_ms=active / 1000,
                                    overlap_with_compute_ms=overlap / 1000,
                                    outside_compute_ms=(active - overlap) / 1000)
        observed_host.append(categories)
    for start, length, name in records[0]["host_markers"]:
        stage = prefill_scope(name)
        if stage:
            host.append((start, start + length, ("host", stage)))
    for start, end in host_reports[0].get("activity_intervals_us", {}).get("HCCL_host_API", []):
        host.append((start, end, ("host", "HCCL API")))
    ledger = sweep(device, host, windows, stages=STAGES, fallback=FALLBACK)
    period_ms = sum(end - start for start, end in windows) / 1000
    rows = sorted((dict(label=label, ms=value / 1000, percent=value / 1000 / period_ms * 100)
                   for label, value in ledger["totals_us"].items()),
                  key=lambda row: -row["ms"])
    result = dict(phase="prefill",
                  unit="request",
                  prompt_tokens=prompt_tokens,
                  period_ms=period_ms,
                  base_time_nanoseconds=base,
                  windows_us=windows,
                  trace_sha256=[record["trace_sha256"] for record in records],
                  coverage_proof=[proof["coverage_proof"] for proof in proofs],
                  additive_rows=rows,
                  rank0_no_activity_other_rank_active_ms=ledger["rank0_no_device_other_rank_active_us"] / 1000,
                  stage_activity_ms={
                      stage: {
                          name: value / 1000
                          for name, value in values.items()
                      }
                      for stage, values in ledger["stage_activity_us"].items()
                  },
                  observed_host_activity_by_rank=observed_host,
                  limits=[
                      "Trace request timing is independent of unprofiled serving speed; no rescaling",
                      "No visible activity does not prove hardware idle; NIC visibility can be incomplete",
                      "Host scopes identify observed stages, not proven causes of device gaps",
                      "Host API/category durations can nest and overlap; they are not additive costs",
                      "Unknown kernel ownership remains explicit; cross-module overlap counted once"
                  ])
    output.mkdir(parents=True, exist_ok=False)
    (output / "ledger.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# TP4：完整{prompt_tokens} token prefill时间账",
        "",
        f"同一次请求、四卡对齐、rank0首token消费边界：**{period_ms:.6f} ms/request**。"  # noqa: E501
        "这是trace诊断时间，与关闭采集的正式速度分开报告。",
        "",
        "下表互斥且相加闭合；四卡时间取并集，不相加。kernel交叠单列。",
        "",  # noqa: E501
        "| 设备活动 / 主机阶段位置 | ms/request | 占比 |",
        "|---|---:|---:|"
    ]
    lines.extend(f"| {row['label']} | {row['ms']:.6f} | {row['percent']:.3f}% |" for row in rows)
    lines.extend([
        f"| 合计 | {sum(row['ms'] for row in rows):.6f} | 100% |",
        "",
        f"rank0无可见活动但其他卡仍活动："
        f"{result['rank0_no_activity_other_rank_active_ms']:.6f} ms。",
        "",
        "无可见设备活动按当时rank0所在Attention、MoE、其余主机阶段拆分；"  # noqa: E501
        "这说明位置，尚不能证明等待原因。NIC未完整暴露时，也不能等同于四卡空闲。",
        "",
        "## 主机API观测（可能相互重叠，不可与上表相加）",
        "",  # noqa: E501
        "| Rank | 类别 | API活动ms | 其中与计算交叠ms | 计算之外ms |",
        "|---:|---|---:|---:|---:|"
    ])
    for rank, record in enumerate(observed_host):
        for name, values in record.items():
            lines.append(f"| {rank} | {name} | {values['activity_ms']:.6f} | "
                         f"{values['overlap_with_compute_ms']:.6f} | {values['outside_compute_ms']:.6f} |")
    lines.extend([
        "", f"[完整kernel明细、shape/dtype、物理调用次数及未知项]({kernel_link})。", "",
        "主机compile/submit/wait区间可以嵌套，表内数值保留原始观测，不当作独占延迟归因。"
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(period_ms=period_ms, rows=rows), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kernel-report-link", default="../kernel-report/index.html")
    parser.add_argument("--expected-prompt-tokens", type=int)
    args = parser.parse_args()
    report(args.root, args.output, args.kernel_report_link, args.expected_prompt_tokens)
