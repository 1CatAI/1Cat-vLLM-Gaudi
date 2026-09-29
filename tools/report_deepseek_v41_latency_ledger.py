# SPDX-License-Identifier: Apache-2.0
# Keep multilingual Markdown/HTML table templates readable.
# ruff: noqa: E501
"""Explain an aligned TP4 token period using module activity and observed gaps."""
import argparse
import collections
import gzip
import html
import json
from pathlib import Path
import statistics

from deepseek_v41_trace_accounting import clipped, duration


def scope_kind(name):
    if name.startswith("v41::compiled::"):
        return "compiled分组入口内"
    if name.startswith("v41::target::"):
        return "target输入/分组衔接入口内"
    if name.startswith("v41::verify_and_commit::"):
        return "采样/结果消费与状态提交入口内"
    return None


def sweep(device,
          host,
          windows,
          *,
          stages=("compiled分组入口内", "target输入/分组衔接入口内", "采样/结果消费与状态提交入口内"),
          fallback="上述入口之外的step衔接"):
    """Partition elapsed wall time; labels describe observations, not causes."""
    events = collections.defaultdict(list)
    ends = [end for _, end in windows]
    for start, end, key in device + host:
        for low, high in clipped(start, end, windows, ends):
            events[low].append((key, 1))
            events[high].append((key, -1))
    for low, high in windows:
        events[low].append((("window", ), 1))
        events[high].append((("window", ), -1))
    active = collections.Counter()
    totals, combinations = collections.Counter(), collections.Counter()
    stage_activity = collections.defaultdict(collections.Counter)
    hccl_in_gaps = collections.Counter()
    rank0_no_device_other_rank_active = 0.
    previous = None
    for stamp, changes in sorted(events.items()):
        if previous is not None and active[("window", )]:
            elapsed = stamp - previous
            owner = next((name for name in stages if active[("host", name)]), fallback)
            devices = [key for key, count in active.items() if count and key[0] == "device"]
            compute = {key[3] for key in devices if key[2] in ("TPC", "MME") and key[3] != "设备调度"}
            if compute:
                label = "设备kernel：" + (next(iter(compute)) if len(compute) == 1 else "跨模块重叠")
                combinations[tuple(sorted(compute))] += elapsed
            elif any(key[2] == "TPC" for key in devices):
                label = "仅TPC null/设备调度描述符"
            elif devices:
                kinds = {key[3] for key in devices}
                label = "仅设备搬运：" + (next(iter(kinds)) if len(kinds) == 1 else "多种DMA重叠")
            else:
                label = "四卡无可见设备活动：" + owner
                if active[("host", "HCCL API")]:
                    hccl_in_gaps[label] += elapsed
            totals[label] += elapsed
            stage_activity[owner][label] += elapsed
            if devices and not any(key[1] == 0 for key in devices):
                rank0_no_device_other_rank_active += elapsed
        for key, delta in changes:
            active[key] += delta
            if active[key] < 0:
                raise ValueError("Unbalanced activity interval")
        previous = stamp
    period = sum(end - start for start, end in windows)
    assert abs(sum(totals.values()) - period) < 1e-5
    return dict(totals_us=dict(totals),
                combinations_us=[dict(categories=list(key), us=value) for key, value in combinations.most_common()],
                hccl_overlap_with_gap_us=dict(hccl_in_gaps),
                stage_activity_us={
                    stage: dict(values)
                    for stage, values in stage_activity.items()
                },
                rank0_no_device_other_rank_active_us=rank0_no_device_other_rank_active)


def report(root, output, kernel_report_link="../kernel-report/index.html"):
    output.mkdir(parents=True, exist_ok=False)
    records, kernels = [], []
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    for rank in range(4):
        with gzip.open(root / f"rank{rank}/activity-intervals.json.gz", "rt") as stream:
            records.append(json.load(stream))
        kernels.append(json.loads((root / f"rank{rank}/kernel-breakdown.json").read_text()))
    windows = records[0]["windows_us"]
    tokens = records[0]["tokens"]
    count = len(tokens)
    base = records[0]["base_time_nanoseconds"]
    periods = [(end - start) / 1000 for start, end in windows]
    period_ms = statistics.mean(periods)
    device, host = [], []
    by_category = collections.defaultdict(list)
    per_rank = collections.defaultdict(lambda: collections.defaultdict(list))
    for rank, record in enumerate(records):
        assert record["tokens"] == tokens
        shift = (record["base_time_nanoseconds"] - base) / 1000
        for group in record["groups"]:
            spans = [(low + shift, high + shift) for low, high in group["intervals_us"]]
            category = group["category"]
            device.extend((low, high, ("device", rank, group["engine"], category)) for low, high in spans)
            by_category[category].extend(spans)
            per_rank[rank][category].extend(spans)
    ends = [end for _, end in windows]
    for start, length, name in records[0]["host_markers"]:
        kind = scope_kind(name)
        if kind:
            host.extend((a, b, ("host", kind)) for a, b in clipped(start, start + length, windows, ends))
    host_record = json.loads((root / "rank0/host-breakdown.json").read_text())
    for start, end in host_record.get("activity_intervals_us", {}).get("HCCL_host_API", []):
        host.append((start, end, ("host", "HCCL API")))
    ledger = sweep(device, host, windows)
    scale = count * 1000
    host_stages = {
        name.removeprefix("四卡无可见设备活动："): value / scale
        for name, value in sweep([], host, windows)["totals_us"].items()
    }
    rows = [
        dict(label=name, ms_per_token=value / scale, period_pct=value / scale / period_ms * 100)
        for name, value in ledger["totals_us"].items()
    ]
    rows.sort(key=lambda row: -row["ms_per_token"])
    modules = [
        dict(category=category,
             global_union_ms=duration(spans) / scale,
             per_rank_ms=[duration(per_rank[rank][category]) / scale for rank in range(4)])
        for category, spans in by_category.items()
    ]
    modules.sort(key=lambda row: -row["global_union_ms"])
    median_index = sorted(range(count), key=lambda i: periods[i])[count // 2]
    details = dict(period_mean_ms=period_ms,
                   period_median_ms=statistics.median(periods),
                   period_min_ms=min(periods),
                   period_max_ms=max(periods),
                   tokens=tokens,
                   selected_median_token=tokens[median_index],
                   periods_ms=periods,
                   selection=selection,
                   rank0_host_stage_ms=host_stages,
                   activity_within_host_stage_ms={
                       stage: {
                           name: value / scale
                           for name, value in values.items()
                       }
                       for stage, values in ledger["stage_activity_us"].items()
                   },
                   partition=rows,
                   module_activity=modules,
                   rank0_no_device_other_rank_active_ms=ledger["rank0_no_device_other_rank_active_us"] / scale,
                   hccl_overlap_with_gap_ms={
                       k: v / scale
                       for k, v in ledger["hccl_overlap_with_gap_us"].items()
                   },
                   overlap_combinations=[
                       dict(categories=row["categories"], ms_per_token=row["us"] / scale)
                       for row in ledger["combinations_us"]
                   ],
                   interpretation="Device activity is the union across four TP ranks. Host scope and HCCL "
                   "coincidence identify where a gap occurs, not its causal dependency or network latency. "
                   "No time is scaled to the historical unprofiled result.")
    (output / "latency-ledger.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# TP4 decode：实际消费周期的功能与空档拆解", "", f"同一采集的{count}个完整40层周期，P{tokens[0]}–P{tokens[-1]}："
        f"均值 **{period_ms:.6f} ms/token**，中位{statistics.median(periods):.6f} ms。",
        "原无profiler三轮中位数是22.782107 ms/token。本表使用本次真实周期，不缩放到原测速值。", "",
        ("**单周期案例：**选择采集中最接近原测速周期的真实完整token，仅用于说明请求如何经过各功能。"
         "不是无profiler三轮平均值的事后还原，也不代表整个采集分布。"
         "[全部31周期及分布](../latency-report/index.html)。" if selection else "保留本次采集的全部完整周期；不按延迟筛选样本。"), "",
        "## 完整token的host入口阶段", "", "这是rank0控制线程经过各入口的墙钟包络，包含异步提交及等待，不是CPU计算量。"
        "以下四项互斥并覆盖完整周期；硬件kernel发生在这些包络内，不能再次相加。", "", "| host所处阶段 | ms/token | 周期占比 |", "|---|---:|---:|"
    ]
    md.extend(f"| {name} | {value:.6f} | {value / period_ms * 100:.3f}% |" for name, value in host_stages.items())
    md.extend([
        "", "## 四卡对齐后的互斥时间账", "", "每段时间优先记录当前可见TPC/MME kernel；没有计算时再列调度/DMA，四卡都没有设备区间时按rank0所处host入口标明位置。"
        "host入口可能包含等待，名称不能证明CPU在做计算；NIC不可见，空档不能直接算成通信。", "", "| 当时可见的活动或空档位置 | ms/token | 周期占比 |", "|---|---:|---:|"
    ])
    md.extend(f"| {row['label']} | {row['ms_per_token']:.6f} | {row['period_pct']:.3f}% |" for row in rows)
    md.extend([
        f"| 合计 | {sum(row['ms_per_token'] for row in rows):.6f} | 100% |", "", f"rank0没有可见设备区间、但其他TP卡仍有设备活动的时间："
        f"**{details['rank0_no_device_other_rank_active_ms']:.6f} ms/token**。因此不能把单卡空档当作整机空闲。", "", "## 各模块硬件活动并集", "",
        "以下各行可能重叠；四卡全局并集不是四卡时间相加，也不是独占关键路径。", "", "| 功能 | Rank0 ms | Rank1 ms | Rank2 ms | Rank3 ms | 四卡并集 ms |",
        "|---|---:|---:|---:|---:|---:|"
    ])
    for row in modules:
        md.append("| " + row["category"] + " | " +
                  " | ".join(f"{value:.6f}" for value in [*row["per_rank_ms"], row["global_union_ms"]]) + " |")
    md.extend([
        "", "## 逐kernel明细", "", f"[四卡全部kernel、物理调用次数、实际A/B/输出shape与dtype]({kernel_report_link})。"
        "数值是同一窗口中的硬件活动；kernel间重叠已在上面的互斥账中单列。", "", "`通信相关DMA`仅依据SDK的HCL dedicated lane标签，不代表完整NIC传输延迟；"
        "`命令搬运`依据pdma_tx_commands标签，不推断模型张量字节量。", ""
    ])
    (output / "REPORT.md").write_text("\n".join(md))

    # The full acquisition uses its median cycle. Explicit examples carry a
    # separate selection record and never replace the complete distribution.
    low, high = windows[median_index]
    width, left, row_height = 1250, 175, 28
    categories = sorted(by_category)
    palette = ["#2878b5", "#c44e52", "#55a868", "#8172b2", "#dd8452", "#64b5cd", "#937860", "#8c8c8c"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(categories)}
    lanes = [(rank, engine) for rank in range(4) for engine in ("TPC", "MME", "DMA")]
    height = 65 + row_height * (len(lanes) + 3)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 20}" height="{height}" '
        f'viewBox="0 0 {left + width + 20} {height}">', '<rect width="100%" height="100%" fill="white"/>'
    ]
    for tick in range(0, int((high - low) / 1000) + 1, 2):
        x = left + tick * 1000 / (high - low) * width
        svg.extend([
            f'<line x1="{x}" x2="{x}" y1="25" y2="{height - 15}" stroke="#e5e7eb"/>',
            f'<text x="{x}" y="17" font-size="11">{tick} ms</text>'
        ])
    for i, (rank, engine) in enumerate(lanes):
        y = 30 + i * row_height
        svg.append(f'<text x="5" y="{y + 15}" font-size="12">rank{rank} {engine}</text>')
        for start, end, key in device:
            if key[1:3] != (rank, engine) or end <= low or start >= high:
                continue
            a, b = max(start, low), min(end, high)
            x, w = left + (a - low) / (high - low) * width, (b - a) / (high - low) * width
            title = html.escape(f"{key[3]}: {(a-low)/1000:.6f}–{(b-low)/1000:.6f} ms")
            svg.append(f'<rect x="{x}" y="{y}" width="{max(w, .08)}" height="19" '
                       f'fill="{colors[key[3]]}"><title>{title}</title></rect>')
    for i, kind in enumerate(("compiled分组入口内", "采样/结果消费与状态提交入口内", "HCCL API")):
        y = 30 + (len(lanes) + i) * row_height
        svg.append(f'<text x="5" y="{y + 15}" font-size="10">rank0 {html.escape(kind)}</text>')
        for start, end, key in host:
            if key != ("host", kind) or end <= low or start >= high:
                continue
            a, b = max(start, low), min(end, high)
            x, w = left + (a - low) / (high - low) * width, (b - a) / (high - low) * width
            svg.append(f'<rect x="{x}" y="{y}" width="{max(w,.08)}" height="19" fill="#334155" '
                       f'opacity=".6"><title>{html.escape(kind)} {(b-a)/1000:.6f} ms</title></rect>')
    svg.append('</svg>')
    figure = "".join(svg)
    (output / "median-cycle.svg").write_text(figure)
    table = '<table><tr><th>活动/空档位置</th><th>ms/token</th><th>占比</th></tr>'
    table += ''.join(f"<tr><td>{html.escape(row['label'])}</td><td>{row['ms_per_token']:.6f}</td>"
                     f"<td>{row['period_pct']:.3f}%</td></tr>" for row in rows) + '</table>'
    stages_html = '<table><tr><th>完整token所处的host阶段</th><th>ms/token</th><th>占比</th></tr>'
    stages_html += ''.join(f'<tr><td>{html.escape(name)}</td><td>{value:.6f}</td>'
                           f'<td>{value / period_ms * 100:.3f}%</td></tr>'
                           for name, value in host_stages.items()) + '</table>'
    modules_html = ('<table><tr><th>功能</th><th>Rank0 ms</th><th>Rank1 ms</th>'
                    '<th>Rank2 ms</th><th>Rank3 ms</th><th>四卡并集 ms</th></tr>')
    for row in modules:
        modules_html += '<tr><td>' + html.escape(row['category']) + '</td>'
        modules_html += ''.join(f'<td>{value:.6f}</td>'
                                for value in [*row['per_rank_ms'], row['global_union_ms']]) + '</tr>'
    modules_html += '</table>'
    legend = ' · '.join(f'<span style="color:{colors[name]}">{html.escape(name)}</span>' for name in categories)
    selection_note = ('<p><strong>这是单周期案例：</strong>该token的实测周期接近原22.782107ms，'
                      '用于说明约22ms的工作构成，不代表原三轮平均值的事后还原或整个采集分布。'
                      '<a href="../latency-report/index.html">查看全部31周期与分布</a>。</p>' if selection else '')
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>TP4 decode latency</title>'
        '<style>body{font:15px system-ui;margin:28px;color:#18202a}td,th{padding:8px 16px;'
        'border-bottom:1px solid #ddd;text-align:left}.timeline{overflow:auto}table{border-collapse:collapse}</style>'
        f'<h1>TP4 decode：{period_ms:.6f} ms/token 的同窗口时间账</h1>'
        f'<p>四卡共同的{count}个完整40层消费周期。本页不将profiler区间折算为无profiler测速结果。</p>' + selection_note +
        '<h2>完整token经过哪些阶段</h2><p>入口墙钟包络包含提交和等待。四项合计为完整周期，'
        '硬件kernel在这些包络内执行，不能再次相加。</p>' + stages_html + '<h2>四卡硬件活动及空档</h2>'
        '<p>设备kernel按实际模块归类；四卡均无可见设备活动时，按rank0所在host入口标明位置。'
        '入口包络包含等待，NIC不可见，不能把空档全部归因于CPU或通信。</p>' + table + '<h2>各模块kernel活动并集</h2><p>各模块可重叠，表中各行不能相加；四卡并集不是四卡耗时之和。</p>' +
        modules_html + f'<h2>{"案例" if selection else "中位"}周期 P{tokens[median_index]}：'
        f'{periods[median_index]:.6f} ms</h2><p>悬停查看实际区间。</p>'
        '<div class="timeline">' + figure + '</div><p>' + legend + '</p>'
        '<p><a href="REPORT.md">模块时间表和解释</a> · <a href="latency-ledger.json">完整时间账JSON</a> · '
        f'<a href="{html.escape(kernel_report_link, quote=True)}">'
        '四卡全部kernel/shape/dtype/调用数</a></p></html>')
    (output / "index.html").write_text(page)
    print(json.dumps(
        {
            key: value
            for key, value in details.items()
            if key in ("period_mean_ms", "period_median_ms", "partition", "rank0_no_device_other_rank_active_ms")
        },
        ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-report-link", default="../kernel-report/index.html")
    args = parser.parse_args()
    report(args.analysis, args.output, args.kernel_report_link)
