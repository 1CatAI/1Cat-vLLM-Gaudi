# SPDX-License-Identifier: Apache-2.0
"""Build a same-clock, four-rank, disjoint functional trace ledger.

Fixed priority is an accounting convention, not causal critical-path ownership.
NIC point events are never expanded into communication duration. Unknown gaps
stay explicit. Detailed kernel durations remain independent activity unions.
"""

import argparse
import bisect
import collections
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path

from analyze_deepseek_v41_trace import symbols

GROUPS = (
    "路由专家：MXFP4 解码、W13/W2 BMM、专家激活",
    "Attention：输入/Q/输出投影与 MLA",
    "Router、共享专家及 MoE 准备",
    "mHC、归一化及尚未还原的融合表达式",
    "CSA2：KV/index/RoPE 与注意力状态准备",
    "已记录 DMA/NIC：数据搬运与命令发布",
    "Embedding、输出头、采样及其他入口/尾部张量",
    "Engram 设备投影",
    "无已记录设备区间：Graph/CPU/TP/PP 待进一步归因",
)


def merge(spans):
    result = []
    for left, right in sorted(spans):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


def subtract(spans, covered):
    result, index = [], 0
    for left, right in spans:
        while index < len(covered) and covered[index][1] <= left:
            index += 1
        cursor = index
        while cursor < len(covered) and covered[cursor][0] < right:
            lo, hi = covered[cursor]
            if lo > left:
                result.append((left, min(lo, right)))
            left = max(left, hi)
            if left >= right:
                break
            cursor += 1
        if left < right:
            result.append((left, right))
    return result


def length(spans):
    return sum(b - a for a, b in spans)


def group_for(row):
    # Charge new matrix-attention copies to its complete changed chain.
    if row["category"] == "Attention" and row["purpose"].startswith("MLA "):
        return 1
    if row["engine"] in ("DMA", "NIC"):
        return 5
    cat = row["category"]
    if cat == "路由专家":
        return 0
    if cat == "Attention":
        return 1
    if cat in ("Router", "共享专家", "MoE 准备"):
        return 2
    if cat in ("mHC", "融合表达式待细分"):
        return 3
    if cat in ("CSA2", "Attention/CSA2"):
        return 4
    if cat == "Engram":
        return 7
    return 6


def write_browser(path, rows):
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    path.write_text('''<!doctype html><meta charset="utf-8"><title>V4.1 kernel details</title>
<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;width:100%}
th,td{padding:8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}
th{position:sticky;top:0;background:white}pre{white-space:pre-wrap;max-width:80vw}input{width:55%;padding:8px}
</style><h1>完整 kernel 拆解（ms）</h1><p>活动可重叠，不能逐行相加。单次/调用数为空表示边界未可靠重建，lane 包数另列。
DMA/NIC 命令描述符没有可推定的矩阵 dtype/shape。分组互斥账本见 REPORT.md。</p>
<select id="rank"><option value="">全部 rank</option><option>0</option>
<option>1</option><option>2</option><option>3</option></select>
<input id="query" placeholder="搜索分组、功能、GUID、shape、源节点"><span id="count"></span>
<table><thead><tr><th>rank / 分组 / 功能</th><th>Kernel / 操作数</th>
<th>单次 ms</th><th>活动 ms/token</th><th>占比</th><th>调用/token</th>
<th>合同</th></tr></thead><tbody id="body"></tbody></table>
<script>const rows=''' + payload + ''';
const esc=v=>String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const num=v=>v===null?'未知':Number(v).toFixed(6);
function render(){const q=document.querySelector('#query').value.toLowerCase(),r=document.querySelector('#rank').value;
const all=rows.filter(x=>(r===''||String(x.rank)===r)&&JSON.stringify(x).toLowerCase().includes(q));
document.querySelector('#count').textContent=` ${all.length} 项；每页显示前 500 项，搜索可缩小范围`;
document.querySelector('#body').innerHTML=all.slice(0,500).map(x=>`
<tr><td>${x.rank} / ${esc(x.group)}<br>${esc(x.purpose)}</td>
<td>${esc(x.kernel)}<pre>${esc(JSON.stringify({inputs:x.inputs.map(v=>[v.dtype,v.shape]),outputs:x.outputs.map(v=>[v.dtype,v.shape])}))}</pre></td>
<td>${num(x.mean_invocation_ms)}</td><td>${num(x.activity_ms_per_token)}</td><td>${num(x.period_pct)}%</td><td>${num(x.calls_per_token)}</td>
<td><details><summary>源节点、SRAM/DRAM、计数</summary><pre>${esc(JSON.stringify(x,null,2))}</pre></details></td></tr>`).join('');}
document.querySelector('#query').oninput=render;document.querySelector('#rank').onchange=render;render();</script>''')


def account(root):
    common = json.loads((root / "common-windows.json").read_text())
    windows, tokens = common["windows_us"], common["tokens"]
    ends = [end for _, end in windows]
    scale, period = len(tokens) * 1000, length(windows)
    all_groups = collections.defaultdict(list)
    rank_activity, all_details, kernel_members = [], [], []
    engines = collections.defaultdict(list)
    for rank in range(4):
        path = root / f"rank{rank}"
        inventory = json.loads((path / "inventory.json").read_text())
        recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
        mapped = symbols(inventory, recipes)
        details = json.loads((path / "node-breakdown.json").read_text())
        rows = {(r["recipe_id"], r["engine"], r["context_id"]): r for r in details}
        index_group = {}
        for index, node in enumerate(inventory["nodes"]):
            symbol = mapped.get(index)
            rid = node["recipe"].split(":")[0] if symbol else node["recipe"]
            key = (rid, node["engine"], symbol["full_context_id"] if symbol else index)
            if key in rows:
                index_group[index] = group_for(rows[key])
        grouped, engine_spans = collections.defaultdict(list), collections.defaultdict(list)
        with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
            for line in stream:
                start, duration, _, index = json.loads(line)
                win = bisect.bisect_right(ends, start)
                # A rare long event may cross a token boundary; preserve both sides.
                while win < len(windows) and windows[win][0] < start + duration:
                    left, right = max(start, windows[win][0]), min(start + duration, windows[win][1])
                    if left < right:
                        if index not in index_group:
                            raise RuntimeError(f"Missing measured kernel attribution rank={rank}, index={index}")
                        grouped[index_group[index]].append((left, right))
                        engine_spans[inventory["nodes"][index]["engine"]].append((left, right))
                    win += 1
        merged = {group: merge(spans) for group, spans in grouped.items()}
        for group, spans in merged.items():
            all_groups[group].extend(spans)
        per_engine = {engine: merge(spans) for engine, spans in engine_spans.items()}
        for engine, spans in per_engine.items():
            engines[engine].extend(spans)
        rank_activity.append({
            "rank": rank,
            "period_ms": period / scale,
            "group_activity_ms": {
                GROUPS[g]: length(s) / scale
                for g, s in merged.items()
            },
            "engine_activity_ms": {
                e: length(s) / scale
                for e, s in per_engine.items()
            },
            "all_device_ms": length(merge([s for spans in merged.values() for s in spans])) / scale
        })
        for row in details:
            row["group"] = GROUPS[group_for(row)]
            all_details.append(row)
            kernel_members.append({
                k: row[k]
                for k in ("rank", "group", "category", "purpose", "kernel", "recipe_id", "context_id", "engine",
                          "mean_invocation_ms", "observed_calls", "calls_per_token", "observed_lane_packets",
                          "activity_ms_per_token", "period_pct")
            })
        print(f"rank{rank}: {len(details)} node contracts retained", flush=True)
    disjoint, covered, raw_groups = [], [], {}
    for group in range(8):
        spans = merge(all_groups[group])
        raw_groups[group] = spans
        exclusive = subtract(spans, covered)
        disjoint.append({
            "group": GROUPS[group],
            "exclusive_ms": length(exclusive) / scale,
            "activity_union_ms": length(spans) / scale,
            "share_percent": length(exclusive) / period * 100
        })
        covered = merge(covered + spans)
    gap = subtract(merge(windows), covered)
    disjoint.append({
        "group": GROUPS[8],
        "exclusive_ms": length(gap) / scale,
        "activity_union_ms": length(gap) / scale,
        "share_percent": length(gap) / period * 100
    })
    assert abs(sum(row["exclusive_ms"] for row in disjoint) - period / scale) < 1e-7
    overlap = {}
    for a in range(8):
        for b in range(a + 1, 8):
            value = length(raw_groups[a]) - length(subtract(raw_groups[a], raw_groups[b]))
            if value > 0:
                overlap[f"{a}:{b}"] = value / scale
    expert_decode = collections.defaultdict(float)
    for rank in range(4):
        screen = json.loads((root / f"rank{rank}/activity-screen.json").read_text())
        # Names remain explicit: this is not a substitute for the per-node union.
        expert_decode[str(rank)] = {
            r["kernel"]: r["activity_ms_per_token"]
            for r in screen["rows"] if "mxfp4" in r["kernel"]
        }
    result = {
        "method": "same trace clock; global four-rank union; fixed group priority 0..7; complement 8",
        "priority_is_not_causal_ownership": True,
        "anchor": common["anchor"],
        "tokens": tokens,
        "period_ms": period / scale,
        "groups": disjoint,
        "per_rank": rank_activity,
        "pairwise_overlap_ms": overlap,
        "pairwise_overlap_must_not_be_summed": True,
        "expert_kernel_activity_per_rank_ms": expert_decode,
        "unattributed_intervals_us": gap,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    (root / "four-rank-disjoint-accounting.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (root / "four-rank-kernel-members.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernel_members[0]))
        writer.writeheader()
        writer.writerows(kernel_members)
    write_browser(root / "kernels.html", all_details)
    # Keep localized report fields on separate lines for display-width checks.
    # yapf: disable
    lines = [
        "# 四 rank 完整 trace 拆解", "",
        f"采集：{root.parent.name}。保留 {len(tokens)} 个完整周期，"
        f"token {min(tokens)}–{max(tokens)}；共同边界：{common['anchor']}。", "",
        "主表为同一时钟下四 rank 活动的固定优先级互斥账本。重叠优先归入表中较前组，"
        "此顺序不代表关键路径所有权；活动并集列允许重叠，不能直接相加。", "",
        "| Kernel／边界组 | 互斥 ms/token | 占比 | 该组活动并集 ms/token |",
        "| --- | ---: | ---: | ---: |"
    ]
    for row in disjoint:
        lines.append(f"| {row['group']} | {row['exclusive_ms']:.6f} | {row['share_percent']:.3f}% | "
                     f"{row['activity_union_ms']:.6f} |")
    lines.extend([
        f"| **完整窗口** | **{period / scale:.6f}** | **100%** | — |", "",
        "完整精度字段严格相加等于窗口；六位小数显示可能存在末位舍入。无设备事件区间尚不能全部归因为 CPU、"
        "通信或可消除空闲；没有用 NIC 点事件包络构造通信耗时。Engram host hash/gather/DMA 等待可能在补集中，"
        "不等于表中 Engram 设备投影时间。", "",
        "[可搜索完整 kernel/张量报告](kernels.html) · [逐节点汇总 CSV](four-rank-kernel-members.csv) · "
        "[互斥区间及重叠 JSON](four-rank-disjoint-accounting.json)", "",
        "每个 rank 的 node-breakdown.json 保存实际 GUID、源节点、编译图、输入/输出 dtype、shape、"
        "SRAM/DRAM、平均完整调用延迟、活动并集、物理调用数和 lane 包数。"
        "无法重建的 DMA 描述符调用数及其单次延迟显式为空。", "",
        "| rank | TPC 并集 ms | MME 并集 ms | 全设备并集 ms |", "| --- | ---: | ---: | ---: |"
    ])
    for row in rank_activity:
        lines.append(f"| {row['rank']} | {row['engine_activity_ms'].get('TPC', 0):.6f} | "
                     f"{row['engine_activity_ms'].get('MME', 0):.6f} | {row['all_device_ms']:.6f} |")
    for group in range(8):
        lines.extend([
            "", f"## {GROUPS[group]}", "",
            "下列为按 GUID、用途、dtype/shape 聚合的较大项目；完整小项在上方报告。不同 rank 分别列出，"
            "不能将这些活动直接相加当作整组耗时。", "",
            "| rank | 功能 | Kernel/GUID | 输入 dtype:shape | 输出 dtype:shape |"
            " 单次 ms | 次/token | 活动 ms/token | 占比 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"
        ])
        # yapf: enable
        top = []
        for rank in range(4):
            rows = json.loads((root / f"rank{rank}/kernel-breakdown.json").read_text())["kernel_rows"]
            top.extend([r for r in rows if group_for(r) == group][:2])
        for row in top:
            shapes = [html.escape(json.dumps(s, ensure_ascii=False)) for s in row["dtype_shapes"]]
            single = "未知" if row["mean_invocation_ms"] is None else f"{row['mean_invocation_ms']:.6f}"
            count = "未知" if row["calls_per_token"] is None else f"{row['calls_per_token']:.3f}"
            lines.append(f"| {row['rank']} | {row['purpose']} | `{row['kernel']}` | {shapes[0]} | {shapes[1]} | "
                         f"{single} | {count} | {row['activity_ms_per_token']:.6f} | {row['period_pct']:.3f}% |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"period_ms": result["period_ms"], "groups": disjoint}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    account(parser.parse_args().analysis)
