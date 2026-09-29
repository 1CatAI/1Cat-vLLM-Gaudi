# SPDX-License-Identifier: Apache-2.0
"""Render complete phase/rank/kernel tables with searchable tensor contracts."""
import argparse
import base64
import gzip
import html
import json
import os
from pathlib import Path


def number(value):
    return "未知" if value is None else f"{value:.6f}"


def node_details(row, details):
    # Each device engine has its own context namespace in the same recipe.
    by_node = (details if isinstance(details, dict) else {
        (item["recipe_id"], item["engine"], item["context_id"]): item
        for item in details
    })
    assert len(by_node) == len(details), "Ambiguous recipe/engine/context identity"
    keys = [tuple(key) if len(key) == 3 else (key[0], row["engine"], key[1]) for key in row["node_keys"]]
    return [by_node[key] for key in keys]


def render(analyses, output):
    output.mkdir(parents=True, exist_ok=True)
    reports, rows, links = [], [], []
    node_fields = ("recipe_id", "engine", "context_id", "source_node", "mean_invocation_ms",
                   "complete_invocation_samples", "observed_calls", "count_limitations", "classification_provenance")
    tensor_fields = ("name", "shape", "dtype", "bytes", "location", "strides", "alias")
    # Keep complete records in a compressed export; the standalone browser
    # view needs only compact tensor/layout details and links to full sources.
    with gzip.open(output / "all-kernels.json.gz", "wt", compresslevel=1) as complete:
        complete.write("[\n")
        first = True
        for root in analyses:
            for rank in range(4):
                report = json.loads((root / f"rank{rank}/kernel-breakdown.json").read_text())
                details = json.loads((root / f"rank{rank}/node-breakdown.json").read_text())
                by_node = {(item["recipe_id"], item["engine"], item["context_id"]): item for item in details}
                assert len(by_node) == len(details), "Ambiguous recipe/engine/context identity"
                reports.append({**report, "source": str(root.resolve())})
                links.append((f"{report.get('phase', 'decode')} / Rank {rank} 全部节点与张量",
                              os.path.relpath(root / f"rank{rank}/node-breakdown.json", output)))
                for item in report["kernel_rows"]:
                    row = dict(item, phase=report.get("phase", "decode"), unit=report.get("unit", "token"))
                    row["nodes"] = node_details(row, by_node)
                    if not first:
                        complete.write(",\n")
                    complete.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    first = False
                    compact = []
                    for node in row["nodes"]:
                        entry = {key: node.get(key) for key in node_fields}
                        entry["compiler_graph"] = (node.get("compiler_contract") or {}).get("graph")
                        for side in ("inputs", "outputs"):
                            entry[side] = [{key: tensor.get(key) for key in tensor_fields} for tensor in node[side]]
                        compact.append(entry)
                    rows.append({**row, "nodes": compact})
        complete.write("\n]\n")
    md = [
        "# TP4 prefill / decode 完整 kernel 拆解", "",
        "计时使用同次采集、相同完整消费周期。每行是活动并集，占比可能因重叠而相加超过 100%。"
        "未归因时间保留；平均单次时长与调用数只使用可重建的完整物理调用。", "",
        "| 阶段 | Rank | 单位 | 完整周期 ms | 样本数 | 未知调用数的节点 |",
        "|---|---:|---|---:|---:|---:|"
    ]
    for report in reports:
        md.append(f"| {report.get('phase', 'decode')} | {report['rank']} | {report.get('unit', 'token')} | "
                  f"{number(report['period_ms'])} | {len(report['tokens'])} | {report['nodes_with_unknown_calls']} |")
    for report in reports:
        phase, unit, rank = report.get("phase", "decode"), report.get("unit", "token"), report["rank"]
        md += ["", f"## {phase} / Rank {rank}", "", "互斥时间分区（ms/" + unit + "）：", ""]
        md += [f"- {name}: {number(value)}" for name, value in report["partition"].items()]
        md += ["", "全部设备引擎互斥分区（含 NIC/DMA，ms/" + unit + "）：", ""]
        md += [
            f"- {name}: {number(value)}"
            for name, value in report.get("device_engine_presence_partition_ms", {}).items()
        ]
        md += [
            "", "设备事件可见性：" + "; ".join(f"{name}: {status}"
                                       for name, status in report.get("device_observability", {}).items()), ""
        ]
        md += ["", "CPU/运行时观测活动（各行可嵌套；不能相加或当作因果归因）：", "",
               "| 类型 | 活动 ms | 与计算重叠 ms | 计算之外 ms |", "|---|---:|---:|---:|"]
        for name, item in report.get("host_observed_activity", {}).items():
            md.append(f"| {name} | {number(item['activity_ms'])} | {number(item['overlap_with_compute_ms'])} | "
                      f"{number(item['outside_compute_ms'])} |")
        categories = sorted({row["category"] for row in report["kernel_rows"]})
        for category in categories:
            md += [
                "", f"### {category}", "",
                f"| 功能 | Kernel | A/B/输出 dtype 与形状 | 平均单次 ms | 次/{unit} | ms/{unit} | 周期占比 |",
                "|---|---|---|---:|---:|---:|---:|"
            ]
            for row in report["kernel_rows"]:
                if row["category"] != category:
                    continue

                def safe(value):
                    return str(value).replace("|", "\\|").replace("\n", " ")

                shapes = safe(json.dumps(row["dtype_shapes"], ensure_ascii=False))
                md.append(f"| {safe(row['purpose'])} | {safe(row['kernel'])} | {shapes} | "
                          f"{number(row['mean_invocation_ms'])} | {number(row['calls_per_' + unit])} | "
                          f"{number(row['activity_ms_per_' + unit])} | {row['period_pct']:.3f}% |")
    (output / "REPORT.md").write_text("\n".join(md) + "\n")
    data = base64.b64encode(
        gzip.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode(), compresslevel=6)).decode()
    export_links = " ".join(f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>'
                            for label, path in links)
    colors = ["#2878b5", "#dc7d29", "#7660b5", "#489c82", "#c5cbd3"]
    labels = ["仅 TPC", "仅 MME", "TPC/MME 重叠", "其他已记录设备活动", "无已记录设备活动（未归因）"]
    overview = ['<table><thead><tr><th>阶段/rank</th><th>完整周期</th><th>互斥时间分区</th></tr></thead><tbody>']
    for report in reports:
        pieces = []
        for (key, value), color, label in zip(report["partition"].items(), colors, labels, strict=True):
            fraction = value / report["period_ms"] * 100
            title = html.escape(f"{label}: {value:.6f} ms ({fraction:.3f}%)", quote=True)
            pieces.append(f'<span title="{title}" style="width:{fraction:.6f}%;background:{color}"></span>')
        overview.append(f'<tr><td>{report["phase"]} / {report["rank"]}</td><td>{report["period_ms"]:.6f} '
                        f'ms/{report["unit"]}</td><td><div class="bar">' + "".join(pieces) + '</div></td></tr>')
    overview.append('</tbody></table><p>' + ' · '.join(f'<span style="color:{color}">{label}</span>'
                                                       for color, label in zip(colors, labels, strict=True)) + '</p>')
    overview = "".join(overview)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>TP4 kernel trace</title>
<style>body{font:14px system-ui;margin:24px;color:#18202a}h1{font-size:22px}input,select{padding:8px;margin-right:10px}
table{border-collapse:collapse;width:100%;margin-top:16px}th,td{padding:8px;border-bottom:1px solid #d5dde5;text-align:left}
th{position:sticky;top:0;background:#eef3f7}td{vertical-align:top}pre{white-space:pre-wrap;max-width:100%;font-size:12px}
.bar{display:flex;width:440px;height:18px;background:#eee}.bar span{height:18px}small{color:#526272}.num{text-align:right;white-space:nowrap}details{max-width:520px}summary{cursor:pointer}</style>
<h1>TP4 prefill / decode 完整 kernel 拆解</h1>
<p>各行活动可重叠；占比以同次 trace 的完整周期为分母。未能可靠恢复的物理调用数和单次时间显示“未知”。</p>
OVERVIEW
<input id="search" placeholder="搜索模块、投影、kernel 或形状"><select id="phase"><option value="">所有阶段</option></select>
<select id="rank"><option value="">所有 rank</option></select>
<select id="engine"><option value="">所有引擎</option></select><select id="category"><option value="">所有模块</option></select>
<p>完整数据：<a href="all-kernels.json.gz">全部 kernel JSON.gz</a> · <a href="REPORT.md">完整 Markdown</a></p>
<details><summary>各 rank 原始节点、完整物理布局与编译图</summary>EXPORT_LINKS</details>
<p id="count">正在加载本地数据…</p><button id="previous">上一页</button> <button id="next">下一页</button>
<table><thead><tr><th>阶段/rank</th><th>模块/功能与明细</th><th>Kernel</th><th>平均单次 ms</th>
<th>调用数/单位</th><th>活动 ms/单位</th><th>周期占比</th></tr></thead><tbody id="body"></tbody></table>
<script>(async()=>{const encoded=atob('DATA');
const bytes=Uint8Array.from(encoded,c=>c.charCodeAt(0));
const decoded=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
const rows=JSON.parse(decoded);let page=0;const pageSize=100;
for(const r of rows)r.searchText=(r.category+' '+r.purpose+' '+r.kernel+' '+JSON.stringify(r.dtype_shapes)).toLowerCase();
for(const key of ['phase','rank','engine','category']){const select=document.getElementById(key);for(const value of [...new Set(rows.map(r=>String(r[key])))].sort()){
 const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option);}}
const fmt=x=>x===null?'未知':Number(x).toFixed(6);
function draw(){const q=document.getElementById('search').value.toLowerCase();const filtered=rows.filter(r=>['phase','rank','engine','category'].every(k=>!document.getElementById(k).value||String(r[k])===document.getElementById(k).value)&&(!q||r.searchText.includes(q)));
 page=Math.min(page,Math.max(0,Math.ceil(filtered.length/pageSize)-1));
 document.getElementById('count').textContent=filtered.length+' / '+rows.length+' 条；第 '+(page+1)+' 页，每页 '+pageSize+' 条；prefill 单位为 request，decode 为 token';
 const body=document.getElementById('body');body.replaceChildren();for(const r of filtered.slice(page*pageSize,(page+1)*pageSize)){const tr=document.createElement('tr');
 const cells=[r.phase+' / '+r.rank,null,r.engine+' / '+r.kernel,fmt(r.mean_invocation_ms),fmt(r['calls_per_'+r.unit]),fmt(r['activity_ms_per_'+r.unit]),r.period_pct.toFixed(3)+'%'];
 cells.forEach((value,i)=>{const td=document.createElement('td');if(i===1){const d=document.createElement('details');const s=document.createElement('summary');s.textContent=r.category+'：'+r.purpose;d.append(s);d.addEventListener('toggle',()=>{if(d.open&&!d.querySelector('pre')){const pre=document.createElement('pre');pre.textContent=JSON.stringify({dtype_shapes:r.dtype_shapes,nodes:r.nodes},null,2);d.append(pre);}});td.append(d);}else{td.textContent=value;if(i>=3)td.className='num';}tr.append(td);});body.append(tr);}}
for(const id of ['search','phase','rank','engine','category'])document.getElementById(id).addEventListener('input',()=>{page=0;draw();});
document.getElementById('previous').onclick=()=>{page=Math.max(0,page-1);draw();};
document.getElementById('next').onclick=()=>{page++;draw();};draw();})().catch(e=>{document.getElementById('count').textContent='加载失败：'+e;});</script></html>'''  # noqa: E501 - embedded HTML/CSS/JavaScript template
    (output / "index.html").write_text(
        page.replace("DATA", data).replace("EXPORT_LINKS", export_links).replace("OVERVIEW", overview))
    print(json.dumps({"reports": len(reports), "kernel_rows": len(rows), "output": str(output.resolve())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.analysis, args.output)
