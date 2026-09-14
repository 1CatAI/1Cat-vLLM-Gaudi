# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Account for an entire four-rank acquisition, including capture and DSpark.

Recipe/tensor joins and pre-graph slicing follow the archived V4 and V4.1
reporters. Invocation reconstruction uses this acquisition's engine contracts
and recipe frequency, with no inherited layer pattern or clustering threshold.
"""

import argparse
import collections
import csv
from functools import lru_cache
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

from map_deepseek_v41_trace_contracts import graph_nodes, tensor


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def duration(spans):
    return sum(b - a for a, b in merged(spans))


def timestamp_marker(kind):
    return (kind in ("TPC_SPU_START", "DBG_DMA_TRC_RD_FRST_ADDR_PUSH")
            or bool(re.match(r"STM_[01]_(RX|TX|QPC|QMAN)", kind)))


def kernel_label(name):
    if "pdma_" in name:
        return re.sub(r"\s+apiId=\d+\s+chunk=\d+$", "", name)
    return name


def io(node, prefix):
    return [
        tensor(value) for key, value in sorted(node["attrs"].items(),
                                               key=lambda item: (int(item[0].rsplit(":", 1)[1])
                                                                 if item[0].startswith(prefix) else -1))
        if key.startswith(prefix)
    ]


@lru_cache(maxsize=256)
def graph(path):
    data = graph_nodes(Path(path))
    return {
        "nodes": data,
        "by_name": {
            n["name"]: n
            for n in data
        },
        "producers": {
            t["name"]: n
            for n in data
            for t in io(n, "outputTensor:")
        },
        "aliases": {
            t["name"]: t["alias"]
            for n in data
            for t in io(n, "inputTensor:") + io(n, "outputTensor:") if t["alias"]
        }
    }


def roots(data, name):
    result = set()
    while name not in result:
        result.add(name)
        short = re.sub(r"^(\d+)_\d+$", r"\1", name.split("_slice_")[0])
        result.add(short)
        if name in data["aliases"]:
            name = data["aliases"][name]
        elif short != name:
            name = short
        else:
            break
    return result


def expression(contract):
    if not contract or not contract.get("graph"):
        return [], "no compiler graph"
    path = Path(contract["graph"]["path"])
    pre = Path(str(path).replace("-PostGraph-", "-PreGraph-"))
    if pre == path or not pre.is_file():
        return [], "pre-graph absent"
    post, before = graph(str(path)), graph(str(pre))
    incoming = set().union(*(roots(post, t["name"]) for t in contract["inputs"]))
    pending = [
        name for t in contract["outputs"] for name in roots(post, t["name"])
        if name in before["producers"] and name not in incoming
    ]
    if not pending:
        return [], "output ancestry unresolved"
    seen, found = set(), {}
    while pending:
        name = pending.pop()
        if name in seen or name in incoming:
            continue
        seen.add(name)
        node = before["producers"].get(name)
        if node is None:
            continue
        if (node["op"] in ("GEMM", "BatchGemm") or node["op"].startswith("custom_") or "linear_fwd" in node["op"]
                or "batch_gemm" in node["op"]):
            return [], "ancestry crosses an independent matrix/custom operation"
        found[node["name"]] = node
        pending.extend(t["name"] for t in io(node, "inputTensor:") if t["name"] not in incoming)
    nodes = sorted(found.values(), key=lambda n: int(n["attrs"].get("Exec_idx", "0")))
    return [{"node": n["name"], "operation": n["op"]} for n in nodes], "pre-graph boundary slice"


def symbols(inventory, recipes):
    names = collections.defaultdict(list)
    for recipe in recipes:
        for node in recipe["nodes"]:
            names[(str(recipe["recipe_id"]), node["device_type"], node["node"], node["kernel"])].append(node)
    result = {}
    for index, node in enumerate(inventory["nodes"]):
        key = (node["recipe"].split(":")[0], {
            "TPC": 1,
            "MME": 0,
            "DMA": 8
        }.get(node["engine"]), node["node"], node["kernel"])
        candidates = names.get(key, [])
        if candidates and all(c == candidates[0] for c in candidates):
            result[index] = candidates[0]
    return result


def reconstruct(rows, packet_count=None, frequency=None, single_roi=False):
    """Return complete-call wall spans, never union/count pseudo-latency."""
    rows = sorted(rows)
    if not rows:
        return [], "no complete hardware events"
    if packet_count is None:
        if not frequency or len(rows) % frequency:
            return [], "recipe frequency does not establish packet groups"
        packet_count = len(rows) // frequency
    if not packet_count or len(rows) % packet_count:
        return [], "incomplete packet group"
    groups = [rows[i:i + packet_count] for i in range(0, len(rows), packet_count)]
    if frequency is not None and len(groups) != frequency:
        return [], "working-engine count disagrees with recipe frequency"
    if single_roi and any(len({r[2] for r in g}) != packet_count for g in groups):
        return [], "single-ROI lane coverage is incomplete"
    signatures = [collections.Counter((r[2], r[3]) for r in g) for g in groups]
    if any(s != signatures[0] for s in signatures):
        return [], "lane/port multiplicity changes across reconstructed calls"
    spans = [(min(r[0] for r in g), max(r[1] for r in g)) for g in groups]
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return [], "successive calls of one recipe context overlap ambiguously"
    return spans, None


def classify(name, kernel, inputs, outputs, origins):
    source = " ".join([name, *(n["node"] for n in origins)]).lower()
    weight = inputs[1]["shape"] if len(inputs) > 1 else []
    shapes = [t["shape"] for t in inputs + outputs]
    if "mxfp4" in source or "mxfp4" in kernel:
        stage = "W13 gate/up" if any(s and 2304 in s for s in shapes) else "W2 down"
        return "路由专家", stage + (" / BF16 矩阵计算" if kernel in ("GEMM", "BatchGemm") else " / ID 寻址、解码及相关处理")
    if "bf16_identity" in kernel:
        return "路由专家", "结果复制 / 编译块完成依赖"
    if "quant_roundtrip" in kernel:
        return "激活量化", "group-32 BF16 → E4M3FN 编码 → BF16；MME 精度不由此改变"
    if "sinkhorn" in kernel:
        return "mHC", "4×4 Sinkhorn"
    if "sparse_attn" in kernel:
        return "Attention/CSA2", "QK → online softmax → V 累加"
    if kernel in ("GEMM", "BatchGemm"):
        if weight == [24, 20480]:
            return "mHC", "控制投影"
        if weight in ([384, 5120], [128, 5120]):
            return "Router", "专家路由打分"
        if weight == [25600, 6144]:
            return "Engram", "host 查询行的 key/value 投影"
        if "markov" in source:
            return "DSpark Markov", "Markov 词表偏置投影"
        if "confidence" in source or weight == [1, 5376]:
            return "DSpark 置信度", "置信度投影"
        if "main_proj" in source or weight == [5120, 15360]:
            return "DSpark context", "target states 合并投影"
        if weight == [64640, 256]:
            return "DSpark Markov", "TP Markov 词表偏置投影"
        draft_projections = {
            (1280, 5120): "wq_a 输入投影",
            (16384, 1280): "wq_b Q 展开",
            (512, 5120): "wkv 投影",
            (5120, 4096): "wo_b 输出投影"
        }
        if inputs and inputs[0]["shape"] and inputs[0]["shape"][0] == 5:
            label = draft_projections.get(tuple(weight or []))
            if label:
                return "DSpark Attention", label + "；由 C5 与独有投影形状关联"
        if "attention" in source:
            labels = {
                (1280, 5120): "wq_a 输入投影",
                (16384, 1280): "wq_b Q 展开",
                (512, 5120): "wkv 投影",
                (5120, 4096): "wo_b 输出投影",
                (128, 512): "index K 投影"
            }
            return "Attention/CSA2", ("wo_a 分组输出 BMM" if kernel == "BatchGemm" else labels.get(
                tuple(weight or []), "其他投影 / Compressor；见张量合同"))
        if "/moe/" in source:
            return "共享专家", "down 投影" if weight == [5120, 1152] else "gate/up 投影"
        if weight and weight[0] == 64640:
            return "输出头", "TP 词表投影"
        if any(s and 1024 in s for s in shapes) and ("vision" in source or "blocks" in source):
            return "Vision", "视觉矩阵投影"
        return "其他矩阵", "用途未恢复；完整 A/B/输出合同见明细"
    if kernel == "DmaMemcpy" and inputs and inputs[0]["shape"] == [512, 256, 1]:
        return "Attention/CSA2", "FP4 KV 解包字节重排"
    if "engram" in source:
        return "Engram", "查询行解码 / 门控 / residual 更新"
    if "attention" in source:
        return "Attention/CSA2", "量化 / KV 和 index 状态 / RoPE / 索引处理"
    if "/moe/" in source:
        return "MoE 准备", "路由 / SwiGLU / 加权归约，见源表达式"
    if "pdma_" in kernel or kernel.startswith(("H2D", "D2H", "D2D")):
        return "设备搬运", "PDMA 传输；硬件包不等于 kernel 调用"
    if kernel.startswith("fused_kernel_"):
        return "其他融合表达式", " → ".join(dict.fromkeys(n["operation"] for n in origins)) or "内部表达式未恢复"
    return "入口/尾部/其他", kernel


def eager_contract(node, launches):
    choices = []
    for name in {r[3] for r in launches}:
        path = Path(name + "-eager_final_graph-symbol.pbtxt")
        if not path.exists():
            continue
        match = graph(str(path))["by_name"].get(node["node"])
        if match and match["op"] == node["kernel"]:
            choices.append({
                "inputs": io(match, "inputTensor:"),
                "outputs": io(match, "outputTensor:"),
                "attributes": match["attrs"],
                "graph": {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                }
            })
    if choices and all((c["inputs"], c["outputs"]) == (choices[0]["inputs"], choices[0]["outputs"]) for c in choices):
        return choices[0]
    return None


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for k, v in row.items()
        } for row in rows)


def analyze(root, rank, bounds):
    path = root / f"rank{rank}"
    inv = json.loads((path / "inventory.json").read_text())
    if inv.get("schema_version") != 2:
        raise ValueError("Re-extract this trace with hardware event kinds retained")
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    mapped = symbols(inv, recipes)
    contracts = {
        (str(n["recipe_id"]), n["symbol"]["device_type"], n["symbol"]["full_context_id"]): n
        for n in json.loads((path / "node-contracts.json").read_text())
    }
    launches = collections.defaultdict(list)
    for row in inv["host_enqueues"]:
        launches[row[2].split(":")[0]].append(row)
    groups = collections.defaultdict(list)
    kinds = inv["hw_event_names"]
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            ts, dur, lane, index, kind = json.loads(line)
            groups[index].append((ts - bounds[0], ts + dur - bounds[0], lane, kind))
    complete = {i: [r for r in rows if not timestamp_marker(kinds[r[3]])] for i, rows in groups.items()}
    tpc_calls, votes = {}, collections.defaultdict(collections.Counter)
    for index, symbol in mapped.items():
        if symbol["device_type"] != 1 or index not in groups:
            continue
        spans, reason = reconstruct(complete[index],
                                    sum(symbol["working_engines"]),
                                    single_roi=len(symbol["working_engines"]) == 1)
        tpc_calls[index] = (spans, reason)
        if spans and len(complete[index]) == len(groups[index]):
            votes[inv["nodes"][index]["recipe"].split(":")[0]][len(spans)] += 1
    frequencies, proofs = {}, {}
    for rid, counts in votes.items():
        ordered = counts.most_common()
        # Several independently complete TPC contexts establish recipe
        # frequency. Missing events in a minority context remain explicit;
        # each MME additionally validates every packet group's lane/port set.
        if len(ordered) == 1 or (ordered[0][1] >= 2 and ordered[0][1] > sum(n for _, n in ordered[1:])):
            frequencies[rid] = ordered[0][0]
        proofs[rid] = {
            "validated_TPC_node_frequencies": dict(counts),
            "selected_frequency": frequencies.get(rid),
            "disagreement": len(ordered) > 1
        }
    all_rows, aggregated, engines = [], collections.defaultdict(list), collections.defaultdict(list)
    window = bounds[1] - bounds[0]
    for index, rows in groups.items():
        node, symbol = inv["nodes"][index], mapped.get(index)
        rid = node["recipe"].split(":")[0]
        contract = contracts.get((rid, symbol["device_type"], symbol["full_context_id"])) if symbol else None
        if contract is None:
            contract = eager_contract(node, launches[rid])
        inputs, outputs = (contract.get(p, []) if contract else [] for p in ("inputs", "outputs"))
        origin, origin_method = expression(contract) if node["kernel"].startswith("fused_kernel_") else ([], None)
        category, purpose = classify(node["node"], node["kernel"], inputs, outputs, origin)
        points = len(rows) - len(complete[index])
        spans = [(r[0], r[1]) for r in complete[index]]
        point_spans = [(r[0], r[1]) for r in rows if timestamp_marker(kinds[r[3]])]
        # Start-only markers have synthetic width; retain them without treating
        # that width as a measured compute duration.
        engines[node["engine"]].extend(merged(spans))
        freq = frequencies.get(rid)
        freq_source = "serialized TPC engine-contract agreement"
        if freq is None and symbol is None and contract and launches[rid]:
            freq = len(launches[rid])
            freq_source = "exact eager recipe enqueue identity and tensor contract"
        if index in tpc_calls:
            calls, reason = tpc_calls[index]
            count_source = "serialized TPC ROI working-engine counts"
            if freq is not None and len(calls) != freq:
                reason = "complete TPC samples disagree with independently established recipe frequency"
        elif node["engine"] in ("TPC", "MME"):
            calls, reason = reconstruct(complete[index], frequency=freq)
            count_source = freq_source
        else:
            calls, reason = [], "DMA descriptor-to-kernel invocation boundary not reconstructed"
            count_source = None
        if points:
            reason = "start-only markers present; complete samples do not establish total call count"
        if node["engine"] == "NIC":
            category, purpose = "通信时间戳", "NIC 发送/接收消息；无完整 collective 耗时边界"
            reason = "NIC message timestamps have synthetic display widths, not measured communication duration"
        values = [(b - a) / 1000 for a, b in calls]
        # T=5 is the draft program's static width; target replay admits only
        # C1/C6. Other operations retain their source rather than guessing.
        phase = "DSpark draft" if any(t["shape"] and t["shape"][0] == 5 for t in inputs + outputs) else "target/other"
        row = {
            "rank": rank,
            "node_index": index,
            "recipe_id": rid,
            "context_id": symbol["full_context_id"] if symbol else None,
            "engine": node["engine"],
            "category": category,
            "purpose": purpose,
            "phase_hint": phase,
            "kernel": node["kernel"],
            "source_node": node["node"],
            "reported_dtype": node.get("reported_dtype"),
            "inputs": inputs,
            "outputs": outputs,
            "source_operations": origin,
            "source_recovery": origin_method,
            "compiler_contract": contract,
            "activity_ms": duration(spans) / 1000,
            "window_pct": duration(spans) / window * 100,
            "mean_invocation_ms": statistics.mean(values) if values else None,
            "complete_invocation_samples": len(values),
            "physical_calls": len(calls) if reason is None else None,
            "observed_lane_packets": len(rows),
            "start_only_packets": points,
            "start_marker_display_ms": duration(point_spans) / 1000,
            "count_method": count_source,
            "count_limitation": reason,
            "hw_events": dict(collections.Counter(kinds[r[3]] for r in rows))
        }
        all_rows.append(row)
        dtype_shapes = [[(t["dtype"], t["shape"]) for t in side] for side in (inputs, outputs)]
        # API/chunk IDs identify transport packets, not different kernels.
        # Keep their exact names in node-breakdown.json and union their spans.
        # Different projections remain separate even when their MME GUID is identical.
        key = (category, purpose, phase, node["engine"], kernel_label(node["kernel"]), json.dumps(dtype_shapes))
        aggregated[key].append((row, merged(spans), values))
    summary = []
    for key, items in aggregated.items():
        spans = [s for _, group, _ in items for s in group]
        values = [v for _, _, group in items for v in group]
        known = all(row["physical_calls"] is not None for row, _, _ in items)
        summary.append({
            "rank": rank,
            "category": key[0],
            "purpose": key[1],
            "phase_hint": key[2],
            "engine": key[3],
            "kernel": key[4],
            "dtype_shapes": json.loads(key[5]),
            "mean_invocation_ms": statistics.mean(values) if values else None,
            "complete_invocation_samples": len(values),
            "physical_calls": sum(row["physical_calls"] for row, _, _ in items) if known else None,
            "observed_lane_packets": sum(row["observed_lane_packets"] for row, _, _ in items),
            "activity_ms": duration(spans) / 1000,
            "window_pct": duration(spans) / window * 100,
            "node_indices": [row["node_index"] for row, _, _ in items]
        })
    summary.sort(key=lambda r: -r["activity_ms"])
    tpc, mme = duration(engines["TPC"]), duration(engines["MME"])
    compute = duration(engines["TPC"] + engines["MME"])
    device = duration([s for e in engines.values() for s in e])
    partition = {
        "TPC_only_ms": (compute - mme) / 1000,
        "MME_only_ms": (compute - tpc) / 1000,
        "TPC_MME_overlap_ms": (tpc + mme - compute) / 1000,
        "other_recorded_device_only_ms": (device - compute) / 1000,
        "unattributed_ms": (window - device) / 1000
    }
    assert abs(sum(partition.values()) - window / 1000) < 1e-8
    assert sum(r["observed_lane_packets"] for r in all_rows) == inv["hardware_events"]
    host = collections.defaultdict(list)
    with gzip.open(path / "host.jsonl.gz", "rt") as stream:
        for line in stream:
            ts, dur, _, _, cat, name = json.loads(line)
            start, end = max(ts, bounds[0]), min(ts + dur, bounds[1])
            if end > start:
                host[(cat, name)].append((start - bounds[0], end - bounds[0]))
    host_rows = [{
        "category": key[0],
        "name": key[1],
        "calls": len(spans),
        "activity_ms": duration(spans) / 1000,
        "mean_call_ms": statistics.mean(b - a for a, b in spans) / 1000
    } for key, spans in host.items()]
    host_rows.sort(key=lambda r: -r["activity_ms"])
    # Reconcile each GUID after functional splitting, in the same trace window.
    raw_by_guid, rebuilt_by_guid = collections.defaultdict(list), collections.defaultdict(list)
    for i, rows in complete.items():
        node = inv["nodes"][i]
        raw_by_guid[(node["engine"], kernel_label(node["kernel"]))].extend((r[0], r[1]) for r in rows)
    for key, items in aggregated.items():
        rebuilt_by_guid[(key[3], key[4])].extend(s for _, spans, _ in items for s in spans)
    for key in raw_by_guid:
        assert abs(duration(raw_by_guid[key]) - duration(rebuilt_by_guid[key])) < 1e-6, key
    result = {
        "rank":
        rank,
        "pp":
        rank // 2,
        "tp":
        rank % 2,
        "trace_sha256":
        inv["trace_sha256"],
        "window_ms":
        window / 1000,
        "partition":
        partition,
        "measured_nodes":
        len(all_rows),
        "tensor_contract_nodes":
        sum(r["compiler_contract"] is not None for r in all_rows),
        "nodes_with_known_calls":
        sum(r["physical_calls"] is not None for r in all_rows),
        "fused_nodes_with_origin":
        sum(bool(r["source_operations"]) for r in all_rows),
        "recorded_nic_activity":
        bool(engines["NIC"]),
        "nic_message_markers":
        sum(r["observed_lane_packets"] for r in all_rows if r["engine"] == "NIC"),
        "kernel_rows":
        summary,
        "host_rows":
        host_rows,
        "limitations": [
            "Window includes native capture, prefill, target verification and draft work.",
            "NIC message markers do not establish complete collective durations; "
            "device communication time remains unmeasured unless complete intervals are present.",
            "Unknown physical calls and fused origins are retained explicitly.",
            "Start-only marker widths are excluded from measured engine activity."
        ]
    }
    (path / "recipe-frequency-proofs.json").write_text(json.dumps(proofs, indent=2) + "\n")
    (path / "node-breakdown.json").write_text(json.dumps(all_rows, ensure_ascii=False) + "\n")
    (path / "acquisition-breakdown.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_csv(path / "kernel-breakdown.csv", summary)
    write_csv(path / "host-breakdown.csv", host_rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ("kernel_rows", "host_rows")}), flush=True)
    return result


def coverage_qualification(root):
    path = root / "coverage-audit.json"
    audit = json.loads(path.read_text()) if path.exists() else {"status": "not audited"}
    if audit["status"] == "incomplete hardware coverage":
        message = "硬件覆盖不完整：以下时间、调用数和占比仅属于保留下来的硬件窗口，不能代表完整请求。"
    elif audit["status"] == "request/target bounds consistent":
        message = "请求与 target 时间边界检查一致；仍需结合实际调用数确认没有丢失中间事件。"
    else:
        message = "尚未验证硬件窗口覆盖完整请求；文件导出成功不等于完整采集。"
    return audit, message


def render(root, bounds, results):
    coverage, qualification = coverage_qualification(root)
    rows = [r for result in results for r in result["kernel_rows"]]
    text = [
        "# DeepSeek V4.1：已记录硬件窗口逐 kernel 拆解", "", qualification, "",
        "单位均为 ms。每行活动统计限于已记录窗口（其中可能含捕获和 prefill）；不是稳态 ms/token。", "所有 rank 使用同一硬件时钟窗口。未知调用数以 — 表示，不能把 lane 包数当作调用数。",
        "", "| Rank | 已记录窗口 | 仅 TPC | 仅 MME | TPC/MME 重叠 | 其他设备独占 | 未归因 |", "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for result in results:
        numbers = [result["window_ms"], *result["partition"].values()]
        text.append(f"| {result['rank']} | " + " | ".join(f"{n:.6f}" for n in numbers) + " |")
    text.extend([
        "", "未归因包含其他 PP stage、CPU/提交/依赖等可能性，不能全部称为通信或空闲。", "NIC 消息标记的显示宽度不代表传输时长；HCCL host API 耗时与设备通信时长分别处理。", "",
        "完整表见 [可筛选报告](report.html)。CSV 和节点 JSON 保存张量、recipe/context 与源图关联。", ""
    ])
    for result in results:
        text.extend([f"## Rank {result['rank']}", ""])
        grouped = collections.defaultdict(list)
        for row in result["kernel_rows"]:
            grouped[row["category"]].append(row)
        for category, items in grouped.items():
            text.extend([
                f"### {category}", "", "| 功能 | Kernel | 平均单次 ms | 活动 ms/采集 | 占比 % | 调用数 | 输入/输出 dtype、shape |",
                "|---|---|---:|---:|---:|---:|---|"
            ])
            for row in items:
                single = "—" if row["mean_invocation_ms"] is None else f"{row['mean_invocation_ms']:.6f}"
                calls = "—" if row["physical_calls"] is None else str(row["physical_calls"])
                text.append(f"| {row['purpose']} | `{row['kernel']}` | {single} | {row['activity_ms']:.6f} | "
                            f"{row['window_pct']:.4f} | {calls} | `{json.dumps(row['dtype_shapes'])}` |")
            text.append("")
    (root / "REPORT.md").write_text("\n".join(text) + "\n")
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    document = """<!doctype html><meta charset="utf-8"><title>V4.1 kernel trace</title>
<style>body{font:14px system-ui;margin:24px;background:#fafafa;color:#172334}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd;vertical-align:top}
th{position:sticky;top:0;background:#e5edf7}
input,select{padding:9px;margin:5px}code{word-break:break-all}summary{cursor:pointer}small{color:#526272}</style>
<h1>DeepSeek V4.1 · 已记录硬件窗口</h1><p><strong>COVERAGE_QUALIFICATION</strong></p>
<p>所有时间 ms。可含捕获、prefill、verify、draft；不是稳态 ms/token。
活动重叠不可相加。— 表示未知，调用数不等于 lane 包数。节点明细可展开并访问完整 JSON。</p>
<p><a href="REPORT.md">全部 Markdown</a></p>
<label>Rank <select id="rank"><option value="">全部</option>
<option>0</option><option>1</option><option>2</option><option>3</option></select></label>
<label>引擎 <select id="engine"><option value="">全部</option>
<option>TPC</option><option>MME</option><option>DMA</option><option>NIC</option></select></label>
<input id="search" placeholder="模块 / kernel / shape / dtype" size="52"><p id="count"></p>
<table><thead><tr><th>Rank / 模块</th><th>功能与 kernel</th><th>平均单次 ms</th><th>活动 ms / 占比</th>
<th>物理调用 / 完整样本</th><th>张量与来源</th></tr></thead><tbody id="body"></tbody></table>
<script>const data=PAYLOAD;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x===null?'—':x.toFixed(6);
function render(){const rs=data.filter(r=>(!rank.value||String(r.rank)===rank.value)&&
(!engine.value||r.engine===engine.value)&&JSON.stringify(r).toLowerCase().includes(search.value.toLowerCase()));
document.getElementById('count').textContent=rs.length+' / '+data.length+' 项';
body.innerHTML=rs.map(r=>`<tr><td>${r.rank} · ${esc(r.category)}<br>${esc(r.engine)}</td>
<td>${esc(r.purpose)}<br><code>${esc(r.kernel)}</code></td><td>${fmt(r.mean_invocation_ms)}</td>
<td>${fmt(r.activity_ms)}<br>${r.window_pct.toFixed(4)}%</td>
<td>${r.physical_calls===null?'—':r.physical_calls} / ${r.complete_invocation_samples}
<br><small>lane 包 ${r.observed_lane_packets}</small></td><td><details><summary>形状 / dtype / 节点</summary>
<pre>${esc(JSON.stringify(r.dtype_shapes,null,2))}</pre><p>节点索引 ${esc(r.node_indices.join(', '))}</p>
<a href="rank${r.rank}/node-breakdown.json">完整张量、recipe、源图、计数限制</a></details></td></tr>`).join('');}
for(const e of [rank,engine,search])e.addEventListener('input',render);render();</script>"""
    (root / "report.html").write_text(
        document.replace("COVERAGE_QUALIFICATION", html.escape(qualification)).replace("PAYLOAD", payload))
    (root / "window.json").write_text(
        json.dumps(
            {
                "bounds_us": bounds,
                "units_report": "ms",
                "scope": "common first-to-last recorded hardware event across all four ranks; recorded events only",
                "coverage_status": coverage["status"],
                "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            },
            indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    inventories = [json.loads((args.analysis / f"rank{r}/inventory.json").read_text()) for r in range(4)]
    assert len({i["base_time_nanoseconds"] for i in inventories}) == 1
    limits = min(i["first_us"] for i in inventories), max(i["last_us"] for i in inventories)
    outcomes = [analyze(args.analysis, r, limits) for r in range(4)]
    render(args.analysis, limits, outcomes)
