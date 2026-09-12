# SPDX-License-Identifier: Apache-2.0
"""Attribute all recorded engines within common four-rank device periods.

Tensor and pre-graph dependency joins reuse the archived V4 reporting method.
Missing invocation boundaries remain unknown, especially for EDMA descriptors.
"""

import argparse
import bisect
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from analyze_deepseek_v41_trace import symbols, union
from map_deepseek_v41_trace_contracts import graph_nodes, tensor


def io(node, prefix):
    return [tensor(v) for k, v in sorted(node["attrs"].items(), key=lambda pair: (
        int(pair[0].split(":")[-1]) if pair[0].startswith(prefix) else -1)) if k.startswith(prefix)]


def classify(node, kernel, inputs, outputs):
    name = node.lower()
    if "topk" in name or "bitonic" in kernel:
        return "Router", "专家 Top-k 排序；实际输入见张量合同"
    if "deepseek_v41_control_gemv" in kernel:
        return "mHC", "FP32 控制投影"
    if "deepseek_v41_rope" in kernel or "deepseek_v41_c1_indices" in kernel:
        return "CSA2", "C1 RoPE/可见索引准备"
    if "deepseek_v41_swa_pack" in kernel:
        return "CSA2", "SWA 精确 FP8 编码/scale/缓存写入"
    if "deepseek_v41_fp4_cache_write" in kernel:
        return "CSA2", "主 KV/index 精确 FP4 编码/scale/缓存写入"
    if "deepseek_v41_selected_kv" in kernel:
        return "CSA2", "按候选槽解码实际读取的 packed KV，含写入完成依赖"
    if "mxfp4" in name or "mxfp4" in kernel:
        shape = inputs[1]["shape"] if kernel in ("GEMM", "BatchGemm") and len(inputs) > 1 else []
        stage = "W13 gate/up" if shape and 2304 in shape else "W2 down" if shape and 1152 in shape else "阶段见张量合同"
        return "路由专家", stage + ("矩阵计算" if shape else "压缩权重直接寻址/解码")
    if "bf16_identity" in kernel:
        return "路由专家", "MoE 结果/编译块完成依赖"
    if kernel in ("GEMM", "BatchGemm"):
        weight = inputs[1]["shape"] if len(inputs) > 1 else []
        if weight == [24, 20480]:
            return "mHC", "FP32 控制投影"
        if weight == [384, 5120]:
            return "Router", "384 专家路由打分"
        if weight == [25600, 6144]:
            return "Engram", "查询行到 key/value 投影"
        if "/attention/" in name:
            shapes = {(1280, 5120): "wq_a 输入投影", (16384, 1280): "wq_b Q 展开",
                      (512, 5120): "wkv 输入投影", (5120, 4096): "wo_b 输出投影",
                      (128, 512): "index K 投影"}
            if "/bmm" in name:
                return "Attention", "wo_a 分组输出 BMM"
            return "Attention", shapes.get(tuple(weight), "Compressor/其他投影，见节点合同")
        if "/moe/" in name:
            return "共享专家", "down 投影" if weight == [5120, 1152] else "gate/up 投影，见源节点"
        if weight and weight[0] == 64640:
            return "输出头", "TP 词表投影"
        return "矩阵计算待细分", "源节点及完整操作数已保留"
    if "sparse_attn" in kernel:
        return "Attention", "顺序 QK/online softmax/V 累加"
    if "sinkhorn" in kernel:
        return "mHC", "4×4 Sinkhorn"
    if kernel == "DmaMemcpy" and inputs and inputs[0]["shape"] == [512, 256, 1]:
        return "CSA2", "FP4 全缓存解包中的字节重排"
    if "/attention/" in name:
        return "Attention/CSA2", "张量准备/量化/状态更新，见源节点"
    if "/moe/" in name:
        return "MoE 准备", "激活量化/激活/路由归约，见源节点"
    if kernel.startswith("fused_kernel_"):
        return "融合表达式待细分", "保留实际 GUID 与输入输出；内部子算子无独立计时"
    return "入口/尾部/其他张量", node


def bundle_key(contract):
    if not contract:
        return None
    bundle = contract.get("attributes", {}).get("Bundle_idx")
    if bundle in (None, "N/A"):
        return None
    return contract["graph"]["path"], str(bundle)


def invocation_samples(rows, symbol, expected):
    rows = sorted(rows)
    if symbol["device_type"] == 1:
        packets = sum(symbol.get("working_engines", []))
    elif symbol["device_type"] == 0:
        packets = len({row[2] for row in rows})
    else:
        return [], None, "EDMA descriptor-to-invocation boundary not reconstructed"
    if not packets or len(rows) % packets:
        return [], None, "missing packet count or incomplete lane packet group"
    groups = [rows[i:i + packets] for i in range(0, len(rows), packets)]
    if expected is not None and len(groups) != expected:
        return [], None, "packet count differs from captured recipe frequency"
    if symbol["device_type"] == 0 and any(len({row[2] for row in group}) != packets for group in groups):
        return [], None, "MME lane ownership differs within a reconstructed invocation"
    # A stage's successive executions of one context are dependency ordered.
    if any(max(r[1] for r in a) > min(r[0] for r in b) for a, b in zip(groups, groups[1:])):
        return [], None, "ambiguous overlapping groups for the same recipe context"
    return [(max(r[1] for r in g) - min(r[0] for r in g)) / 1000 for g in groups], len(groups), None


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def analyze(root, rank, common):
    path = root / f"rank{rank}"
    inv = json.loads((path / "inventory.json").read_text())
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    mapped = symbols(inv, recipes)
    contracts = {(str(row["recipe_id"]), row["symbol"]["device_type"], row["symbol"]["full_context_id"]): row
                 for row in json.loads((path / "node-contracts.json").read_text())}
    bundle_categories = collections.defaultdict(set)
    for contract in contracts.values():
        symbol = contract["symbol"]
        category, _ = classify(symbol["node"], symbol["kernel"], contract["inputs"], contract["outputs"])
        key = bundle_key(contract)
        if key and category in ("路由专家", "Attention", "共享专家", "Engram", "mHC", "Router"):
            bundle_categories[key].add(category)
    own = json.loads((path / "device-windows.json").read_text())
    mhc_recipes = set()
    plan_entries, plan_sources = [], []
    plan_root = root.parent / f"plans/rank{rank}"
    for group in range(5):
        files = list(plan_root.glob(f"group{group}-*-prepare1.json"))
        if len(files) != 1:
            plan_entries = []
            break
        plan_sources.append(str(files[0]))
        plan_entries.extend(row for row in json.loads(files[0].read_text()) if row.get("kind") == "compute")
    provenance = []
    if any(entry.get("name", "").endswith("_mhc_submod_0") for entry in plan_entries):
        if len(plan_entries) != len(own["capture_order"]):
            raise RuntimeError("Prepared compute order differs from captured native segments")
        by_id = {str(recipe["recipe_id"]): recipe for recipe in recipes}
        for entry, enqueue in zip(plan_entries, own["capture_order"], strict=True):
            if not entry["name"].endswith("_mhc_submod_0"):
                continue
            # Bridge's program ID and Synapse's serialized recipe ID are
            # different namespaces. The exact captured segment order joins them.
            rid = enqueue[2].split(":")[0]
            if rid not in by_id or not any("control_gemv" in n["kernel"] for n in by_id[rid]["nodes"]):
                raise RuntimeError("mHC capture order differs from actual compiler symbols")
            mhc_recipes.add(rid)
            provenance.append(dict(program=entry, synapse_recipe=rid, enqueue=enqueue))
    (path / "mhc-partition-provenance.json").write_text(json.dumps(
        dict(plans=plan_sources, bindings=provenance), indent=2) + "\n")
    frequency = collections.Counter(row[2].split(":")[0] for row in own["capture_order"])
    windows, tokens = common["windows_us"], common["tokens"]
    ends = [w[1] for w in windows]
    period = sum(b - a for a, b in windows)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    raw, engines = collections.defaultdict(list), collections.defaultdict(list)
    first_index = {}
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            start, duration, lane, index = json.loads(line)
            win = bisect.bisect_right(ends, start)
            if win >= len(windows):
                continue
            lo, hi = windows[win]
            stop, begin = min(start + duration, hi), max(start, lo)
            if stop <= begin:
                continue
            node, symbol = inv["nodes"][index], mapped.get(index)
            key = ((node["recipe"].split(":")[0], symbol["device_type"], symbol["full_context_id"])
                   if symbol else (node["recipe"], -1, index))
            first_index[key] = index
            groups[key][win].append((begin, stop, lane))
            kernel = symbol["kernel"] if symbol else node["kernel"]
            raw[(node["engine"], kernel)].append((begin, stop))
            engines[node["engine"]].append((begin, stop))
    details, aggregated, recovered = [], collections.defaultdict(list), collections.defaultdict(list)
    for key, by_window in groups.items():
        index = first_index[key]
        node, symbol = inv["nodes"][index], mapped.get(index)
        contract = contracts.get(key)
        if contract is None:
            # Ordinary eager graphs have short-lived recipe handles and may not
            # be serialized. Join their exact enqueue identity to saved graphs.
            launches = [r for r in inv["host_enqueues"] if r[2] == node["recipe"]]
            options = []
            for launch in launches:
                graph = Path(launch[3] + "-eager_final_graph-symbol.pbtxt")
                if graph.exists():
                    options.extend((graph, n) for n in graph_nodes(graph)
                                   if n["name"] == node["node"] and n["op"] == node["kernel"])
            if len(options) == 1:
                graph, match = options[0]
                contract = {"inputs": io(match, "inputTensor:"), "outputs": io(match, "outputTensor:"),
                            "graph": {"path": str(graph), "sha256": hashlib.sha256(graph.read_bytes()).hexdigest()},
                            "attributes": match["attrs"]}
        kernel = symbol["kernel"] if symbol else node["kernel"]
        source = symbol["node"] if symbol else node["node"]
        inputs, outputs = (contract.get(part, []) if contract else [] for part in ("inputs", "outputs"))
        category, purpose = classify(source, kernel, inputs, outputs)
        origin = None
        if category == "融合表达式待细分" and bundle_categories.get(bundle_key(contract)) == {"路由专家"}:
            category = "路由专家"
            purpose = "同一编译 bundle 的专家激活/结果处理；内部表达式未独立计时"
            origin = {"rule": "unambiguous expert bundle", "bundle": bundle_key(contract)}
        if key[0] in mhc_recipes:
            category, purpose = "mHC", "已由原生计划和编译分区核验的独立控制/统计/混合分支"
        spans = [row[:2] for rows in by_window.values() for row in rows]
        samples, count, missing = [], 0, []
        for rows in by_window.values():
            if symbol:
                values, n, reason = invocation_samples(rows, symbol, frequency.get(key[0]))
            else:
                values, n, reason = [], None, "unserialized eager node has no working-engine contract"
            samples.extend(values)
            if n is not None:
                count += n
            if reason:
                missing.append(reason)
        duration = union(spans)
        row = {"rank": rank, "recipe_id": key[0], "context_id": key[2], "engine": node["engine"],
               "category": category, "purpose": purpose, "kernel": kernel, "source_node": source,
               "mean_invocation_ms": statistics.mean(samples) if samples else None,
               "complete_invocation_samples": len(samples), "observed_calls": count if not missing else None,
               "calls_per_token": count / len(tokens) if not missing else None,
               "observed_lane_packets": sum(map(len, by_window.values())), "count_limitations": sorted(set(missing)),
               "activity_ms_per_token": duration / len(tokens) / 1000, "period_pct": duration / period * 100,
               "inputs": inputs, "outputs": outputs, "compiler_contract": contract}
        row["classification_provenance"] = origin
        details.append(row)
        dtype_shape = json.dumps([[(v["dtype"], v["shape"]) for v in part] for part in (inputs, outputs)])
        group_key = (category, purpose, node["engine"], kernel, dtype_shape)
        aggregated[group_key].append((row, spans, samples))
        recovered[(node["engine"], kernel)].extend(spans)
    for key in raw:
        assert abs(union(raw[key]) - union(recovered[key])) < 1e-6, key
    summary = []
    for key, items in aggregated.items():
        spans = [s for _, group, _ in items for s in group]
        samples = [v for _, _, group in items for v in group]
        unknown = any(row["observed_calls"] is None for row, _, _ in items)
        count = None if unknown else sum(row["observed_calls"] for row, _, _ in items)
        duration = union(spans)
        summary.append({"rank": rank, "category": key[0], "purpose": key[1], "engine": key[2], "kernel": key[3],
                        "dtype_shapes": json.loads(key[4]),
                        "mean_invocation_ms": statistics.mean(samples) if samples else None,
                        "complete_invocation_samples": len(samples), "observed_calls": count,
                        "calls_per_token": count / len(tokens) if count is not None else None,
                        "activity_ms_per_token": duration / len(tokens) / 1000, "period_pct": duration / period * 100,
                        "node_keys": [[r["recipe_id"], r["context_id"]] for r, _, _ in items]})
    scale = len(tokens) * 1000
    tpc, mme = union(engines["TPC"]), union(engines["MME"])
    compute = union(engines["TPC"] + engines["MME"])
    all_device = union([span for spans in engines.values() for span in spans])
    partition = {"TPC_only_ms": (compute - mme) / scale, "MME_only_ms": (compute - tpc) / scale,
                 "TPC_MME_overlap_ms": (tpc + mme - compute) / scale,
                 "other_recorded_device_only_ms": (all_device - compute) / scale,
                 "unattributed_or_other_stage_ms": (period - all_device) / scale}
    assert abs(sum(partition.values()) - period / scale) < 1e-8
    details.sort(key=lambda r: -r["activity_ms_per_token"])
    summary.sort(key=lambda r: -r["activity_ms_per_token"])
    write_csv(path / "kernel-breakdown.csv", summary)
    (path / "node-breakdown.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    result = {"rank": rank, "tokens": tokens, "period_ms": period / scale, "partition": partition,
              "trace_sha256": inv["trace_sha256"], "measured_nodes": len(details),
              "unmatched_nodes": sum(r["compiler_contract"] is None for r in details),
              "nodes_with_unknown_calls": sum(r["observed_calls"] is None for r in details),
              "status": "all activity retained; unknown invocation boundaries and fused origins remain explicit",
              "kernel_rows": summary}
    (path / "kernel-breakdown.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "kernel_rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    # The same global trace clock and PP1-final-MoE boundary is used for all
    # ranks. Keep token IDs present in every rank's complete stage sequence.
    data = [json.loads((args.analysis / f"rank{r}/device-windows.json").read_text()) for r in range(4)]
    tokens = sorted(set.intersection(*(set(d["tokens"]) for d in data)))
    anchor = dict(zip(data[2]["tokens"], data[2]["windows_us"]))
    common = {"tokens": tokens, "windows_us": [anchor[t] for t in tokens], "anchor": "rank2 final MoE to final MoE"}
    (args.analysis / "common-windows.json").write_text(json.dumps(common, indent=2) + "\n")
    for rank in range(4):
        analyze(args.analysis, rank, common)
