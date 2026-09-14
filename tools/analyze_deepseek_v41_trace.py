# SPDX-License-Identifier: Apache-2.0
"""Reconstruct complete V4.1 stage periods before per-node attribution."""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import re


def union(spans):
    total, left, right = 0.0, None, None
    for start, end in sorted(spans):
        if right is not None and start <= right:
            right = max(right, end)
        else:
            if right is not None:
                total += right - left
            left, right = start, end
    return total + (right - left if right is not None else 0)


def symbols(inventory, recipes):
    context = {(str(r["recipe_id"]), n["device_type"], n["full_context_id"]): n
               for r in recipes for n in r["nodes"]}
    names = {(str(r["recipe_id"]), n["device_type"], n["node"], n["kernel"]): n
             for r in recipes for n in r["nodes"]}
    result = {}
    for index, node in enumerate(inventory["nodes"]):
        rid = node["recipe"].split(":")[0]
        device = {"TPC": 1, "MME": 0, "DMA": 8}.get(node["engine"])
        symbol = names.get((rid, device, node["node"], node["kernel"]))
        if symbol is None and node["kernel"].startswith(("TPC_SPU_", "MMEH_", "DMA_")):
            match = re.search(r" (\d+)$", node["kernel"])
            identity = int(match[1]) if match else 0
            symbol = context.get((rid, device, 0 if identity == int(rid) else identity))
        if symbol is not None:
            result[index] = symbol
    return result


def analyze(root, rank):
    path = root / f"rank{rank}"
    inv = json.loads((path / "inventory.json").read_text())
    if inv.get("complete_json") is False:
        raise ValueError("A truncated trace prefix cannot establish complete stage periods")
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    byid = {str(recipe["recipe_id"]): recipe for recipe in recipes}
    stats = json.loads((root.parent / f"traces/rank{rank}-native-profile-stop.json").read_text())
    mapped = symbols(inv, recipes)
    markers = [row for row in inv["cpu_markers"] if row[2] == "vllm_gaudi::native_decoder_enqueue"]
    assert len(markers) == stats["native_replays"]
    first = min(row[0] for row in markers)
    capture = [row for row in inv["host_enqueues"] if row[0] < first
               and ("/graph_" in row[3] or row[3].endswith(".recipe"))][-stats["native_segments"]:]
    assert len(capture) == stats["native_segments"]
    boundaries = {}
    for rid in {row[2].split(":")[0] for row in capture}:
        candidates = [node for node in byid[rid]["nodes"]
                      if node["kernel"] == "custom_deepseek_v41_bf16_identity_gaudi2"]
        if candidates:
            final = max(candidates, key=lambda node: node["full_context_id"])
            assert "_bundle_" not in final["node"]
            boundaries[rid] = final
    pattern = [row[2].split(":")[0] for row in capture if row[2].split(":")[0] in boundaries]
    assert len(pattern) == 20, pattern
    boundary_rows = collections.defaultdict(list)
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row[0] < first or row[3] not in mapped:
                continue
            node = inv["nodes"][row[3]]
            rid = node["recipe"].split(":")[0]
            if (node["engine"] == "TPC" and rid in boundaries
                    and mapped[row[3]]["full_context_id"] == boundaries[rid]["full_context_id"]):
                boundary_rows[rid].append(row)
    calls, proofs = [], {}
    for rid, rows in boundary_rows.items():
        rows.sort()
        gaps = sorted({b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[0] > a[0]})
        low, high = max(zip(gaps, gaps[1:]), key=lambda pair: pair[1] / pair[0])
        assert high / low > 100, (rid, low, high)
        threshold = (low * high)**0.5
        groups = []
        for row in rows:
            if not groups or row[0] - groups[-1][-1][0] > threshold:
                groups.append([])
            groups[-1].append(row)
        workers = sum(boundaries[rid]["working_engines"])
        for group in groups:
            calls.append({"rid": rid, "start": min(row[0] for row in group),
                          "end": max(row[0] + row[1] for row in group),
                          "complete": len(group) == len({row[2] for row in group}) == workers})
        proofs[rid] = {"boundary": boundaries[rid], "observed_gap_split_us": [low, high],
                       "threshold_us": threshold, "packets_per_call": workers}
    calls.sort(key=lambda row: row["start"])
    expected = pattern * len(markers)
    assert [row["rid"] for row in calls] == expected[-len(calls):], "Device MoE sequence differs from stage plan"
    by_token = collections.defaultdict(list)
    for offset, call in enumerate(calls, len(expected) - len(calls)):
        call.update(token=offset // 20, layer=offset % 20)
        by_token[call["token"]].append(call)
    ends = {token: next(call["end"] for call in rows if call["layer"] == 19)
            for token, rows in by_token.items() if any(call["layer"] == 19 for call in rows)}
    tokens = [token for token, rows in sorted(by_token.items()) if token >= 10 and token - 1 in ends
              and len(rows) == 20 and all(row["complete"] for row in rows)]
    assert tokens
    windows = [(ends[token - 1], ends[token]) for token in tokens]
    (path / "device-windows.json").write_text(json.dumps({"tokens": tokens, "windows_us": windows,
        "calls": calls, "boundary_proofs": proofs, "capture_order": capture}, indent=2) + "\n")
    period = sum(b - a for a, b in windows)
    grouped, engines = collections.defaultdict(list), collections.defaultdict(list)
    counts = collections.Counter()
    window_ends = [end for _, end in windows]
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            start, duration, lane, index = json.loads(line)
            selected = bisect.bisect_right(window_ends, start)
            if selected >= len(windows):
                continue
            node = inv["nodes"][index]
            symbol = mapped.get(index)
            kernel = symbol["kernel"] if symbol else node["kernel"]
            counted = False
            while selected < len(windows) and windows[selected][0] < start + duration:
                low, high = windows[selected]
                begin, end = max(start, low), min(start + duration, high)
                if begin < end:
                    grouped[(node["engine"], kernel)].append((begin, end))
                    engines[node["engine"]].append((begin, end))
                    counted = True
                selected += 1
            counts[(node["engine"], kernel)] += int(counted)
    scale = len(windows) * 1000
    tpc, mme = union(engines["TPC"]), union(engines["MME"])
    compute = union(engines["TPC"] + engines["MME"])
    rows = sorted([{"engine": engine, "kernel": kernel, "activity_ms_per_token": union(spans) / scale,
                    "period_pct": union(spans) / period * 100, "observed_lane_packets": counts[(engine, kernel)]}
                   for (engine, kernel), spans in grouped.items()], key=lambda row: -row["activity_ms_per_token"])
    result = {"rank": rank, "tokens": tokens, "period_ms": period / scale,
              "tpc_ms": tpc / scale, "mme_ms": mme / scale, "compute_ms": compute / scale,
              "overlap_ms": (tpc + mme - compute) / scale, "unattributed_or_other_ms": (period - compute) / scale,
              "rows": rows, "status": "Activity screening; full physical-call/tensor attribution pending"}
    (path / "activity-screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)
    print(json.dumps(rows[:12]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    analyze(args.analysis, args.rank)
