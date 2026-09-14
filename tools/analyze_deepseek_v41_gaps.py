# SPDX-License-Identifier: Apache-2.0
"""Locate the last/next recorded work around global idle intervals.

Endpoints describe observed boundaries, not causal ownership or removable time.
Only exact same-recipe tensor names prove a local data dependency. Hardware
completion counters and cross-recipe communication dependencies need separate
native-plan evidence; this report does not invent them from NIC point events.
"""
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

from analyze_deepseek_v41_trace import symbols


def boundaries(root, rank, gaps):
    if any(a >= b for a, b in gaps) or any(a[1] > b[0] for a, b in zip(gaps, gaps[1:])):
        raise ValueError("Require ordered, nonoverlapping, positive gap intervals")
    path = root / f"rank{rank}"
    inventory = json.loads((path / "inventory.json").read_text())
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    mapping = symbols(inventory, recipes)
    rows = json.loads((path / "node-breakdown.json").read_text())
    keyed = {(r["recipe_id"], r["engine"], r["context_id"]): r for r in rows}
    contracts = {}
    for index, node in enumerate(inventory["nodes"]):
        symbol = mapping.get(index)
        rid = node["recipe"].split(":")[0] if symbol else node["recipe"]
        context = symbol["full_context_id"] if symbol else index
        key = (rid, node["engine"], context)
        if key in keyed:
            contracts[index] = keyed[key]
    starts, ends = [a for a, _ in gaps], [b for _, b in gaps]
    previous, following = [None] * len(gaps), [None] * len(gaps)
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            start, duration, _, index = json.loads(line)[:4]
            node = inventory["nodes"][index]
            if node["engine"] not in ("TPC", "MME", "DMA") or duration <= 0:
                continue
            end = start + duration
            after = bisect.bisect_left(starts, end)
            if after < len(gaps) and (previous[after] is None or previous[after][1] < end):
                previous[after] = (start, end, index)
            before = bisect.bisect_right(ends, start) - 1
            if before >= 0 and (following[before] is None or following[before][0] > start):
                following[before] = (start, end, index)
    for i in range(1, len(gaps)):
        if previous[i] is None:
            previous[i] = previous[i - 1]
    for i in range(len(gaps) - 2, -1, -1):
        if following[i] is None:
            following[i] = following[i + 1]

    def describe(event):
        if event is None:
            return None
        start, end, index = event
        node, row = inventory["nodes"][index], contracts.get(index, {})
        contract = row.get("compiler_contract") or {}
        return {
            "rank": rank,
            "stage": rank // 2,
            "start_us": start,
            "end_us": end,
            "recipe_id": row.get("recipe_id", node["recipe"]),
            "context_id": row.get("context_id"),
            "engine": node["engine"],
            "kernel": row.get("kernel"),
            "purpose": row.get("purpose"),
            "source_node": row.get("source_node"),
            "category": row.get("category"),
            "execution_index": contract.get("attributes", {}).get("Exec_idx"),
            "inputs": [x["name"] for x in row.get("inputs", [])],
            "outputs": [x["name"] for x in row.get("outputs", [])]
        }

    return [(describe(a), describe(b)) for a, b in zip(previous, following, strict=True)]


def summarize(root):
    ledger = json.loads((root / "four-rank-disjoint-accounting.json").read_text())
    gaps = ledger["unattributed_intervals_us"]
    count = len(ledger["tokens"])
    ranks = [boundaries(root, rank, gaps) for rank in range(4)]
    result, groups = [], collections.defaultdict(lambda: {"count": 0, "duration_us": 0., "examples": []})
    categories = collections.defaultdict(float)
    for i, (start, end) in enumerate(gaps):
        pairs = [rows[i] for rows in ranks]
        prior = [a for a, _ in pairs if a is not None]
        later = [b for _, b in pairs if b is not None]
        previous = max(prior, key=lambda x: x["end_us"]) if prior else None
        following = min(later, key=lambda x: x["start_us"]) if later else None
        proofs = []
        for a, b in pairs:
            if a and b and a["recipe_id"] == b["recipe_id"]:
                shared = sorted(set(a["outputs"]) & set(b["inputs"]))
                if shared:
                    proofs.append({
                        "rank": a["rank"],
                        "recipe_id": a["recipe_id"],
                        "tensor_names": shared,
                        "qualification": "Exact graph edge; invocation identity and criticality unresolved"
                    })
        key = (previous["stage"] if previous else None, previous["purpose"] if previous else None,
               following["stage"] if following else None, following["purpose"] if following else None)
        group = groups[key]
        group["count"] += 1
        group["duration_us"] += end - start
        category_key = tuple((x["stage"], x["engine"], x["category"]) if x else None for x in (previous, following))
        categories[category_key] += end - start
        if len(group["examples"]) < 3:
            group["examples"].append(i)
        result.append({
            "start_us": start,
            "end_us": end,
            "duration_ms": (end - start) / 1000,
            "last_global_work": previous,
            "next_global_work": following,
            "rank_boundaries": pairs,
            "exact_tensor_edges": proofs
        })
    rows = [{
        "previous_stage": k[0],
        "previous_purpose": k[1],
        "next_stage": k[2],
        "next_purpose": k[3],
        "interval_count": v["count"],
        "ms_per_token": v["duration_us"] / count / 1000,
        "examples": v["examples"]
    } for k, v in groups.items()]
    rows.sort(key=lambda x: -x["ms_per_token"])
    total = sum(x["ms_per_token"] for x in rows)
    assert abs(total - ledger["groups"][-1]["exclusive_ms"]) < 1e-7
    output = {
        "tokens":
        ledger["tokens"],
        "unknown_ms_per_token":
        total,
        "boundary_groups":
        rows,
        "category_boundary_groups": [{
            "previous": key[0],
            "next": key[1],
            "ms_per_token": value / count / 1000
        } for key, value in sorted(categories.items(), key=lambda item: -item[1])],
        "intervals":
        result,
        "causal_attribution_complete":
        False,
        "limitations": [
            "Last/next engine work is not necessarily the cause of a gap.",
            "Equal recipe IDs can recur; no generation or completion event is fabricated.",
            "A boundary group partitions observed unknown time, not communication duration.",
            "DMA descriptor endpoints are not whole-copy invocation boundaries."
        ]
    }
    (root / "gap-boundaries.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"unknown_ms_per_token": total, "top_boundaries": rows[:12]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    summarize(parser.parse_args().analysis)
