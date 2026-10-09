# SPDX-License-Identifier: Apache-2.0
"""Coarse Target C6 activity from exact cached recipe IDs, without recapture.

Use the existing raw-clock round periods. This measures compute activity, not
formal verification latency: communication and queue dependencies are outside
the compute union. Do not substitute it for the unprofiled round timer.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from report_deepseek_v41_raw_entry_periods import merge, length
from report_deepseek_v41_c6_named_periods import classify


def report(analysis, rank):
    root = analysis / f"rank{rank}"
    inventory = json.loads((root / "inventory.json").read_text())
    proof = json.loads((root / "cached-native-metadata-proof.json").read_text())
    contracts = json.loads((root / "node-contracts.json").read_text())
    ledger = json.loads((analysis / "RAW_ENTRY_PERIOD_LEDGER.json").read_text())
    recipes = defaultdict(list)
    by_node = {}
    for contract in contracts:
        rid = str(contract.get("raw_recipe_id"))
        recipes[rid].append(contract)
        by_node[rid, str(contract.get("symbol", {}).get("unique_node_id"))] = contract
    selected, rejected = {}, {}
    for evidence in proof["evidence"]:
        rid = str(evidence["recipe_id"])
        shapes = [t["shape"] for node in recipes[rid]
                  for t in node.get("inputs", []) + node.get("outputs", [])]
        c6 = any(shape and shape[0] == 6 for shape in shapes)
        full_sampler = any(129280 in shape for shape in shapes)
        destination = selected if c6 and not full_sampler else rejected
        destination[rid] = dict(C6_shape_observed=c6, full_vocabulary_sampler=full_sampler,
                                cache_identity=evidence["identity"])
    if not selected:
        raise ValueError("No exact cached Target C6 recipe identities")
    low = ledger["periods"][0]["start_raw_us"]
    high = ledger["periods"][-1]["end_raw_us"]
    base = inventory["base_time_nanoseconds"] / 1000
    categories, target, other = defaultdict(list), [], []
    lookup = []
    for node in inventory["nodes"]:
        rid = node["recipe"].split("@")[0]
        contract = by_node.get((rid, str(node.get("raw_unique_node_id"))), {})
        lookup.append((rid in selected, classify(node, contract)))
    with gzip.open(root / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            a, b = base + event[0], base + event[0] + event[1]
            if b <= low or a >= high or b <= a:
                continue
            node = inventory["nodes"][event[3]]
            if node["engine"] not in ("TPC", "MME"):
                continue
            span = max(a, low), min(b, high)
            is_target, category = lookup[event[3]]
            if is_target:
                target.append(span)
                categories[category].append(span)
            else:
                other.append(span)
    rounds = ledger["rounds"]
    union = merge(target + other)
    result = dict(
        rank=rank, rounds=rounds, source="Exact cached binary/graph recipe IDs and raw hardware clock",
        measurement="TPC/MME activity unions inside asynchronous round submission periods",
        formal_verification_latency=False, critical_path=False,
        target_compute_union_ms=length(target) / rounds,
        other_compute_union_ms=length(other) / rounds,
        target_other_overlap_ms=(length(target) + length(other) - length(union)) / rounds,
        target_categories=[dict(name=name, activity_union_ms=length(spans) / rounds)
                           for name, spans in sorted(categories.items(), key=lambda x: -length(x[1]))],
        target_recipes=selected, excluded_cached_recipes=rejected,
        exact_recipe_proof=str(root / "cached-native-metadata-proof.json"),
        attribution_precision_ms=.5,
    )
    (analysis / f"TARGET_NATIVE_PERIODS_RANK{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("target_recipes", "excluded_cached_recipes", "target_categories")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    report(args.analysis, args.rank)


if __name__ == "__main__":
    main()
