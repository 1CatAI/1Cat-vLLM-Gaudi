# SPDX-License-Identifier: Apache-2.0
"""Named rank activity within the existing raw-clock submission periods."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from report_deepseek_v41_raw_entry_periods import merge, length


def contract_index(rows):
    result = {}
    for row in rows:
        symbol = row.get('symbol', {})
        rid = str(row.get('raw_recipe_id', str(row.get('recipe_id', '')).split('@')[0]))
        if symbol.get('unique_node_id') is not None:
            result[(rid, 'uid', str(symbol['unique_node_id']))] = row
        if symbol.get('full_context_id') is not None and symbol.get('device_type') is not None:
            result[(rid, str(symbol['device_type']), str(symbol['full_context_id']))] = row
    return result


def node_contract(node, contracts):
    rid = str(node.get('recipe', '')).split('@')[0]
    uid = node.get('raw_unique_node_id')
    if uid not in (None, ''):
        return contracts.get((rid, 'uid', str(uid)), {})
    device = {'TPC': 1, 'MME': 0, 'DMA': 8}.get(node.get('engine'))
    return contracts.get((rid, str(device), str(node.get('raw_context_id'))), {})


def classify(node, contract):
    kernel = node["kernel"].lower()
    shapes = [t["shape"] for t in contract.get("inputs", [])]
    if "journal_copy" in kernel or "journal_batch" in kernel:
        return "rollback_state_journal"
    if any(t in kernel for t in ("silu_quant", "scale_reduce")):
        return "expert_activation_reduce"
    if "expert" in kernel and any(t in kernel for t in ("sat", "n256", "token_wide", "split_scale")):
        return "routed_experts"
    if "router" in kernel:
        return "router"
    if any(t in kernel for t in ("logical_mla", "logical_scale_cache", "logical_main_mirror",
                                "selected_mla", "sparse_attn", "main_batch_publish_gather",
                                "main_batch_reuse_gather", "swa_only_reuse_gather")):
        return "MLA_gather_softmax"
    if "index_keys" in kernel or "index_score" in kernel:
        return "index_KV_scoring"
    if "rope" in kernel:
        return "RoPE"
    if any(t in kernel for t in ("control_", "mhc_", "sinkhorn", "post_collapse")):
        return "mHC"
    if "gemm" in kernel:
        source = (contract.get("symbol", {}).get("node", "") + " " + node.get("node", "")).lower()
        # The compiler can squeeze a one-row batch to rank2. Its parent
        # operator still identifies QK/PV; a rank3-only shape rule misses it.
        if any(name in source for name in ("main_split_publish_mla", "main_split_reuse_mla",
                                           "main_qk_flat_", "main_stream_exp_")):
            return "MLA_QK_PV"
        # Raw recipe debug names remain available when automatic SDK graph
        # export is disabled. These compound owners uniquely identify their
        # MME role without inventing missing tensor shapes or placement.
        if "custom_deepseek_v41_control_mme" in source:
            return "mHC"
        if "custom_deepseek_v41_expert_n256_moe" in source:
            return "routed_experts"
        if "custom_deepseek_v41_paged_mla_mme_gaudi2" in source:
            return "MLA_QK_PV"
        if "custom_deepseek_v41_prefill_index_scores_gaudi2" in source:
            return "index_KV_scoring"
        if "/moe/fp8_gemm" in source:
            return "shared_experts"
        if "/moe/" in source and "custom_deepseek_v41_bf16_linear_f32_gaudi2" in source:
            return "router_shared_joint_MME"
        if "/moe/linear_" in source and "linear_fwd_f32" in source:
            return "router"
        if "/attention/bmm" in source:
            # The ordinary grouped output projection is an einsum lowered
            # to batch_gemm; source owner distinguishes it from QK/PV.
            return "attention_output_group_projection"
        if "/attention/linear" in source:
            return "attention_dense_projection"
        if any(32320 in s or 129280 in s for s in shapes):
            return "vocabulary_head"
        if any(20480 in s for s in shapes):
            return "mHC"
        if any(384 in s and 5120 in s for s in shapes):
            return "router"
        if any(t.get("dtype") == "hfloat8" and len(t["shape"]) == 3 for t in contract.get("inputs", [])):
            return "routed_experts"
        if any(len(s) == 3 and 16 in s and 512 in s for s in shapes):
            return "MLA_QK_PV"
        return "dense_or_unclassified_MME"
    if any(t in kernel for t in ("bitonic", "topk", "cumsum", "merge_sort", "argmax", "arg_max",
                                "weighted_mass", "nucleus", "bounded_local_distribution",
                                "probability_parts", "probability_draw", "probability_full_select")):
        return "selection_sampling"
    if "engram" in kernel:
        return "Engram"
    if node.get("metadata_provenance") is None and not contract:
        return "unresolved_" + node["engine"]
    return "auxiliary"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    directory = args.analysis / f"rank{args.rank}"
    inventory = json.loads((directory / "inventory.json").read_text())
    ledger = json.loads((args.analysis / "RAW_ENTRY_PERIOD_LEDGER.json").read_text())
    low, high = ledger["periods"][0]["start_raw_us"], ledger["periods"][-1]["end_raw_us"]
    base = inventory["base_time_nanoseconds"] / 1000
    contracts = contract_index(json.loads((directory / "node-contracts.json").read_text()))
    categories, kernels, anchors = defaultdict(list), defaultdict(list), defaultdict(list)
    node_info = []
    for node in inventory["nodes"]:
        contract = node_contract(node, contracts)
        category = classify(node, contract)
        head = category == "vocabulary_head" and any(
            t["shape"][:1] in ([6], [12]) and len(t["shape"]) == 2 for t in contract.get("outputs", [])
        )
        node_info.append((category, head))
    with gzip.open(directory / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            event = json.loads(line)
            a, b = base + event[0], base + event[0] + event[1]
            if b <= low or a >= high or b <= a:
                continue
            node = inventory["nodes"][event[3]]
            if node["engine"] not in ("TPC", "MME"):
                continue
            span = (max(a, low), min(b, high))
            category, head = node_info[event[3]]
            categories[category].append(span)
            kernels[(node["engine"], node["kernel"], category)].append(span)
            if head:
                anchors["C6_vocabulary_head"].append(span)
    normalized = {name: merge(spans) for name, spans in categories.items()}
    edges = defaultdict(lambda: defaultdict(int))
    for category, spans in normalized.items():
        for a, b in spans:
            edges[a][category] += 1
            edges[b][category] -= 1
    active, exclusive, prior = defaultdict(int), defaultdict(float), low
    mixed = 0
    for stamp, changes in sorted(edges.items()):
        names = [name for name, count in active.items() if count]
        if len(names) == 1:
            exclusive[names[0]] += stamp - prior
        elif names:
            mixed += stamp - prior
        for name, delta in changes.items():
            active[name] += delta
        prior = stamp
    rounds = ledger["rounds"]
    result = dict(
        rank=args.rank,
        rounds=rounds,
        window="same raw-clock host entry periods; not completion windows",
        critical_path=False,
        mean_period_ms=ledger["mean_period_ms"],
        mean_four_cards_no_activity_ms=ledger["mean_four_cards_no_recorded_activity_ms"],
        rank_compute_ms=length([s for rows in normalized.values() for s in rows]) / rounds,
        mixed_categories_ms=mixed / 1000 / rounds,
        categories=[
            dict(name=name, activity_union_ms=length(spans) / rounds, exclusive_ms=exclusive[name] / 1000 / rounds)
            for name, spans in sorted(normalized.items(), key=lambda item: -length(item[1]))
        ],
        kernels=[
            dict(
                engine=e,
                kernel=k,
                category=c,
                activity_union_ms=length(spans) / rounds,
                activity_segments=len(merge(spans)),
            )
            for (e, k, c), spans in sorted(kernels.items(), key=lambda item: -length(item[1]))[:40]
        ],
        C6_head_activity_intervals_raw_us=merge(anchors["C6_vocabulary_head"]),
        exact_metadata_proof=str(directory / "cached-native-metadata-proof.json"),
    )
    path = args.analysis / f"NAMED_PERIODS_RANK{args.rank}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("kernels", "C6_head_activity_intervals_raw_us")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
