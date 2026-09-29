# SPDX-License-Identifier: Apache-2.0
"""Reconcile bounded four-rank Prefill span traces without double counting."""
import argparse
import json
from pathlib import Path


def merged(spans):
    intervals = sorted((max(0.0, float(a)), float(b)) for a, b in spans if b > a)
    result = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def difference(left, right):
    """Subtract union(right) from union(left), retaining disjoint intervals."""
    right = merged(right)
    result = []
    for start, end in merged(left):
        cursor = start
        for low, high in right:
            if high <= cursor:
                continue
            if low >= end:
                break
            if low > cursor:
                result.append((cursor, min(low, end)))
            cursor = min(end, max(cursor, high))
        if cursor < end:
            result.append((cursor, end))
    return result


def intersection(left, right):
    """Return the intersection of two interval sets on one rank clock."""
    return [(max(a, c), min(b, d)) for a, b in merged(left) for c, d in merged(right) if a < d and c < b]


def summarize(path, *, stage_layers=None, allow_decoder_halo=False):
    raw = json.loads(path.read_text())
    spans = raw["spans"]
    if stage_layers is None:
        stage_layers = list(range(raw["pp_rank"] * 20, (raw["pp_rank"] + 1) * 20))
    stage_layers = list(stage_layers)

    def intervals(name):
        return [(item["device_start_ms"], item["device_end_ms"]) for item in spans if item["name"] == name]

    chunks = intervals("transaction_chunk")
    layers = intervals("layer")
    attention = intervals("attention")
    moe = intervals("moe")
    selection = intervals("index_selection")
    mla = intervals("attention_mla")
    mhc = intervals("mhc_pre")
    all_window = [(0.0, raw["device_ms"])]
    # Give nested spans one owner. Keep the unmodified activity unions below
    # for inspection; mutually exclusive ownership is a wall-time ledger,
    # not a claim that other engines were idle during the owned interval.
    attention_in_layers = intersection(attention, layers)
    moe_in_layers = difference(intersection(moe, layers), attention_in_layers)
    mla_in_attention = intersection(mla, attention_in_layers)
    selection_in_attention = difference(intersection(selection, attention_in_layers), mla_in_attention)
    # These diagnostic phases are submitted in order on the same rank stream.
    # Own each interval once, keeping an explicit remainder for code paths or
    # asynchronous work that has not been marked. The outer 9-group ledger is
    # unchanged so it stays comparable to previous four-rank traces.
    detail_names = (
        "attention_input_projection",
        "attention_query_norm_projection",
        "attention_kv_norm_rope",
        "attention_compress_kv",
        "attention_index_query",
        "attention_swa_workspace",
        "attention_main_workspace",
        "attention_output_inverse_rope",
        "attention_output_projection",
        "attention_output_reduce",
    )
    attention_other = difference(attention_in_layers, mla_in_attention + selection_in_attention)
    attention_components = {
        "attention_mla": duration(mla_in_attention),
        "index_selection": duration(selection_in_attention)
    }
    remaining_detail = attention_other
    for name in detail_names:
        claimed = intersection(intervals(name), remaining_detail)
        attention_components[name] = duration(claimed)
        remaining_detail = difference(remaining_detail, claimed)
    attention_components["attention_unattributed"] = duration(remaining_detail)
    if abs(sum(attention_components.values()) - duration(attention_in_layers)) > 0.2:
        raise ValueError(f"{path}: Attention subspans do not reconcile")
    mhc_in_layers = difference(intersection(mhc, layers), attention_in_layers + moe_in_layers)
    other_remaining = difference(layers, attention_in_layers + moe_in_layers)
    other_components = {}
    for name in ("mhc_input", "mhc_post"):
        claimed = intersection(intervals(name), other_remaining)
        other_components[name] = duration(claimed)
        other_remaining = difference(other_remaining, claimed)
    other_components["layer_other_unattributed"] = duration(other_remaining)
    ingress, between, egress = [], [], []
    for chunk in merged(chunks):
        owned_layers = intersection([chunk], layers)
        if not owned_layers:
            between.append(chunk)
            continue
        first_layer = min(start for start, _ in owned_layers)
        last_layer = max(end for _, end in owned_layers)
        before = (chunk[0], first_layer)
        after = (last_layer, chunk[1])
        if before[1] > before[0]:
            ingress.append(before)
        if after[1] > after[0]:
            egress.append(after)
        between.extend(difference([chunk], owned_layers + [before, after]))
    grouped = {
        "Attention sparse MLA": duration(mla_in_attention),
        "Attention index selection": duration(selection_in_attention),
        "Attention other": duration(attention_other),
        "Routed expert MoE": duration(moe_in_layers),
        "Layer other": duration(difference(layers, attention_in_layers + moe_in_layers)),
        "mHC captured inside layer other": duration(mhc_in_layers),
        "Chunk ingress before first layer": duration(ingress),
        "Gaps between layers": duration(between),
        "Chunk egress after last layer": duration(egress),
        "Request work outside chunks": duration(difference(all_window, chunks)),
    }
    total = sum(value for name, value in grouped.items() if name != "mHC captured inside layer other")
    if abs(total - raw["device_ms"]) > 0.2:
        raise ValueError(f"{path}: span partition conflicts with device window: total={total}, "
                         f"window={raw['device_ms']}")
    raw["activity_unions_ms"] = {
        name: duration(items)
        for name, items in (("layer", layers), ("attention", attention), ("attention_mla", mla),
                            ("index_selection", selection), ("moe", moe), ("mhc_pre", mhc))
    }
    per_block = []
    for number, (start, end) in enumerate(chunks):
        clip = lambda items, start=start, end=end: [(max(a, start), min(b, end)) for a, b in items
                                                    if a < end and b > start]
        per_block.append(
            dict(block=number,
                 start_ms=start,
                 end_ms=end,
                 duration_ms=end - start,
                 attention_ms=duration(clip(attention)),
                 moe_ms=duration(clip(moe)),
                 layer_ms=duration(clip(layers))))
    raw["groups_ms"] = grouped
    raw["attention_components_ms"] = attention_components
    raw["layer_other_components_ms"] = other_components
    by_layer = {}
    for layer in sorted(
        {item["layer"]
         for item in spans if item["name"] == "attention" and isinstance(item.get("layer"), int)}):
        owner = [(item["device_start_ms"], item["device_end_ms"]) for item in spans
                 if item["name"] == "attention" and item.get("layer") == layer]
        phases = {}
        remaining = owner
        for name in ("attention_mla", "index_selection", *detail_names):
            same_layer = [(item["device_start_ms"], item["device_end_ms"]) for item in spans
                          if item["name"] == name and item.get("layer") == layer]
            claimed = intersection(same_layer, remaining)
            phases[name] = duration(claimed)
            remaining = difference(remaining, claimed)
        phases["attention_unattributed"] = duration(remaining)
        phases["total"] = duration(owner)
        by_layer[str(layer)] = phases
    raw["attention_per_layer_ms"] = by_layer
    raw["per_block"] = per_block
    coverage = []
    for chunk in (item for item in spans if item["name"] == "transaction_chunk"):
        owned = [
            item for item in spans
            if chunk["host_start_ns"] <= item["host_start_ns"] and item["host_end_ns"] <= chunk["host_end_ns"]
        ]
        actual = [item.get("layer") for item in owned if item["name"] == "layer"]
        expected = stage_layers
        prefix = [layer for layer in stage_layers if layer <= 20]
        prefix_only = allow_decoder_halo and 20 in stage_layers and actual == prefix
        if prefix_only:
            expected = prefix
        attention_ids = [item.get("layer") for item in owned if item["name"] == "attention"]
        coverage.append(
            dict(layer_ids=actual,
                 expected_layer_ids=expected,
                 policy="prefix_only" if prefix_only else "full",
                 attention_ids=attention_ids,
                 complete=actual == expected and attention_ids == expected))
    raw["layer_coverage"] = coverage
    raw["python_attention_span_count"] = sum(item["name"] == "attention" for item in spans)
    raw["nominal_attention_span_count"] = sum(len(item["expected_layer_ids"]) for item in coverage)
    raw["python_attention_complete"] = bool(coverage) and all(item["complete"] for item in coverage)
    raw["attention_subphases_captured"] = any(item["name"] in (*detail_names, "attention_mla", "index_selection")
                                              for item in spans)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--request-id", help="Select one request from a retained service's diagnostic directory")
    parser.add_argument("--expected-prompt-tokens",
                        type=int,
                        help="Require complete prompt capture, including a full final decoder transaction")
    parser.add_argument("--allow-decoder-halo",
                        action="store_true",
                        help="Admit complete source-through-layer20 prefix transactions in the coverage audit")
    parser.add_argument("--output", type=Path, help="Keep each selected request's analysis in its own directory")
    args = parser.parse_args()
    paths = sorted(args.directory.glob("prefill-events-pp*-tp*-g*.json"))
    headers = {path: json.loads(path.read_text()) for path in paths}
    if args.request_id:
        headers = {path: raw for path, raw in headers.items() if raw["request_id"] == args.request_id}
    if not headers or len({raw["request_id"] for raw in headers.values()}) != 1:
        raise ValueError("Select exactly one captured request with --request-id")
    ranks = {(raw["pp_rank"], raw["tp_rank"]) for raw in headers.values()}
    tp4 = ranks == {(0, rank) for rank in range(4)}
    if not tp4 and ranks != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Trace must cover all four ranks of TP4xPP1 or TP2xPP2")
    records = [
        summarize(path, stage_layers=range(40) if tp4 else None, allow_decoder_halo=args.allow_decoder_halo)
        for path in headers
    ]
    records.sort(key=lambda raw: (raw["pp_rank"], raw["tp_rank"], raw["started_ns"]))
    rank_records = {rank: [raw for raw in records if (raw["pp_rank"], raw["tp_rank"]) == rank] for rank in ranks}
    if len({tuple(raw["tokens"] for raw in group) for group in rank_records.values()}) != 1:
        raise ValueError("Ranks did not record the same scheduler transaction lengths")
    if args.expected_prompt_tokens:
        for group in rank_records.values():
            if sum(raw["tokens"] for raw in group) != args.expected_prompt_tokens:
                raise ValueError("Capture does not cover the complete requested prompt")
            if group[-1]["layer_coverage"][-1]["policy"] == "prefix_only":
                raise ValueError("Complete prompt capture cannot end at a source-only transaction")
    output = args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema":
        1,
        "scope":
        "four-rank diagnostic Prefill span timeline, not a hardware kernel trace",
        "topology":
        dict(tensor_parallel_size=4 if tp4 else 2, pipeline_parallel_size=1 if tp4 else 2),
        "rank_totals": [
            dict(pp_rank=rank[0],
                 tp_rank=rank[1],
                 transactions=len(group),
                 prompt_tokens=sum(raw["tokens"] for raw in group),
                 device_windows_ms=sum(raw["device_ms"] for raw in group),
                 host_between_transactions_ms=sum((right["started_ns"] - left["finished_ns"]) / 1e6
                                                  for left, right in zip(group, group[1:])),
                 groups_ms={name: sum(raw["groups_ms"][name] for raw in group)
                            for name in group[0]["groups_ms"]}) for rank, group in sorted(rank_records.items())
        ],
        "records": [{
            k: v
            for k, v in record.items() if k != "spans"
        } for record in records]
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    detailed_attention = all(record["attention_subphases_captured"] for record in records)
    attention_header = "MLA | Index | Attention other" if detailed_attention else "Attention envelope"
    attention_rule = "---: | ---: | ---:" if detailed_attention else "---:"
    ingress_note = ("For TP4xPP1 there is no pipeline-stage receive; chunk ingress contains input preparation "
                    "and dependencies whose finer attribution is not recorded."
                    if tp4 else "On pipeline stage 1, chunk ingress may include PP receive/dependency time; "
                    "the envelope is not a communication measurement.")
    lines = [
        "# Four-rank Prefill event trace", "",
        "Current-stream HPU event intervals use each rank's own anchor. Rows are mutually exclusive "
        "within that rank; do not add or align device offsets across ranks. This diagnostic does "
        "not provide hardware-kernel calls, FLOP counters, or HBM byte counters. mHC is included "
        "in Layer other; selected mHC subspans are shown separately below when present. " + ingress_note, "",
        f"| Rank | Window ms | {attention_header} | MoE | Layer other | Chunk ingress | "
        "Inter-layer gap | Chunk egress | Outside |",
        f"| --- | ---: | {attention_rule} | ---: | ---: | ---: | ---: | ---: | ---: |"
    ]
    for record in records:
        groups = record["groups_ms"]
        attention_names = ("Attention sparse MLA", "Attention index selection", "Attention other")
        attention_values = (" | ".join(f"{groups[name]:.3f}" for name in attention_names)
                            if detailed_attention else f"{sum(groups[name] for name in attention_names):.3f}")
        lines.append("| PP{pp_rank}/TP{tp_rank}/G{generation} | {device_ms:.3f} | {attention_values} | "
                     "{moe:.3f} | {other:.3f} | "
                     "{ingress:.3f} | {between:.3f} | {egress:.3f} | {outside:.3f} |".format(
                         **record,
                         attention_values=attention_values,
                         moe=groups["Routed expert MoE"],
                         other=groups["Layer other"],
                         ingress=groups["Chunk ingress before first layer"],
                         between=groups["Gaps between layers"],
                         egress=groups["Chunk egress after last layer"],
                         outside=groups["Request work outside chunks"]))
    if not detailed_attention:
        lines.extend([
            "", "The selected coarse capture records the whole Attention envelope. Its internal "
            "MLA/index subdivisions were not requested and are not separately reported."
        ])
    if not all(record["python_attention_complete"] for record in records):
        lines.extend([
            "", "**Python span coverage is incomplete on at least one stage.** Compiled/replayed layer "
            "work may appear under `Gaps between layers`; that label must not be interpreted as idle "
            "time. The Attention phase total is complete only for ranks with every layer/chunk observed.", "",
            "| Rank | Observed Attention spans | Nominal layer/chunk spans | Complete |", "| --- | ---: | ---: | --- |"
        ])
        for record in records:
            lines.append(f"| PP{record['pp_rank']}/TP{record['tp_rank']} | "
                         f"{record['python_attention_span_count']} | "
                         f"{record['nominal_attention_span_count']} | "
                         f"{record['python_attention_complete']} |")
    if all("attention_input_projection" in record["activity_unions_ms"] or any(
            item.get("name") == "attention_input_projection" for item in record["spans"]) for record in records):
        lines.extend([
            "", "## Attention internal phases (same diagnostic request)", "",
            "These rows partition each rank's outer Attention span. The remaining time is kept "
            "explicit; no component activity is added to the outer ledger.", "",
            "| Phase | " + " | ".join(f"PP{r['pp_rank']}/TP{r['tp_rank']}/G{r['generation']} ms"
                                      for r in records) + " |", "| --- | " + " | ".join("---:" for _ in records) + " |"
        ])
        for name in records[0]["attention_components_ms"]:
            values = [record["attention_components_ms"][name] for record in records]
            lines.append("| {} | {} |".format(name, " | ".join(f"{value:.3f}" for value in values)))
    if any(item.get("name") in ("mhc_input", "mhc_post") for record in records for item in record["spans"]):
        lines.extend([
            "", "## Layer-other internal phases (same diagnostic request)", "",
            "These rows partition Layer other; they are not added again to the outer ledger.", "",
            "| Phase | " + " | ".join(f"PP{r['pp_rank']}/TP{r['tp_rank']}/G{r['generation']} ms"
                                      for r in records) + " |", "| --- | " + " | ".join("---:" for _ in records) + " |"
        ])
        for name in records[0]["layer_other_components_ms"]:
            values = [record["layer_other_components_ms"][name] for record in records]
            lines.append("| {} | {} |".format(name, " | ".join(f"{value:.3f}" for value in values)))
    lines.extend([
        "", "Per-rank, per-transaction raw spans and host times remain in the source JSON files. "
        "Hardware kernels require a separately validated Synapse capture. This ledger includes stream "
        "dependencies and must not be interpreted as engine occupancy.", ""
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
