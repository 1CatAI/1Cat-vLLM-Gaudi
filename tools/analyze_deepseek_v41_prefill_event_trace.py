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


def summarize(path):
    raw = json.loads(path.read_text())
    spans = raw["spans"]

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
    raw["python_attention_span_count"] = sum(item["name"] == "attention" for item in spans)
    raw["nominal_attention_span_count"] = len(chunks) * 20
    raw["python_attention_complete"] = (raw["python_attention_span_count"] == raw["nominal_attention_span_count"])
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("prefill-events-pp*-tp*-g*.json"))
    if len(paths) != 4:
        raise ValueError(f"Expected four rank traces, found {len(paths)} in {args.directory}")
    records = [summarize(path) for path in paths]
    if {(r["pp_rank"], r["tp_rank"]) for r in records} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Trace does not cover TP2xPP2 exactly")
    if len({r["tokens"] for r in records}) != 1 or len({r["request_id"] for r in records}) != 1:
        raise ValueError("Ranks did not record the same request and prompt length")
    summary = {
        "schema": 1,
        "scope": "four-rank diagnostic Prefill span timeline, not a hardware kernel trace",
        "records": [{
            k: v
            for k, v in record.items() if k != "spans"
        } for record in records]
    }
    (args.directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Four-rank Prefill event trace", "",
        "Current-stream HPU event intervals use each rank's own anchor. Rows are mutually exclusive "
        "within that rank; do not add or align device offsets across ranks. This diagnostic does "
        "not provide hardware-kernel calls, FLOP counters, or HBM byte counters. mHC is included "
        "in Layer other; it has no separate event in this serving path. Chunk ingress on PP1 "
        "includes PP receive/dependency time but is not synonymous with communication duration.", "",
        "| Rank | Window ms | MLA | Index | Attention other | MoE | Layer other | Chunk ingress | "
        "Inter-layer gap | Chunk egress | Outside |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ]
    for record in records:
        groups = record["groups_ms"]
        lines.append("| PP{pp_rank}/TP{tp_rank} | {device_ms:.3f} | {mla:.3f} | {selection:.3f} | "
                     "{attention:.3f} | {moe:.3f} | {other:.3f} | "
                     "{ingress:.3f} | {between:.3f} | {egress:.3f} | {outside:.3f} |".format(
                         **record,
                         mla=groups["Attention sparse MLA"],
                         selection=groups["Attention index selection"],
                         attention=groups["Attention other"],
                         moe=groups["Routed expert MoE"],
                         other=groups["Layer other"],
                         ingress=groups["Chunk ingress before first layer"],
                         between=groups["Gaps between layers"],
                         egress=groups["Chunk egress after last layer"],
                         outside=groups["Request work outside chunks"]))
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
            "| Phase | PP0/TP0 ms | PP0/TP1 ms | PP1/TP0 ms | PP1/TP1 ms |", "| --- | ---: | ---: | ---: | ---: |"
        ])
        for name in records[0]["attention_components_ms"]:
            values = [record["attention_components_ms"][name] for record in records]
            lines.append("| {} | {} |".format(name, " | ".join(f"{value:.3f}" for value in values)))
    lines.extend([
        "", "Per-rank, per-block raw spans and host times are in the four JSON files. "
        "Hardware kernels must be attributed using a valid Synapse trace; the previous "
        "four-rank HwTrace attempt crashed before publishing data.", ""
    ])
    (args.directory / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
