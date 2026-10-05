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
    # IDs are finite-width runtime values. Never let a colliding archived
    # recipe silently overwrite another node contract.
    contexts, named = collections.defaultdict(list), collections.defaultdict(list)
    for recipe in recipes:
        for node in recipe["nodes"]:
            prefix = (str(recipe["recipe_id"]), node["device_type"])
            contexts[(*prefix, node["full_context_id"])].append(node)
            named[(*prefix, node["node"], node["kernel"].lower().replace("_", ""))].append(node)

    def unambiguous(candidates):
        return {key: values[0] for key, values in candidates.items() if all(value == values[0] for value in values)}

    context, names = unambiguous(contexts), unambiguous(named)
    result = {}
    for index, node in enumerate(inventory["nodes"]):
        rid = node["recipe"].split(":")[0]
        device = {"TPC": 1, "MME": 0, "DMA": 8}.get(node["engine"])
        symbol = names.get((rid, device, node["node"], node["kernel"].lower().replace("_", "")))
        if symbol is None and rid.isdigit() and node["kernel"].startswith(("TPC_SPU_", "MMEH_", "DMA_")):
            match = re.search(r" (\d+)$", node["kernel"])
            identity = int(match[1]) if match else 0
            symbol = context.get((rid, device, 0 if identity == int(rid) else identity))
        if symbol is None and not node.get("raw_unique_node_id") and node.get("raw_context_id", "").isdigit():
            raw_rid = rid.partition("@")[0]
            identity = int(node["raw_context_id"])
            if raw_rid.isdigit():
                # Single-node recipes may encode the recipe ID as context0.
                symbol = context.get((rid, device, 0 if identity == int(raw_rid) else identity))
        if symbol is not None:
            result[index] = symbol
    return result


def logical_replay_markers(inventory, expected):
    legacy = sorted(row for row in inventory["cpu_markers"] if row[2] == "vllm_gaudi::native_decoder_enqueue")
    if len(legacy) == expected:
        return legacy, {"mode": "single", "single": len(legacy)}

    prefixes = sorted(row for row in inventory["cpu_markers"] if row[2] == "vllm_gaudi::native_decoder_prefix_enqueue")
    finishes = sorted(row for row in inventory["cpu_markers"] if row[2] == "vllm_gaudi::native_decoder_finish_enqueue")
    if len(legacy) + len(prefixes) != expected or len(prefixes) != len(finishes):
        raise AssertionError(
            {
                "expected": expected,
                "single": len(legacy),
                "prefix": len(prefixes),
                "finish": len(finishes),
            }
        )
    for index, (prefix, finish) in enumerate(zip(prefixes, finishes)):
        if prefix[0] > finish[0] or (index + 1 < len(prefixes) and finish[0] > prefixes[index + 1][0]):
            raise AssertionError({"unpaired_segmented_replay": index, "prefix": prefix, "finish": finish})
    return sorted(legacy + prefixes), {
        "mode": "segmented",
        "single": len(legacy),
        "prefix": len(prefixes),
        "finish": len(finishes),
    }


def tp4_windows(inventory, phase, request_start_ns=None, native_coverage=None, position_window=None):
    """Use real sampled-token consumers, including host submission/wait time.

    Async workers publish their real completion-consumption marker separately
    from the early sampling return. Their cadence may contain the current
    suffix and next token's prefix; it is a throughput cycle, not a per-token
    isolated-forward latency or a pure CPU execution span.
    """
    commits = []
    asynchronous = phase == "decode" and any(
        row[2].startswith("v41::worker_commit::PP0::") for row in inventory["cpu_markers"]
    )
    commit_name = "worker_commit" if asynchronous else "verify_and_commit"
    for start, duration, name in inventory["cpu_markers"]:
        match = re.fullmatch(rf"v41::{commit_name}::PP0::(prefill|decode)::P(\d+)::C(\d+)::emit([01])", name)
        if match:
            kind, position, count, emit = match.groups()
            commits.append(
                dict(
                    start=start,
                    end=start + duration,
                    phase=kind,
                    position=int(position),
                    count=int(count),
                    emit=bool(int(emit)),
                )
            )
    commits.sort(key=lambda item: item["start"])
    if phase == "prefill":
        targets = sorted(row for row in inventory["cpu_markers"] if row[2].startswith("v41::target::PP0::prefill::"))
        completed = [item for item in commits if item["phase"] == "prefill" and item["emit"]]
        if len(completed) != 1 or not targets:
            raise ValueError("Prefill capture must contain one complete request through its first-token consumer")
        prefill_commits = [item for item in commits if item["phase"] == "prefill"]
        cursor = 0
        for commit in prefill_commits:
            if commit["position"] != cursor:
                raise ValueError("Prefill commits omit or repeat a logical prompt interval")
            cursor += commit["count"]
        target_counts = [int(re.search(r"::C(\d+)$", row[2])[1]) for row in targets]
        if target_counts != [item["count"] for item in prefill_commits]:
            raise ValueError("Prefill targets and completed prompt intervals differ")
        low, high = targets[0][0], completed[0]["end"]
        target_start = low
        if request_start_ns is not None:
            base = inventory.get("base_time_nanoseconds")
            if base is None:
                raise ValueError("Request/trace alignment requires the Kineto clock base")
            low = (request_start_ns - base) / 1000
            if not inventory["all_activity_start_us"] <= low <= target_start:
                raise ValueError("Request timestamp lies outside the capture or after its first model target")
        layer_rows = [
            row
            for row in inventory["cpu_markers"]
            if low <= row[0] < high and re.fullmatch(r"v41::prefill::layer::layer\d+::C\d+", row[2])
        ]
        counts = collections.Counter(int(re.search(r"::layer(\d+)::", row[2])[1]) for row in layer_rows)
        if set(counts) != set(range(40)) or set(counts.values()) != {len(targets)}:
            raise ValueError(f"Incomplete forty-layer prefill coverage: {dict(counts)}")
        units, windows, proof = (
            [0],
            [(low, high)],
            dict(
                chunks=targets,
                prompt_tokens=cursor,
                layer_counts=dict(counts),
                first_target_start_us=target_start,
                request_start_ns=request_start_ns,
                before_first_target_ms=(target_start - low) / 1000,
            ),
        )
    else:
        units, windows, proof = [], [], []
        for previous, current in zip(commits, commits[1:]):
            if current["phase"] != "decode" or not current["emit"] or current["count"] != 1:
                continue
            if current["position"] != previous["position"] + previous["count"]:
                continue
            if position_window is not None and not position_window[0] <= current["position"] <= position_window[1]:
                continue
            low, high = previous["end"], current["end"]
            if (position_window is not None and
                    (low < inventory.get("first_us", -float("inf")) or
                     high > inventory.get("last_us", float("inf")))):
                continue
            groups = []
            for begin, duration, name in inventory["cpu_markers"]:
                match = re.fullmatch(r"v41::compiled::layers(\d+)-(\d+)::C1", name)
                if match and low <= begin and begin + duration <= high:
                    groups.append(tuple(map(int, match.groups())))
            if native_coverage is not None:
                # A whole native stage has no per-group Python scopes. Its
                # completed target and worker commit are the serving boundary;
                # capture start/stop counters prove the complete replay path.
                target = [
                    row
                    for row in inventory["cpu_markers"]
                    if row[2]
                    in ("v41::target::PP0::decode::C1", f"v41::target::PP0::decode::C1::P{current['position']}")
                    and low <= row[0]
                    and row[0] + row[1] <= high
                ]
                if len(target) != 1:
                    # In the device feedback path, the previous commit queued
                    # this stage. Require counter coverage plus an actual
                    # completion consumed inside the current worker commit.
                    consumed = any(name == f"v41::completion_consume::P{current['position']}"
                                   and current["start"] <= begin and begin + duration <= high
                                   for begin, duration, name in inventory["cpu_markers"])
                    if not (native_coverage.get("device_loop_complete") and consumed and not target):
                        continue
            elif sorted(groups) != [(start, start + 3) for start in range(0, 40, 4)]:
                continue
            units.append(current["position"])
            windows.append((low, high))
            proof.append(
                dict(position=current["position"], groups=groups, consumer=current, native_coverage=native_coverage)
            )
        if not windows:
            raise ValueError("No complete TP4 forty-layer decode cycle and sampled-token consumer")
        if units != list(range(units[0], units[-1] + 1)):
            raise ValueError("Missing an interior TP4 decode cycle; trace coverage is incomplete")
        if position_window is not None and units != list(range(position_window[0], position_window[1] + 1)):
            raise ValueError("Requested position subset lacks complete hardware/consumer windows")
    return dict(
        topology={"tensor_parallel_size": 4, "pipeline_parallel_size": 1},
        phase=phase,
        unit="request" if phase == "prefill" else "token",
        tokens=units,
        windows_us=windows,
        capture_order=[],
        coverage_proof=proof,
        base_time_nanoseconds=inventory.get("base_time_nanoseconds"),
        asynchronous_completion=asynchronous,
        boundary=(
            "recorded client request dispatch to first-token consumer; includes initial state setup"
            if request_start_ns is not None
            else "first prefill submission to first-token consumer"
        )
        if phase == "prefill"
        else (
            "successive async worker token commits; includes overlapping next-token prefix, scheduler and device waits"
            if asynchronous
            else "successive sampled-token commit completions; includes scheduler, CPU and device waits"
        ),
    )


def analyze(root, rank, phase=None, request_result=None):
    path = root / f"rank{rank}"
    inv = json.loads((path / "inventory.json").read_text())
    if inv.get("complete_json") is False:
        raise ValueError("A truncated trace prefix cannot establish complete stage periods")
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    byid = {str(recipe["recipe_id"]): recipe for recipe in recipes}
    collection = root / "collection.json"
    if collection.exists():
        record = next(row for row in json.loads(collection.read_text())["ranks"] if row["rank"] == rank)
        stats = record["stats"]
    else:
        stats = json.loads((root.parent / f"traces/rank{rank}-native-profile-stop.json").read_text())
    if stats.get("topology", {}).get("tensor_parallel_size", 2) == 4:
        if phase not in ("prefill", "decode"):
            raise ValueError("TP4 trace analysis requires --phase prefill or decode")
        request_start_ns = None
        if request_result is not None:
            request = json.loads(request_result.read_text())
            if (request["status"] not in (("passed", "diagnostic_passed") if phase == "decode" else ("passed",))
                    or request["profile"] != phase):
                raise ValueError("Trace request does not have a matching successful phase")
            if phase == "prefill":
                request_start_ns = request["request_start_ns"]
        native_coverage = None
        if phase == "decode" and stats.get("native_entry_replays", 0) and collection.exists():
            capture = Path(json.loads(collection.read_text())["capture_dir"])
            before = json.loads((capture / f"rank{rank}-native-profile-start.json").read_text())
            steps = stats["v41"]["decode_steps"] - before["v41"]["decode_steps"]
            entries = stats["native_entry_replays"] - before["native_entry_replays"]
            joint = stats["native_joint_replays"] - before["native_joint_replays"]
            if steps != entries or steps != joint or stats["tp4_direct_group_replays"] != 0:
                raise ValueError("Native serving coverage differs from completed decode steps")
            if stats["prepares"] != before["prepares"] or stats["native_captures"] != before["native_captures"]:
                raise ValueError("Native trace contains compilation or graph capture")
            native_coverage = dict(
                kind="complete native stage replay",
                decode_steps=steps,
                native_entry_replays=entries,
                native_joint_replays=joint,
                direct_group_replays=0,
                hot_prepares=0,
                hot_captures=0,
            )
            queued = stats["v41"].get("device_loop_queued", 0) - before["v41"].get("device_loop_queued", 0)
            if queued:
                consumed = stats["v41"].get("device_loop_consumed", 0) - before["v41"].get("device_loop_consumed", 0)
                targets = sum(bool(re.fullmatch(r"v41::target::PP0::decode::C1(?:::P\d+)?", row[2]))
                              for row in inv["cpu_markers"])
                repairs = sum(stats["v41"].get(key, 0) - before["v41"].get(key, 0)
                              for key in ("device_loop_recomputes", "device_loop_discards", "device_sampling_fallbacks"))
                if queued != consumed or queued + targets != steps or repairs:
                    raise ValueError("Device-feedback trace needs exact queue/consume coverage without repairs")
                native_coverage.update(device_loop_complete=True, queued=queued, consumed=consumed,
                                       explicit_targets=targets, repairs=repairs)
        position_window = (json.loads(collection.read_text()).get("decode_position_window")
                           if collection.exists() and phase == "decode" else None)
        result = tp4_windows(inv, phase, request_start_ns, native_coverage, position_window)
        if request_result is not None and phase == "prefill":
            if result["coverage_proof"]["prompt_tokens"] != request["usage"]["prompt_tokens"]:
                raise ValueError("Prefill trace does not cover the complete measured prompt")
        (path / "device-windows.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "coverage_proof"}), flush=True)
        return
    mapped = symbols(inv, recipes)
    markers, marker_proof = logical_replay_markers(inv, stats["native_replays"])
    first = min(row[0] for row in markers)
    capture = [
        row for row in inv["host_enqueues"] if row[0] < first and ("/graph_" in row[3] or row[3].endswith(".recipe"))
    ][-stats["native_segments"] :]
    assert len(capture) == stats["native_segments"]
    boundaries = {}
    for rid in {row[2].split(":")[0] for row in capture}:
        candidates = [
            node for node in byid[rid]["nodes"] if node["kernel"] == "custom_deepseek_v41_bf16_identity_gaudi2"
        ]
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
            if (
                node["engine"] == "TPC"
                and rid in boundaries
                and mapped[row[3]]["full_context_id"] == boundaries[rid]["full_context_id"]
            ):
                boundary_rows[rid].append(row)
    calls, proofs = [], {}
    for rid, rows in boundary_rows.items():
        rows.sort()
        gaps = sorted({b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[0] > a[0]})
        low, high = max(zip(gaps, gaps[1:]), key=lambda pair: pair[1] / pair[0])
        assert high / low > 100, (rid, low, high)
        threshold = (low * high) ** 0.5
        groups = []
        for row in rows:
            if not groups or row[0] - groups[-1][-1][0] > threshold:
                groups.append([])
            groups[-1].append(row)
        workers = sum(boundaries[rid]["working_engines"])
        for group in groups:
            calls.append(
                {
                    "rid": rid,
                    "start": min(row[0] for row in group),
                    "end": max(row[0] + row[1] for row in group),
                    "complete": len(group) == len({row[2] for row in group}) == workers,
                }
            )
        proofs[rid] = {
            "boundary": boundaries[rid],
            "observed_gap_split_us": [low, high],
            "threshold_us": threshold,
            "packets_per_call": workers,
        }
    calls.sort(key=lambda row: row["start"])
    expected = pattern * len(markers)
    assert [row["rid"] for row in calls] == expected[-len(calls) :], "Device MoE sequence differs from stage plan"
    by_token = collections.defaultdict(list)
    for offset, call in enumerate(calls, len(expected) - len(calls)):
        call.update(token=offset // 20, layer=offset % 20)
        by_token[call["token"]].append(call)
    ends = {
        token: next(call["end"] for call in rows if call["layer"] == 19)
        for token, rows in by_token.items()
        if any(call["layer"] == 19 for call in rows)
    }
    tokens = [
        token
        for token, rows in sorted(by_token.items())
        if token >= 10 and token - 1 in ends and len(rows) == 20 and all(row["complete"] for row in rows)
    ]
    assert tokens
    windows = [(ends[token - 1], ends[token]) for token in tokens]
    (path / "device-windows.json").write_text(
        json.dumps(
            {
                "tokens": tokens,
                "windows_us": windows,
                "calls": calls,
                "boundary_proofs": proofs,
                "capture_order": capture,
                "replay_marker_proof": marker_proof,
            },
            indent=2,
        )
        + "\n"
    )
    period = sum(b - a for a, b in windows)
    grouped, engines = collections.defaultdict(list), collections.defaultdict(list)
    counts = collections.Counter()
    window_ends = [end for _, end in windows]
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            start, duration, lane, index = json.loads(line)[:4]
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
    rows = sorted(
        [
            {
                "engine": engine,
                "kernel": kernel,
                "activity_ms_per_token": union(spans) / scale,
                "period_pct": union(spans) / period * 100,
                "observed_lane_packets": counts[(engine, kernel)],
            }
            for (engine, kernel), spans in grouped.items()
        ],
        key=lambda row: -row["activity_ms_per_token"],
    )
    result = {
        "rank": rank,
        "tokens": tokens,
        "period_ms": period / scale,
        "tpc_ms": tpc / scale,
        "mme_ms": mme / scale,
        "compute_ms": compute / scale,
        "overlap_ms": (tpc + mme - compute) / scale,
        "unattributed_or_other_ms": (period - compute) / scale,
        "rows": rows,
        "status": "Activity screening; full physical-call/tensor attribution pending",
    }
    (path / "activity-screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)
    print(json.dumps(rows[:12]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument(
        "--request-result",
        type=Path,
        help="Same profiled request's result.json, including its wall-clock dispatch timestamp",
    )
    args = parser.parse_args()
    analyze(args.analysis, args.rank, args.phase, args.request_result)
