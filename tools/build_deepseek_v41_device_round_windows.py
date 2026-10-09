# SPDX-License-Identifier: Apache-2.0
"""Align archived round completions with explicit raw-clock enqueue brackets.

The recorded perf-counter stage submission lies after the Target scope ends
and before sampled_draws begins. Their intersection bounds the clock offset;
it is not a fabricated simultaneous clock sample or a collective arrival.
"""
import argparse
import json
from pathlib import Path
import re
import statistics


def build(run, request_id):
    analysis = run / "analysis"
    records = json.loads((run / "round-timing/rank0.json").read_text())["records"]
    records = [row for row in records if row["request_id"] == request_id]
    request = json.loads((run / "official-16k-eos-trace/result.json").read_text())
    position = request["usage"]["prompt_tokens"] - 1
    by_position = {}
    for index, row in enumerate(records):
        by_position[position] = index
        position += row["committed"]
    inventory = json.loads((analysis / "rank0/inventory.json").read_text())
    base = inventory["base_time_nanoseconds"]
    markers = inventory["cpu_markers"]
    anchors = []
    pattern = re.compile(r"v41::verify_and_commit::PP0::decode::P(\d+)::C6::emit1")
    for start, length, name in markers:
        match = pattern.fullmatch(name)
        if not match:
            continue
        index = by_position[int(match[1])]
        following = records[index + 1]
        target = [row for row in markers if row[2] == "v41::device_round::target"
                  and start <= row[0] < start + length]
        draws = [row for row in markers if row[2] in ("v41::device_round::sampled_draws",
                                                    "v41::device_round::sampled_native_protocol")
                 and start <= row[0] < start + length]
        # A repaired request can enqueue both discarded and corrected
        # lookahead Targets in the same host scope. Use unambiguous scopes
        # for clock calibration; retain repaired round windows afterwards.
        if len(target) != 1 or len(draws) != 1:
            continue
        lower_raw = base + (target[0][0] + target[0][1]) * 1000
        upper_raw = base + draws[0][0] * 1000
        stamp = following["stage_submitted_ns"]
        anchors.append(dict(generation=following["generation"], submission_perf_ns=stamp,
                            submission_raw_bracket_ns=[lower_raw, upper_raw],
                            offset_bracket_ns=[stamp - upper_raw, stamp - lower_raw]))
    assert len(anchors) >= 3
    lower = max(row["offset_bracket_ns"][0] for row in anchors)
    upper = min(row["offset_bracket_ns"][1] for row in anchors)
    if lower > upper:
        raise ValueError("No constant clock-offset intersection; need an explicit drift model")
    offset, uncertainty = (lower + upper) / 2, (upper - lower) / 2
    clock = dict(method="intersection of same-source submission timestamps bracketed by explicit raw scopes",
                 perf_minus_raw_offset_ns=offset, uncertainty_ns=uncertainty,
                 offset_interval_ns=[lower, upper], anchors=anchors,
                 assumption="constant offset over the captured interval; all observed anchors satisfy it",
                 collective_arrival_measurement=False)
    summaries = []
    for rank in range(4):
        path = analysis / f"rank{rank}"
        inv = json.loads((path / "inventory.json").read_text())
        provenance = json.loads((path / "raw-provenance.json").read_text())
        parser_low, parser_high = provenance["parser_window_ns"]
        rank_base = inv["base_time_nanoseconds"]
        before = json.loads((run / f"traces/rank{rank}-native-profile-start.json").read_text())
        after = json.loads((run / f"traces/rank{rank}-native-profile-stop.json").read_text())
        steps = after["v41"]["decode_steps"] - before["v41"]["decode_steps"]
        protocol_replays = sum(name == "v41::device_round::sampled_native_protocol"
                               for _, _, name in inv["cpu_markers"])
        entries = after["native_entry_replays"] - before["native_entry_replays"]
        joint = after["native_joint_replays"] - before["native_joint_replays"]
        # Target and the sampled verification/draft protocol now use the same
        # executor. Its global counters include both, not just Target C6.
        environment = json.loads((run / "process.json").read_text())["environment"]
        full_repairs = (sum(name == "v41::device_round::sampled_exact_repair"
                            for _, _, name in inv["cpu_markers"])
                        if environment.get("VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR") == "1" else 0)
        assert entries == joint == steps + protocol_replays + full_repairs
        assert after["prepares"] == before["prepares"]
        assert after["native_captures"] == before["native_captures"]
        coverage = dict(kind="native Target C6 stage and sampled protocol", decode_steps=steps,
                        protocol_replays=protocol_replays, full_repair_replays=full_repairs, entry_replays=entries,
                        hot_prepares=0, hot_captures=0)
        captured_positions = {int(match[1]) for _, _, name in inv["cpu_markers"]
                              if (match := pattern.fullmatch(name))}
        units, windows, committed, proof = [], [], [], []
        for position in sorted(captured_positions):
            index = by_position[position]
            row = records[index]
            if index == 0:
                continue
            previous = records[index - 1]
            low, high = previous["end_ns"] - offset, row["end_ns"] - offset
            if low - uncertainty < parser_low or high + uncertainty > parser_high:
                continue
            assert row["ring_released"] and previous["ring_released"]
            units.append(position)
            windows.append([(low - rank_base) / 1000, (high - rank_base) / 1000])
            committed.append(row["committed"])
            proof.append(dict(position=position, generation=row["generation"], request_id=request_id,
                              rank0_post_ring_release_perf_ns=[previous["end_ns"], row["end_ns"]],
                              native_coverage=coverage))
        assert len(units) > 1
        doc = dict(topology=dict(tensor_parallel_size=4, pipeline_parallel_size=1), phase="decode", unit="round",
                   tokens=units, windows_us=windows, committed_tokens=committed,
                   total_committed_tokens=sum(committed), mean_committed=statistics.mean(committed),
                   capture_order=[], coverage_proof=proof, base_time_nanoseconds=rank_base,
                   asynchronous_completion=True, clock_proof=clock,
                   boundary="successive actual rank0 post-ring-release completions of the profiled request",
                   clock_boundary_uncertainty_ns=uncertainty)
        (path / "device-windows.json").write_text(json.dumps(doc, indent=2) + "\n")
        summaries.append(dict(rank=rank, units=units, rounds=len(units),
                              period_ms=statistics.mean((b - a) / 1000 for a, b in windows)))
    (analysis / "LEDGER_WINDOW_PROOF.json").write_text(
        json.dumps(dict(clock=clock, ranks=summaries, formal_qualified=False), indent=2) + "\n")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    build(args.run, args.request_id)
