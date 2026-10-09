# SPDX-License-Identifier: Apache-2.0
"""Offline physical W2 occupancy and current sampler cost from existing traces."""
import argparse
import bisect
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def duration_union(rows):
    merged = []
    for begin, end in sorted(rows):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return sum(end-begin for begin, end in merged)


def analyze(analysis):
    rank = analysis / 'rank0'
    inventory = json.loads((rank / 'inventory.json').read_text())
    periods = json.loads((analysis / 'RAW_DEVICE_CYCLE_LEDGER.json').read_text())['periods']
    base = inventory['base_time_nanoseconds']/1000
    starts = [p['start_raw_us']-base for p in periods]
    ends = [p['end_raw_us']-base for p in periods]
    nodes = inventory['nodes']
    w2_ids = {i for i, n in enumerate(nodes) if n.get('kernel') == 'custom_deepseek_v41_expert_w2_split_scale_gaudi2'}
    sampling_ids = {i for i, n in enumerate(nodes) if any(x in n.get('kernel', '') for x in
                    ('weighted_mass', 'probability_draw', 'vocab_softmax', 'nucleus'))}
    selected = w2_ids | sampling_ids
    w2, sampling = defaultdict(list), defaultdict(list)
    with gzip.open(rank / 'hardware.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row[3] not in selected:
                continue
            cycle = bisect.bisect_right(starts, row[0])-1
            if cycle < 0 or row[0]+row[1] > ends[cycle]:
                continue
            if row[3] in w2_ids:
                w2[cycle, row[3]].append(row)
            else:
                sampling[nodes[row[3]]['kernel']].append((row[0], row[0]+row[1]))
    rows = []
    for (cycle, node), events in sorted(w2.items()):
        events.sort()
        if len(events) % 24:
            raise ValueError('Incomplete W2 physical launch; no occupancy inference')
        for offset in range(0, len(events), 24):
            group = events[offset:offset+24]
            if len({x[2] for x in group}) != 24:
                raise ValueError('W2 physical launches overlap or have ambiguous lane counts')
            first = min(x[0] for x in group)
            last = max(x[0]+x[1] for x in group)
            busy = sum(x[1] for x in group)/24
            front = sum(x[0]-first for x in group)/24
            tail = sum(last-x[0]-x[1] for x in group)/24
            if abs(last-first-busy-front-tail) > 1e-5:
                raise ValueError('Physical capacity allocation does not close')
            durations = sorted(x[1] for x in group)
            rows.append(dict(cycle=cycle, node=node, recipe=nodes[node]['recipe'],
                source_node=nodes[node]['node'], ordinal=offset//24,
                start_raw_us=first+base, end_raw_us=last+base,
                span_us=last-first, mean_core_busy_us=busy,
                front_capacity_lost_us=front, tail_capacity_lost_us=tail,
                start_skew_us=max(x[0] for x in group)-first,
                end_skew_us=last-min(x[0]+x[1] for x in group),
                fastest8_mean_us=statistics.mean(durations[:8]),
                slowest16_mean_us=statistics.mean(durations[8:]),
                max_duration_us=durations[-1],
                lanes=[dict(core=x[2], offset_us=x[0]-first, duration_us=x[1]) for x in group]))
    fields = ('span_us', 'mean_core_busy_us', 'front_capacity_lost_us', 'tail_capacity_lost_us',
              'start_skew_us', 'end_skew_us', 'fastest8_mean_us', 'slowest16_mean_us', 'max_duration_us')
    summary = {k: dict(mean=statistics.mean(x[k] for x in rows), median=statistics.median(x[k] for x in rows))
               for k in fields}
    scale = 1000*len(periods)
    return dict(source=str(analysis), source_inventory_sha256=hashlib.sha256((rank/'inventory.json').read_bytes()).hexdigest(),
        periods=periods, launches=len(rows), summary_us=summary,
        ideal_full24_core_capacity_ceiling_ms=sum(x['span_us']-x['mean_core_busy_us'] for x in rows)/scale,
        sampler=[dict(kernel=k, union_ms_per_round=duration_union(v)/scale,
                      physical_descriptors_per_round=len(v)/len(periods)) for k, v in sorted(sampling.items())],
        rows=rows, new_trace=False, gain_measured=False,
        limitation='Capacity includes scheduling, geometry and stalls; overlap is not deducted. Not a critical-path gain.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.analysis)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('launches', 'summary_us', 'ideal_full24_core_capacity_ceiling_ms', 'sampler')}))
