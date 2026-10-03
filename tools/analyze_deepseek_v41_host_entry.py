# SPDX-License-Identifier: Apache-2.0
"""Compare explicit four-worker host entry scopes on one shared monotonic clock.

Nested phase times are retained individually and are never added. Device stage
arrival and span must be obtained from the companion hardware trace.
"""
import argparse
import gzip
import json
from pathlib import Path
import re
import statistics


def summarize(values):
    values = sorted(values)
    if not values:
        return None
    return dict(count=len(values), median=statistics.median(values),
                p95=values[min(len(values)-1, int(.95*(len(values)-1)))], maximum=max(values))


def host_entries(document):
    if document.get('clockDomain') != 'CLOCK_MONOTONIC_RAW':
        raise ValueError('Four-card host alignment requires explicit monotonic-raw scopes')
    base = document['baseTimeNanoseconds'] / 1000
    events = [row for row in document['traceEvents'] if row.get('ph') == 'X']
    commits = sorted((row for row in events if row['name'].startswith('v41::worker_commit::')), key=lambda r: r['ts'])
    rows = []
    for index, commit in enumerate(commits):
        match = re.search(r'::P(\d+)(?:::|$)', commit['name'])
        if match is None:
            continue
        stop = commits[index+1]['ts'] if index+1 < len(commits) else float('inf')
        spans = [row for row in events if row['tid'] == commit['tid']
                 and commit['ts'] <= row['ts'] < stop]
        phases = []
        for row in spans:
            detail = row.get('args', {})
            phases.append(dict(name=row['name'], start_raw_us=base+row['ts'], wall_us=row.get('dur', 0),
                               thread_cpu_us=detail['thread_cpu_ns']/1000 if 'thread_cpu_ns' in detail else None,
                               thread_nonrunning_us=(detail['thread_nonrunning_ns']/1000
                                                     if 'thread_nonrunning_ns' in detail else None),
                               voluntary_switches=detail.get('voluntary_switches'),
                               involuntary_switches=detail.get('involuntary_switches')))
        prefix = [row for row in phases if 'begin_segmented_from_input_ids' in row['name']]
        rows.append(dict(position=int(match[1])+1, commit_start_raw_us=base+commit['ts'],
                         prefix_call_start_raw_us=prefix[0]['start_raw_us'] if len(prefix)==1 else None,
                         phases=phases))
    return rows


def compare(ranks):
    if len(ranks) != 4:
        raise ValueError('Exactly four worker traces are required')
    positions = set.intersection(*(set(row['position'] for row in rows) for rows in ranks))
    lookup = [{row['position']: row for row in rows} for rows in ranks]
    aligned = []
    for position in sorted(positions):
        starts = [rank[position]['prefix_call_start_raw_us'] for rank in lookup]
        if all(value is not None for value in starts):
            aligned.append(dict(position=position, prefix_host_call_skew_us=max(starts)-min(starts)))
    return dict(complete_host_positions=len(positions), per_token=aligned,
                prefix_host_call_skew_us=summarize(row['prefix_host_call_skew_us'] for row in aligned),
                device_stage_skew_not_measured=True,
                note='Host call arrival is not device arrival; nested CPU/elapsed times must not be summed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs=4, help='Rank0..3 explicit-scope CPU captures')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ranks = []
    for path in args.traces:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            ranks.append(host_entries(json.load(stream)))
    result = dict(comparison=compare(ranks), ranks=ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
