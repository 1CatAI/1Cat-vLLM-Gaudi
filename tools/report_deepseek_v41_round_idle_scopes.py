# SPDX-License-Identifier: Apache-2.0
"""Locate recorded four-card inactivity against existing raw-clock host scopes.

Scope overlap is observation, not proof that the host caused a device wait.
No additional acquisition, fitted clocks, or inferred tensor traffic is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import statistics

from report_deepseek_v41_raw_entry_periods import merge


def load_rank(arguments):
    analysis, rank, low, high, selector = arguments
    directory = analysis / f'rank{rank}'
    inventory = json.loads((directory / 'inventory.json').read_text())
    base = inventory['base_time_nanoseconds'] / 1000
    spans = []
    with gzip.open(directory / 'hardware.jsonl.gz', 'rt') as source:
        for line in source:
            start, duration, _, index, *_ = json.loads(line)
            node = inventory['nodes'][index]
            if selector == 'compute' and (node['engine'] not in ('TPC', 'MME') or node['kernel'] == 'null'):
                continue
            a, b = base + start, base + start + duration
            if b > low and a < high and duration > 0:
                spans.append((max(a, low), min(b, high)))
    scopes = []
    with gzip.open(directory / 'host.jsonl.gz', 'rt') as source:
        for line in source:
            row = json.loads(line)
            # Explicit application scopes are captured in MONOTONIC_RAW;
            # do not join raw Synapse API events through an assumed clock.
            if not str(row[5]).startswith('v41::') or row[4] != 'user_annotation':
                continue
            a, b = base + row[0], base + row[0] + row[1]
            if b > low and a < high:
                scopes.append(dict(start_us=a, end_us=b, name=row[5]))
    return dict(rank=rank, spans=merge(spans), scopes=scopes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--activity-selector', choices=('all', 'compute'), default='all')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing analysis')
    ledger = json.loads((args.analysis / 'RAW_DEVICE_CYCLE_LEDGER.json').read_text())
    windows = ledger['periods']
    low, high = windows[0]['start_raw_us'], windows[-1]['end_raw_us']
    for rank in range(4):
        provenance = json.loads((args.analysis / f'rank{rank}' / 'raw-provenance.json').read_text())
        if provenance['metadata'].get('scope_clock_domain') != 'CLOCK_MONOTONIC_RAW':
            raise ValueError('Explicit scope clock domain is not proven raw')
    with ProcessPoolExecutor(max_workers=2) as pool:
        ranks = list(pool.map(load_rank, [(args.analysis, r, low, high, args.activity_selector) for r in range(4)]))
    active = merge([span for rank in ranks for span in rank['spans']])
    idle, cursor = [], low
    for a, b in active:
        if a > cursor:
            idle.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < high:
        idle.append((cursor, high))
    rounds = []
    for index, window in enumerate(windows):
        a, b = window['start_raw_us'], window['end_raw_us']
        gaps = [(max(a, x), min(b, y)) for x, y in idle if x < b and y > a]
        rank_rows = []
        for rank in ranks:
            by_name, observed = {}, []
            for scope in rank['scopes']:
                intersections = [(max(x, scope['start_us']), min(y, scope['end_us'])) for x, y in gaps
                                 if x < scope['end_us'] and y > scope['start_us']]
                if intersections:
                    by_name.setdefault(scope['name'], []).extend(intersections)
                    observed.extend(intersections)
            covered = sum(y - x for x, y in merge(observed)) / 1000
            rank_rows.append(dict(rank=rank['rank'], annotated_scope_overlap_ms=covered,
                                  by_scope_overlap_ms={name: sum(y - x for x, y in merge(spans)) / 1000
                                                       for name, spans in by_name.items()}))
        total = sum(y - x for x, y in gaps) / 1000
        rounds.append(dict(round=index, four_cards_no_selected_activity_ms=total,
                           gaps_raw_us=gaps, rank_scope_overlaps=rank_rows))
    result = dict(source=str(args.analysis), rounds=rounds, activity_selector=args.activity_selector,
                  mean_four_cards_no_selected_activity_ms=statistics.mean(
                      row['four_cards_no_selected_activity_ms'] for row in rounds),
                  mean_scope_overlap_ms_by_rank=[statistics.mean(
                      row['rank_scope_overlaps'][rank]['annotated_scope_overlap_ms'] for row in rounds)
                      for rank in range(4)],
                  limitations=['Compute-only gaps include DMA, HCL and descriptors; '
                               'all-activity gaps can include unrecorded waits.',
                               'Nested scopes overlap; per-name values must not be summed.',
                               'Scopes-only acquisition does not count every item/cpu/scheduler operation.',
                               'Same-phase physical cycles are not drained Target critical-path intervals.'],
                  causal_host_wait_proved=False, acquisition_performed=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rounds'}, indent=2))


if __name__ == '__main__':
    main()
