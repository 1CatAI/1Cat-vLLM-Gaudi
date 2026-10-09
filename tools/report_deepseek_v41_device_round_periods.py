# SPDX-License-Identifier: Apache-2.0
"""Use a repeated physical C6 producer as the cycle boundary, not host enqueue.

The explicit source-node identity is verified against each rank's recipe
symbols. One selected physical MME lane emits exactly once per acquisition
round. No timestamp fitting or gap-based invocation clustering is performed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import statistics


def anchors(arguments):
    analysis, rank, node_name, lane, allow_partial = arguments
    root = analysis / f'rank{rank}'
    inv = json.loads((root / 'inventory.json').read_text())
    indices = [i for i, node in enumerate(inv['nodes']) if node['engine'] == 'MME' and node['node'] == node_name]
    if len(indices) != 1 or not inv['nodes'][indices[0]].get('metadata_provenance'):
        raise ValueError(f'Rank{rank}: producer identity is missing or ambiguous')
    node = inv['nodes'][indices[0]]
    before, after = [json.loads((analysis.parent / 'traces' / f'rank{rank}-native-profile-{phase}.json').read_text())
                     for phase in ('start', 'stop')]
    count = after['v41']['decode_steps'] - before['v41']['decode_steps']
    if before['native_program_generation'] != after['native_program_generation']:
        raise ValueError('Native acquisition generation changed')
    values = []
    with gzip.open(root / 'hardware.jsonl.gz', 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if event[3] == indices[0] and event[2] == lane:
                values.append(inv['base_time_nanoseconds'] / 1000 + event[0])
    values.sort()
    partial = 4 <= len(values) < count
    if ((len(values) != count and not (allow_partial and partial)) or len(values) < 4
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError(f'Rank{rank}: {len(values)} producer events differ from {count} acquisition rounds')
    return dict(rank=rank, lane=lane, node=node, node_index=indices[0], starts_raw_us=values,
                acquisition_rounds=count, recorded_producers=len(values), partial_acquisition=partial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--node', required=True, nargs='+',
                        help='One identical name or four exact rank-specific first Target projection names')
    parser.add_argument('--lane', default='[D0] MME')
    parser.add_argument('--allow-partial-acquisition', action='store_true',
                        help='Report only complete recorded producer cycles; retain unmatched acquisition counters')
    args = parser.parse_args()
    if len(args.node) not in (1, 4):
        parser.error('Provide one or four explicit producer identities')
    nodes = args.node * 4 if len(args.node) == 1 else args.node
    path = args.analysis / 'RAW_DEVICE_CYCLE_LEDGER.json'
    if path.exists():
        raise FileExistsError('Reuse the preserved device-cycle ledger')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(anchors, [(args.analysis, rank, nodes[rank], args.lane,
                                      args.allow_partial_acquisition) for rank in range(4)]))
    if len({row['acquisition_rounds'] for row in rows}) != 1:
        raise ValueError('Four ranks have different captured round counts')
    if len({row['recorded_producers'] for row in rows}) != 1:
        raise ValueError('Four ranks have different recorded producer counts')
    starts = rows[0]['starts_raw_us']
    periods = [dict(position=index, next_position=index + 1, start_raw_us=a, end_raw_us=b,
                    period_ms=(b - a) / 1000,
                    producer_arrival_skew_us=max(row['starts_raw_us'][index] for row in rows)
                                             - min(row['starts_raw_us'][index] for row in rows))
               for index, (a, b) in enumerate(zip(starts, starts[1:]))]
    result = dict(method='Consecutive physical first Target QKV projection starts on one recorded MME lane',
                  source='Existing capture only; exact recipe/context/source-node and native round-count check',
                  round_boundaries_measured=True, completion_windows=False, critical_path=False,
                  limit='Same-phase hardware cycles include the full round. They are not drained Target latency.',
                  rounds=len(periods), mean_period_ms=statistics.mean(p['period_ms'] for p in periods),
                  partial_acquisition=any(row['partial_acquisition'] for row in rows),
                  acquisition_counter_equality=all(not row['partial_acquisition'] for row in rows),
                  periods=periods, anchors=rows)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(rounds=result['rounds'], mean_period_ms=result['mean_period_ms'])), flush=True)


if __name__ == '__main__':
    main()
