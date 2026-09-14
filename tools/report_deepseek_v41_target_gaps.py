# SPDX-License-Identifier: Apache-2.0
"""Attribute saved Target gaps by intersecting host intervals, without re-profiling."""
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

import ijson
from report_deepseek_v41_target_hardware import merge


def host_class(cat, name):
    if name == 'compileGraph':
        return 'compile'
    if name.startswith(('synchronize', 'scal_completion_group_wait', 'hcclSynchronize')):
        return 'wait'
    if name.startswith(('enqueue', 'scal_stream_submit', 'hcclAll', 'eventRecord', 'streamWaitEvent')) or name in ('Launch', 'launch', 'LaunchRecipe', 'launch_recipe', 'vllm_gaudi::native_decoder_enqueue', 'vllm_gaudi::native_decoder_publish'):
        return 'submit'
    if cat == 'hpu_op':
        return 'runtime_other'
    if cat == 'cpu_op' and name.startswith('Torch-Compiled Region'):
        return 'compiled_wrapper'
    if cat in ('cpu_op', 'privateuse1_runtime'):
        return 'cpu_other'
    return None


def intersections(a, b, spans, ends):
    i = bisect.bisect_right(ends, a)
    while i < len(spans) and spans[i][0] < b:
        lo, hi = max(a, spans[i][0]), min(b, spans[i][1])
        if hi > lo:
            yield lo, hi
        i += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--stable-replay', action='store_true',
                        help='Both stages use one native entry; input preparation may contain compiled tensor updates')
    args = parser.parse_args()
    root = args.run
    out = root / 'target-gap-breakdown-01'
    out.mkdir(exist_ok=True)
    windows = json.loads((root / 'target-hardware-breakdown/selected-windows.json').read_text())
    if not (out / 'gaps.json').exists():
        spans = collections.defaultdict(list)
        with gzip.open(root / 'target-hardware-breakdown/kernel-details.json.gz', 'rb') as f:
            for node in ijson.items(f, 'item', use_float=True):
                for obs in node['observations']:
                    spans[str(node['rank']), str(obs['generation'])].extend((x[0], x[1]) for x in obs['events'])
        gaps = {}
        for rank, gens in windows.items():
            gaps[rank] = {}
            for gen, w in gens.items():
                cursor, found = w['start_us'], []
                for a, b in merge(spans[rank, gen]):
                    if a > cursor:
                        found.append([cursor, a])
                    cursor = max(cursor, b)
                if cursor < w['end_us']:
                    found.append([cursor, w['end_us']])
                gaps[rank][gen] = found
        (out / 'gaps.json').write_text(json.dumps(gaps))
    gaps = json.loads((out / 'gaps.json').read_text())
    result = {}
    for rank, gens in gaps.items():
        allgaps = sorted(x for spans in gens.values() for x in spans)
        ends = [x[1] for x in allgaps]
        named = collections.defaultdict(list)
        classes = collections.defaultdict(list)
        host_events = []
        counts = collections.Counter()
        with gzip.open(root / f'trace-analysis/rank{rank}/host.jsonl.gz', 'rt') as f:
            for line in f:
                ts, dur, pid, tid, cat, name = json.loads(line)
                host_events.append([ts, dur, pid, tid, cat, name])
                clipped = list(intersections(ts, ts + dur, allgaps, ends))
                if clipped:
                    key = tid, cat, name
                    named[key].extend(clipped)
                    counts[key] += 1
                    label = host_class(cat, name)
                    if label:
                        classes[label].extend(clipped)
        rows = []
        for key, spans in named.items():
            rows.append(dict(tid=key[0], cat=key[1], name=key[2],
                             gap_overlap_ms=sum(b-a for a,b in merge(spans))/len(gens)/1000,
                             calls=counts[key]))
        rows.sort(key=lambda x: -x['gap_overlap_ms'])
        boundaries = collections.defaultdict(list)
        for a, b in allgaps:
            boundaries[a].append(('gap', 1)); boundaries[b].append(('gap', -1))
        for label, spans in classes.items():
            for a, b in merge(spans):
                boundaries[a].append((label, 1)); boundaries[b].append((label, -1))
        active, ledger = collections.Counter(), collections.Counter()
        points = sorted(boundaries)
        blank = []
        for a, b in zip(points, points[1:]):
            for label, delta in boundaries[a]:
                active[label] += delta
            if active['gap']:
                key = '|'.join(sorted(k for k, n in active.items() if n and k != 'gap')) or 'unobserved_host'
                ledger[key] += (b-a)/len(gens)/1000
                if key == 'unobserved_host':
                    blank.append([a,b])
        (out / f'rank{rank}-host-blank.json').write_text(json.dumps(merge(blank)))
        host_events.sort()
        phases = []
        phase_totals = collections.Counter()
        for gen, gaps_gen in gens.items():
            w = windows[rank][gen]
            lo = w['record']['host']['round_start']/1000 + w['host_to_trace_us']
            hi = w['record']['host']['target_submit_done']/1000 + w['host_to_trace_us']
            inventory = json.loads((root / f'trace-analysis/rank{rank}/inventory.json').read_text())
            pid = str(next(e['pid'] for e in inventory['metadata']
                           if e['name'] == 'process_name' and e.get('args', {}).get('name', '').startswith('VLLM::Worker')))
            es = [e for e in host_events if lo <= e[0] < hi]
            chunks = [e for e in es if e[3] == pid and e[5].startswith('Torch-Compiled Region')]
            replays = [e for e in es if e[3] == pid and e[5] == 'vllm_gaudi::native_decoder_enqueue']
            bounds = []
            if args.stable_replay:
                assert len(replays) == 1, (rank, gen, len(replays))
                t, enqueue_end = replays[0][0], replays[0][0] + replays[0][1]
                assert all(e[0] + e[1] <= t for e in chunks), (rank, gen, 'compiled wrapper during decoder replay')
                bounds = [(w['start_us'], t, 'Stage input preparation before joint replay'),
                          (t, enqueue_end, 'Native joint replay enqueue'),
                          (enqueue_end, w['end_us'], 'Queued work after joint replay enqueue')]
            elif int(rank) < 2:
                assert len(replays) == 1 and not chunks, (rank, gen)
                t = replays[0][0]
                bounds = [(w['start_us'], t, 'PP0 input preparation before native replay'),
                          (t, w['end_us'], 'PP0 after native replay submission')]
            else:
                assert len(chunks) == 5, (rank, gen, len(chunks))
                if replays:
                    last = chunks[-1][0] + chunks[-1][1]
                    sync = [e for e in es if e[0] >= last and e[5] == 'synchronizeStream (accel0)']
                    assert len(sync) == 1, (rank, gen, sync)
                    sa, sb = sync[0][0], sync[0][0]+sync[0][1]
                    restores = [e for e in es if e[0] >= sb and e[3] == pid and e[5] == 'aten::copy_']
                    assert restores
                    rb = restores[0][0]
                    re = replays[0][0] + replays[0][1]
                    bounds = [(w['start_us'], last, 'PP1 cached group wrappers before capture'),
                              (last, sa, 'PP1 snapshot and native capture before instantiate barrier'),
                              (sa, sb, 'PP1 instantiate stream barrier'),
                              (sb, rb, 'PP1 native plan finalization before state restore'),
                              (rb, re, 'PP1 state restore and final replay enqueue'),
                              (re, w['end_us'], 'PP1 queued work after final replay enqueue')]
                else:
                    cursor = w['start_us']
                    for e in chunks:
                        bounds.append((cursor, e[0], 'PP1 invalidation or inter-group boundary'))
                        bounds.append((e[0], e[0]+e[1], 'PP1 five group plan preparation and ordinary execution'))
                        cursor = e[0]+e[1]
                    bounds.append((cursor, w['end_us'], 'PP1 work after last group wrapper'))
            n = 0.
            for a,b,label in bounds:
                clipped = list(intersections(max(a,w['start_us']), min(b,w['end_us']), gaps_gen, [x[1] for x in gaps_gen])) if min(b,w['end_us']) > max(a,w['start_us']) else []
                ms = sum(y-x for x,y in clipped)/1000
                n += ms
                phase_totals[label] += ms/len(gens)
                if clipped:
                    phases.append(dict(generation=int(gen), phase=label, start_us=max(a,w['start_us']), end_us=min(b,w['end_us']), gap_ms=ms, gaps=clipped))
            assert abs(n-sum(b-a for a,b in gaps_gen)/1000) < 1e-8, (rank,gen,n)
        (out / f'rank{rank}-phase-intervals.json').write_text(json.dumps(phases,indent=2))
        result[rank] = dict(gap_ms=sum(b-a for a,b in allgaps)/len(gens)/1000, rows=rows,
                            combinations=dict(ledger.most_common()),
                            unions_ms={k:sum(b-a for a,b in merge(v))/len(gens)/1000 for k,v in classes.items()},
                            phase_ms=dict(phase_totals))
        print('RANK', rank, 'GAP', result[rank]['gap_ms'], flush=True)
        print('PHASES', result[rank]['phase_ms'], flush=True)
    (out / 'host-inclusive-overlap.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
