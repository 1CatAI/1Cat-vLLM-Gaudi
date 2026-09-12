# SPDX-License-Identifier: Apache-2.0
"""Audit tensor producers, consumers and work distribution in a captured trace."""
import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from analyze_deepseek_v41_trace import symbols, union
from account_deepseek_v41_trace import merge, subtract, length
from analyze_deepseek_v41_resources import clip_windows


def statistics_ms(v):
    return dict(n=len(v), mean=statistics.mean(v) if v else None, min=min(v) if v else None, max=max(v) if v else None)


def main(rank, ROOT, OUT):
    p = ROOT / f'rank{rank}'
    inv = json.loads((p / 'inventory.json').read_text())
    recipes = json.loads((p / 'recipe-symbols.json').read_text())['recipes']
    mapped = symbols(inv, recipes)
    rows = json.loads((p / 'node-breakdown.json').read_text())
    bykey = {(r['recipe_id'], r['engine'], r['context_id']): r for r in rows}
    common = json.loads((ROOT / 'common-windows.json').read_text())
    windows = common['windows_us']
    ends = [b for a, b in windows]
    scale = len(windows) * 1000
    own = json.loads((p / 'device-windows.json').read_text())
    freq = collections.Counter(x[2].split(':')[0] for x in own['capture_order'])
    selected = {}
    reasons = {}
    for key, r in bykey.items():
        if 'mxfp4_prepared_dequant' in r['kernel']:
            reasons[key] = 'expert_decode'
        elif 'decoded_write' in r['kernel']:
            reasons[key] = 'incremental_kv_write'
        elif 'selected_kv' in r['kernel']:
            reasons[key] = 'selected_kv'
        elif ('sparse_attn' in r['kernel'] or 'decoded_attn' in r['kernel']) and 'bf16' in r['kernel']:
            reasons[key] = 'sparse_attention'
        elif r['engine'] == 'MME' and r['category'] == '路由专家':
            reasons[key] = 'expert_mme'
        elif r['kernel'] == 'DmaMemcpy' and r['inputs'] and r['inputs'][0]['shape'] == [1, 4096, 1024]:
            reasons[key] = 'woa_copy'
        elif r['engine'] == 'MME' and r['purpose'] in ('wo_a 分组输出 BMM', 'wo_a 分组输出 GEMM'):
            reasons[key] = 'woa_mme'
    for i, symbol in mapped.items():
        node = inv['nodes'][i]
        key = (node['recipe'].split(':')[0], node['engine'], symbol['full_context_id'])
        if key in reasons:
            selected[i] = key
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    all_compute = []
    with gzip.open(p / 'hardware.jsonl.gz', 'rt') as stream:
        for line in stream:
            start, dur, lane, i = json.loads(line)
            for w, left, right in clip_windows(start, start + dur, windows, ends):
                if inv['nodes'][i]['engine'] in ('TPC', 'MME'):
                    all_compute.append((left, right))
                if i in selected:
                    data[selected[i]][w].append((left, right, lane))
    calls = {}
    details = []
    spans = collections.defaultdict(list)
    raw = []
    for key, bywin in data.items():
        r = bykey[key]
        symbol = r['compiler_contract']['symbol']
        why = reasons[key]
        samples = []
        packs = symbol['working_engines']
        allspans = [row[:2] for rs in bywin.values() for row in rs]
        spans[why] += allspans
        complete = True
        for w, rs in bywin.items():
            rs.sort()
            if r['engine'] == 'TPC':
                packet_count = sum(packs)
            elif r['engine'] == 'MME':
                packet_count = len({x[2] for x in rs})
            else:
                continue
            if not packet_count or len(rs) != packet_count * freq[key[0]]:
                complete = False
                continue
            c = []
            for offset in range(0, len(rs), packet_count):
                bunch = rs[offset:offset + packet_count]
                a = min(x[0] for x in bunch)
                b = max(x[1] for x in bunch)
                lane_groups = collections.defaultdict(list)
                for x, y, lane in bunch:
                    lane_groups[lane].append((x, y))
                events = []
                for ls in lane_groups.values():
                    for x, y in merge(ls):
                        events.extend([(x, 1), (y, -1)])
                active = peak = 0
                for t, delta in sorted(events):
                    active += delta
                    peak = max(peak, active)
                coretime = sum(union(ls) for ls in lane_groups.values())
                c.append(
                    dict(start=a,
                         end=b,
                         wall_ms=(b - a) / 1000,
                         active_cores_mean=coretime / (b - a),
                         peak_cores=peak,
                         unique_lanes=len(lane_groups)))
            assert not any(a['end'] > b['start'] for a, b in zip(c, c[1:])), key
            samples += c
            calls[(key, w)] = c
        detail = dict(key=key,
                      purpose=r['purpose'],
                      kind=why,
                      source=r['source_node'],
                      inputs=r['inputs'],
                      outputs=r['outputs'],
                      graph=r['compiler_contract']['graph'],
                      working_engines=packs,
                      complete=complete,
                      activity_ms=union(allspans) / scale,
                      wall_ms=statistics_ms([x['wall_ms'] for x in samples]),
                      active_cores_mean=statistics.mean(x['active_cores_mean'] for x in samples) if samples else None,
                      peak_cores_histogram=dict(collections.Counter(x['peak_cores'] for x in samples)),
                      unique_lanes_histogram=dict(collections.Counter(x['unique_lanes'] for x in samples)))
        details.append(detail)
        raw.append(dict(key=key, first_window_lane_events=bywin.get(0, []), first_window_calls=calls.get((key, 0), [])))
    # Join exact tensor producers with MME input names within one serialized recipe.
    producers = {
        (key[0], r['outputs'][0]['name']): key
        for key, r in bykey.items() if reasons.get(key) in ('expert_decode', 'woa_copy')
    }
    paired = []
    for key, r in bykey.items():
        if reasons.get(key) not in ('expert_mme', 'woa_mme'):
            continue
        prod = producers.get((key[0], r['inputs'][1]['name']))
        if prod is None:
            if reasons[key] == 'woa_mme' and r['inputs'][1]['location'] == 'DRAM':
                paired.append(
                    dict(producer=None,
                         consumer=key,
                         kind='woa_direct_weight',
                         input_bytes=r['inputs'][1]['bytes'],
                         qualification='MME reads DRAM operand; no separate decoded/copy producer. '
                         'Memory stalls are included in MME activity and not independently measured.'))
                continue
            raise RuntimeError(('missing producer', key))
        wait = []
        chain = []
        sample = None
        incomplete_windows = []
        if reasons[prod] == 'expert_decode':
            for w in range(len(windows)):
                aa, bb = calls.get((prod, w), []), calls.get((key, w), [])
                if len(aa) != len(bb):
                    # A common window may cut a different rank's invocation.
                    # Retain all activity but do not invent matching endpoints.
                    incomplete_windows.append({
                        'window': w,
                        'reconstructed_producers': len(aa),
                        'reconstructed_consumers': len(bb)
                    })
                    continue
                for a, b in zip(aa, bb):
                    wait.append((b['start'] - a['end']) / 1000)
                    chain.append((b['end'] - a['start']) / 1000)
                if w == 0:
                    sample = {'producer': aa, 'consumer': bb}
        else:
            # EDMA descriptors do not expose physical-copy invocation boundaries. Use
            # only windows bounded by each consuming BMM, retaining conservative envelope.
            for w in range(len(windows)):
                packets = sorted(data.get(prod, {}).get(w, []))
                mmes = calls.get((key, w), [])
                previous = windows[w][0]
                for mme in mmes:
                    a = [x for x in packets if previous <= x[0] < mme['end']]
                    if a:
                        wait.append((mme['start'] - max(x[1] for x in a)) / 1000)
                        chain.append((mme['end'] - min(x[0] for x in a)) / 1000)
                    previous = mme['end']
            # Envelope is NOT a copy duration. Negative wait indicates in-node pipeline.
        paired.append(
            dict(producer=prod,
                 consumer=key,
                 kind=reasons[prod],
                 input_bytes=r['inputs'][1]['bytes'],
                 wait_or_overlap_ms=statistics_ms(wait),
                 chain_envelope_ms=statistics_ms(chain),
                 incomplete_windows=incomplete_windows,
                 pairing_qualification='Endpoint statistics use only equally reconstructed complete calls; '
                 'activity unions retain all clipped packets.',
                 first_window_example=sample))
    es = merge(spans['expert_decode'])
    em = merge(spans['expert_mme'])
    wc = merge(spans['woa_copy'])
    wm = merge(spans['woa_mme'])
    compute = merge(all_compute)
    summary = dict(rank=rank,
                   tokens=len(windows),
                   window_ms=length(windows) / scale,
                   expert_decode_union_ms=length(es) / scale,
                   expert_mme_union_ms=length(em) / scale,
                   expert_overlap_ms=(length(es) + length(em) - length(merge(es + em))) / scale,
                   expert_decode_without_expert_mme_ms=length(subtract(es, em)) / scale,
                   expert_mme_without_decode_ms=length(subtract(em, es)) / scale,
                   woa_copy_union_ms=length(wc) / scale,
                   woa_mme_union_ms=length(wm) / scale,
                   woa_copy_without_any_compute_ms=length(subtract(wc, compute)) / scale,
                   woa_copy_without_woa_mme_ms=length(subtract(wc, wm)) / scale,
                   details=details,
                   pairs=paired,
                   method=('Common capture window; same-context TPC/MME invocation reconstruction; '
                           'EDMA copies remain descriptor intervals; no profiler rerun'))
    (OUT / f'rank{rank}-dataflow.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (OUT / f'rank{rank}-first-token-lanes.json').write_text(json.dumps(raw, indent=2) + '\n')
    print(json.dumps({
        k: v
        for k, v in summary.items() if k not in ('details', 'pairs')
    }, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_root', type=Path)
    parser.add_argument('--rank', type=int, choices=range(4), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args.rank, args.trace_root, args.output_dir)
