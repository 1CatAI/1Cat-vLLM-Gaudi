# SPDX-License-Identifier: Apache-2.0
"""Report engine activity and TPC concurrency in a captured common token window."""
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

from analyze_deepseek_v41_trace import symbols
from account_deepseek_v41_trace import merge, length, subtract


def tags(row):
    k, p, cat = row['kernel'], row['purpose'], row['category']
    names = []
    if 'mxfp4_prepared_dequant' in k:
        names.append('expert_decode')
    if row['engine'] == 'MME' and cat == '路由专家':
        names.append('expert_mme')
    if k == 'DmaMemcpy' and row['inputs'] and row['inputs'][0]['shape'] == [1, 4096, 1024]:
        names.append('woa_copy')
    if p == 'wo_a 分组输出 BMM':
        names.append('woa_mme')
    if row['engine'] == 'MME' and cat == 'Attention':
        names.append('attention_all_projection_mme')
    if 'sparse_attn_bf16' in k:
        names.append('sparse_attention')
    if 'selected_kv' in k:
        names.append('selected_kv')
    if 'generate_bitonic_chunks' in k:
        names.append('router_bitonic')
    if row['engine'] == 'MME' and cat == 'Router':
        names.append('router_score_mme')
    if row['engine'] == 'MME' and cat == '共享专家':
        names.append('shared_mme')
    if row['engine'] == 'MME' and cat == '输出头':
        names.append('head_mme')
    if 'arg_max' in k:
        names.append('argmax')
    if row['engine'] in ('TPC', 'MME'):
        names.append('all_compute')
    return names


def occupancy(lanes, selected):
    # Intersect all-core activity with a specified wall-time union. Events
    # from unrelated kernels are excluded from expert-only lanes, retained
    # for all-TPC lanes. Busy time does not measure instruction issue rate.
    events = []
    for intervals in lanes.values():
        overlap = subtract(merge(intervals), subtract(merge(intervals), selected))
        for a, b in overlap:
            events.extend(((a, 1), (b, -1)))
    hist = collections.defaultdict(float)
    active = 0
    previous = None
    for t, delta in sorted(events):
        if previous is not None and active:
            hist[active] += t - previous
        active += delta
        previous = t
    union = length(selected)
    if not union:
        return {
            'busy_cores_mean_in_selected_window': None,
            'peak_busy_cores': None,
            'busy_cores_wall_fraction': {},
            'reason': 'No evidenced window'
        }
    hist[0] = max(0., union - sum(hist.values()))
    return {
        'busy_cores_mean_in_selected_window': sum(k * v for k, v in hist.items()) / union,
        'peak_busy_cores': max(hist, default=0),
        'busy_cores_wall_fraction': {
            k: v / union
            for k, v in sorted(hist.items())
        }
    }


def clip_windows(start, end, windows, ends):
    """Retain every overlap, including activity crossing a token boundary."""
    index = bisect.bisect_right(ends, start)
    while index < len(windows) and windows[index][0] < end:
        left, right = max(start, windows[index][0]), min(end, windows[index][1])
        if left < right:
            yield index, left, right
        index += 1


def stage_windows(own, markers, boundary_spans):
    """Bound the stage by its first/last recipe's actual device execution."""
    enqueues = sorted(t for t, _, name in markers if name == 'vllm_gaudi::native_decoder_enqueue')
    grouped = collections.defaultdict(list)
    for call in own['calls']:
        if call['complete']:
            grouped[call['token']].append(call)
    result = []
    for token, calls in sorted(grouped.items()):
        calls.sort(key=lambda call: call['layer'])
        if [call['layer'] for call in calls] != list(range(20)):
            continue
        index = bisect.bisect_right(enqueues, calls[0]['start']) - 1
        if index < 0 or index + 1 == len(enqueues):
            continue
        lower, upper = enqueues[index:index + 2]
        starts = [a for a, _ in boundary_spans['first'] if lower <= a < calls[0]['start']]
        ends = [b for a, b in boundary_spans['last'] if calls[-1]['end'] <= a < upper]
        if starts and ends:
            result.append({'token': token, 'start_us': min(starts), 'end_us': max(ends), 'enqueue_us': lower})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ROOT, OUT = args.trace_root.resolve(), args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    windows = json.loads((ROOT / 'common-windows.json').read_text())['windows_us']
    ends = [b for a, b in windows]
    scale = len(windows) * 1000
    global_spans = collections.defaultdict(list)
    per_rank = []
    for rank in range(4):
        root = ROOT / f'rank{rank}'
        inv = json.loads((root / 'inventory.json').read_text())
        own = json.loads((root / 'device-windows.json').read_text())
        recipe_order = [row[2].split(':')[0] for row in own['capture_order']]
        boundary_ids = {'first': recipe_order[0], 'last': recipe_order[-1]}
        boundary_spans = collections.defaultdict(list)
        recipes = json.loads((root / 'recipe-symbols.json').read_text())['recipes']
        mapped = symbols(inv, recipes)
        rows = json.loads((root / 'node-breakdown.json').read_text())
        keyed = {(r['recipe_id'], r['engine'], r['context_id']): tags(r) for r in rows}
        tag_by_id = {}
        for i, symbol in mapped.items():
            node = inv['nodes'][i]
            key = (node['recipe'].split(':')[0], node['engine'], symbol['full_context_id'])
            tag_by_id[i] = keyed.get(key, [])
        spans = collections.defaultdict(list)
        expert_lanes, tpc_lanes = collections.defaultdict(list), collections.defaultdict(list)
        with gzip.open(root / 'hardware.jsonl.gz', 'rt') as stream:
            for line in stream:
                a, d, lane, i = json.loads(line)
                b = a + d
                node = inv['nodes'][i]
                if node['engine'] in ('TPC', 'MME', 'DMA'):
                    for name, rid in boundary_ids.items():
                        if node['recipe'].split(':')[0] == rid:
                            boundary_spans[name].append((a, b))
                names = tag_by_id.get(i, [])
                for _, left, right in clip_windows(a, b, windows, ends):
                    for tag in names:
                        spans[tag].append((left, right))
                    spans['engine_' + node['engine']].append((left, right))
                    if 'expert_decode' in names:
                        expert_lanes[lane].append((left, right))
                    if node['engine'] == 'TPC':
                        tpc_lanes[lane].append((left, right))
        unions = {tag: merge(v) for tag, v in spans.items()}
        for tag, v in unions.items():
            global_spans[tag].extend(v)
        stages = stage_windows(own, inv['cpu_markers'], boundary_spans)
        stage_union = merge([(left, right) for stage in stages
                             for _, left, right in clip_windows(stage['start_us'], stage['end_us'], windows, ends)])
        stage_duration = length(stage_union)
        stats = {
            'rank':
            rank,
            'activity_union_ms': {
                tag: length(v) / scale
                for tag, v in unions.items()
            },
            'expert_only_occupancy':
            occupancy(expert_lanes, unions['expert_decode']),
            'all_tpc_occupancy_during_expert_decode':
            occupancy(tpc_lanes, unions['expert_decode']),
            'all_tpc_occupancy_in_token':
            occupancy(tpc_lanes, windows),
            'all_tpc_occupancy_in_stage':
            occupancy(tpc_lanes, stage_union),
            'stage_device_envelope_ms_per_token':
            stage_duration / scale,
            'stage_windows':
            stages,
            'engine_activity_fraction_in_stage': {
                tag: (length(value) - length(subtract(value, stage_union))) / stage_duration if stage_duration else None
                for tag, value in unions.items() if tag.startswith('engine_')
            },
            'unknown_resource_metrics': [
                'instruction issue/stall counters', 'register spills/local memory',
                'physical HBM/SRAM traffic counters', 'MME peak throughput fraction'
            ]
        }
        per_rank.append(stats)
        print('completed rank', rank, flush=True)
    unions = {tag: merge(v) for tag, v in global_spans.items()}
    chains = {
        'expert_decode_and_mme': ['expert_decode', 'expert_mme'],
        'selected_kv_and_attention': ['selected_kv', 'sparse_attention'],
        'woa_copy_and_mme': ['woa_copy', 'woa_mme'],
        'router_score_and_bitonic': ['router_score_mme', 'router_bitonic']
    }
    result = {
        'capture':
        str(ROOT),
        'periods':
        len(windows),
        'common_window_ms':
        length(windows) / scale,
        'per_rank':
        per_rank,
        'global_activity_union_ms': {
            tag: length(v) / scale
            for tag, v in unions.items()
        },
        'chain_activity_union_ms': {
            name: length(merge([x for tag in parts for x in unions[tag]])) / scale
            for name, parts in chains.items()
        },
        'woa_copy_without_any_rank_compute_ms':
        length(subtract(unions['woa_copy'], unions['all_compute'])) / scale,
        'limitations': [
            'Overlapping activity unions are not additive or removable wall time.',
            'Busy TPC cores do not imply useful instruction issue; hardware stall counters are unavailable here.',
            'DMA remains descriptor activity; no physical-copy invocation count inferred.'
        ]
    }
    (OUT / 'resource-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_rank'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
