# SPDX-License-Identifier: Apache-2.0
"""Additive per-round utilization/fragmentation ledger from an existing capture.

Raw host-entry windows are asynchronous submission periods. This report does
not rename activity unions as completed Target latency. Null TPC descriptors
count as physical activity, but never as useful TPC computation. Counts preserve
physical descriptors separately from reconstructed recipe/node invocations.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import statistics

from report_deepseek_v41_raw_entry_periods import merge
from report_deepseek_v41_c6_named_periods import classify, contract_index, node_contract


def spans_ms(spans):
    return sum(b - a for a, b in merge(spans)) / 1000


def intervals(edges, low, high, classify_active):
    active, result, previous = Counter(), defaultdict(float), low
    for stamp, changes in sorted(edges.items()):
        if stamp > previous:
            result[classify_active(active)] += (stamp - previous) / 1000
        for key, value in changes:
            active[key] += value
        previous = stamp
    if previous < high:
        result[classify_active(active)] += (high - previous) / 1000
    return dict(result)


def state(active, rank):
    tpc, mme = active[(rank, 'TPC')], active[(rank, 'MME')]
    if tpc and mme:
        return 'overlap'
    if tpc:
        return 'TPC_only'
    if mme:
        return 'MME_only'
    if any(active[(rank, engine)] for engine in ('HCL', 'DMA', 'TPC_null', 'MME_null', 'OTHER')):
        return 'HCL_DMA_descriptor_only'
    if any(value for (card, _), value in active.items() if card != rank):
        return 'single_card_idle_other_cards_active'
    return 'four_cards_idle'


def category_allocation(spans, low, high):
    """Separate exposed category activity from overlapping useful work.

    Count a category once even when its TPC and MME descriptors overlap.
    Missing useful compute remains separate from engine/communication idle;
    this allocation is activity evidence, not a dependency critical path.
    """
    edges = defaultdict(list)
    for category, values in spans.items():
        for a, b in merge(values):
            a, b = max(a, low), min(b, high)
            if a < b:
                edges[a].append((category, 1))
                edges[b].append((category, -1))

    def classify_categories(active):
        names = sorted(name for name, count in active.items() if count > 0)
        return ('exclusive:' + names[0] if len(names) == 1 else
                'overlap:' + '+'.join(names) if names else 'no_useful_compute')

    return intervals(edges, low, high, classify_categories)


def load_rank(directory):
    inventory = json.loads((directory / 'inventory.json').read_text())
    base = inventory.get('base_time_nanoseconds', 0) / 1000
    contracts = {}
    contract_path = directory / 'node-contracts.json'
    if contract_path.exists():
        contracts = contract_index(json.loads(contract_path.read_text()))
    starts = defaultdict(list)
    for start, _, recipe, *_ in inventory.get('device_recipe_starts', []):
        starts[str(recipe).split('@')[0]].append(base + start)
    for values in starts.values():
        values.sort()
    return inventory, base, contracts, starts


def native_submission_counts(before, after):
    """Use native counters without conflating replay calls with recipes.

    These snapshots cover the whole acquisition, including its terminal
    round. Host-entry windows usually exclude that round. Keep both scopes
    explicit rather than dividing the counters by the window count.
    """
    if before.get('native_program_generation') != after.get('native_program_generation'):
        return dict(status='generation changed; counters incomparable')
    count = after.get('v41', {}).get('decode_steps', 0) - before.get('v41', {}).get('decode_steps', 0)
    if count <= 0:
        return dict(status='no completed decode steps in counter scope')
    values = {}
    for key in ('native_entry_replays', 'native_joint_replays', 'native_joint_compute_submissions',
                'native_joint_hcl_callbacks'):
        if key in before and key in after:
            difference = after[key] - before[key]
            if difference < 0:
                return dict(status='counter reset; counters incomparable', counter=key)
            values[key] = difference / count
    return dict(status='measured native counter deltas', rounds=count, per_round=values,
                exact_repair_rounds=after.get('v41', {}).get('sampled_exact_repairs', 0) -
                before.get('v41', {}).get('sampled_exact_repairs', 0),
                limit='Native replay and compute-submission counters; not host synLaunch calls '
                      'or unique recipe count. Counter scope includes its terminal round.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Existing C1 anatomy JSON; no new trace')
    parser.add_argument('--reference-kernels', type=Path, help='Archived C1 all-kernels.json.gz')
    parser.add_argument('--serving-run', type=Path, help='Existing profile-start/stop snapshots; no acquisition')
    parser.add_argument('--window-ledger', type=Path, help='Verified physical producer cycle boundaries')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = json.loads((args.window_ledger or args.analysis / 'RAW_ENTRY_PERIOD_LEDGER.json').read_text())
    windows = ledger['periods']
    starts = [window['start_raw_us'] for window in windows]
    ends = [window['end_raw_us'] for window in windows]
    events = [defaultdict(list) for _ in windows]
    gap_spans = [[defaultdict(list) for _ in windows] for _ in range(4)]
    semantic_spans = [[defaultdict(list) for _ in windows] for _ in range(4)]
    counts = [[Counter() for _ in windows] for _ in range(4)]
    invocation = [[set() for _ in windows] for _ in range(4)]
    kernels = [defaultdict(lambda: dict(spans=[], descriptors=0, invocations=set(), contracts={}, active_nodes=set()))
               for _ in range(4)]
    host_sync = [[Counter() for _ in windows] for _ in range(4)]
    launches = [[Counter() for _ in windows] for _ in range(4)]
    metadata = []
    node_rows = []
    for rank in range(4):
        directory = args.analysis / f'rank{rank}'
        inventory, base, contracts, recipe_starts = load_rank(directory)
        metadata.append(dict(rank=rank, inventory=str(directory / 'inventory.json'),
                             unknown_nodes=sum(node.get('kernel') == 'null' for node in inventory['nodes'])))
        for recipe, stamps in recipe_starts.items():
            for stamp in stamps:
                i = bisect_right(starts, stamp) - 1
                if i >= 0 and stamp < ends[i]:
                    launches[rank][i][recipe] += 1
        for node in inventory['nodes']:
            node['_spans'], node['_windows'], node['_descriptors'] = [], set(), 0
            node['_contract'] = node_contract(node, contracts)
        with gzip.open(directory / 'hardware.jsonl.gz', 'rt') as stream:
            for line in stream:
                start, duration, lane, index, _, api = json.loads(line)
                a, b = base + start, base + start + duration
                if b <= starts[0] or a >= ends[-1] or b <= a:
                    continue
                node = inventory['nodes'][index]
                engine, kernel = node.get('engine', 'OTHER'), node.get('kernel', 'unresolved')
                engine = engine if engine in ('TPC', 'MME', 'HCL', 'DMA') else 'OTHER'
                useful = engine in ('TPC', 'MME') and kernel != 'null'
                kind = engine + '_null' if kernel == 'null' and engine in ('TPC', 'MME') else engine
                rid = str(node.get('recipe', '')).split('@')[0]
                recipe_start = bisect_right(recipe_starts[rid], a) - 1
                # Per-node descriptors share a recipe invocation across lanes.
                # Missing recipe starts remain unresolved, never invented calls.
                call = (rid, index, recipe_start) if recipe_start >= 0 else None
                category = classify(node, node["_contract"])
                item = kernels[rank][(engine, kernel, category)]
                node['_spans'].append((max(a, starts[0]), min(b, ends[-1])))
                node['_descriptors'] += 1
                item['descriptors'] += 1
                item['spans'].append((max(a, starts[0]), min(b, ends[-1])))
                if call is not None:
                    item['invocations'].add(call)
                item['contracts'][index] = node['_contract']
                i = max(0, bisect_right(starts, a) - 1)
                while i < len(windows) and a < ends[i]:
                    low, high = max(a, starts[i]), min(b, ends[i])
                    if high > low:
                        node['_windows'].add(i)
                        item['active_nodes'].add((i, index))
                        gap_spans[rank][i][kind].append((low, high))
                        counts[rank][i][kind + '_physical_descriptors'] += 1
                        if useful:
                            counts[rank][i][engine + '_lane_' + lane] += 1
                            semantic_spans[rank][i][category].append((low, high))
                        if call is not None:
                            invocation[rank][i].add(call)
                    i += 1
        sync_pattern = re.compile(r'item|\.cpu|synchroniz|wait.*event|stream.*wait|DtoH|D2H', re.I)
        host_path = directory / 'host.jsonl.gz'
        if host_path.exists():
            with gzip.open(host_path, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    i = bisect_right(starts, base + row[0]) - 1
                    if i >= 0 and base + row[0] < ends[i] and sync_pattern.search(str(row[5])):
                        host_sync[rank][i][str(row[5])] += 1
        for index, node in enumerate(inventory['nodes']):
            if not node['_spans']:
                continue
            matched = re.search(r'layers/(\d+)', node['node'])
            contract = node['_contract']
            node_rows.append(dict(rank=rank, recipe=node['recipe'], kernel=node['kernel'],
                                  layer_in_recipe=int(matched[1]) if matched else None,
                                  node=node['node'],
                                  activity_union_ms_per_round=spans_ms(node['_spans']) / len(windows),
                                  active_windows=len(node['_windows']),
                                  physical_descriptors_per_round=node['_descriptors'] / len(windows),
                                  roi_count=(contract.get('raw_node') or {}).get('num_of_ROIs'),
                                  calls_per_round=None))
        for i, spans in enumerate(gap_spans[rank]):
            for kind, values in spans.items():
                # Lane descriptors overlap heavily. Merge first so sweeps do
                # not sort every physical descriptor four separate times.
                spans[kind] = merge(values)
                for low, high in spans[kind]:
                    events[i][low].append(((rank, kind), 1))
                    events[i][high].append(((rank, kind), -1))
        print(f'rank{rank}: existing hardware/host parsed', flush=True)
    periods = []
    for i, window in enumerate(windows):
        ranks = []
        for rank in range(4):
            allocation = intervals(events[i], starts[i], ends[i], lambda active, card=rank: state(active, card))
            assert abs(sum(allocation.values()) - window['period_ms']) < 1e-6
            spans = gap_spans[rank][i]
            blocks = []
            for a, b, engine in sorted((a, b, engine) for engine, values in spans.items() for a, b in merge(values)):
                if blocks and a <= blocks[-1][1]:
                    if b > blocks[-1][1]:
                        blocks[-1][3] = engine
                    blocks[-1][1] = max(blocks[-1][1], b)
                else:
                    blocks.append([a, b, engine, engine])
            gaps = defaultdict(float)
            for left, right in zip(blocks, blocks[1:]):
                label = ('communication_or_descriptor' if any(
                    name in ('HCL', 'DMA', 'TPC_null', 'MME_null') for name in (left[3], right[2]))
                         else left[3] + '->' + right[2])
                gaps[label] += (right[0] - left[1]) / 1000
            if blocks:
                gaps['before_first_device_activity'] += (blocks[0][0] - starts[i]) / 1000
                gaps['after_last_device_activity'] += (ends[i] - blocks[-1][1]) / 1000
            else:
                gaps['no_activity_window'] += window['period_ms']
            categories = category_allocation(semantic_spans[rank][i], starts[i], ends[i])
            assert abs(sum(categories.values()) - window['period_ms']) < 1e-6
            ranks.append(dict(rank=rank, allocation_ms=allocation,
                              semantic_allocation_ms=categories, no_engine_gap_types_ms=dict(gaps),
                              physical_counts=dict(counts[rank][i]),
                              recorded_device_recipe_starts=sum(launches[rank][i].values()),
                              known_node_invocations=len(invocation[rank][i]),
                              observed_host_sync_calls=dict(host_sync[rank][i])))
        periods.append(dict(position=window['position'], next_position=window['next_position'],
                            period_ms=window['period_ms'], ranks=ranks))
    kernel_rows = []
    for rank, table in enumerate(kernels):
        for (engine, kernel, category), item in table.items():
            nodes = list(item['contracts'].values())
            payload = sum(sum(t.get('bytes', 0) for t in row.get('inputs', []) + row.get('outputs', []))
                          for row in nodes if row) / max(1, sum(bool(row) for row in nodes))
            duration = spans_ms(item['spans'])
            calls = len(item['invocations'])
            kernel_rows.append(dict(rank=rank, engine=engine, kernel=kernel,
                                    category=category,
                                    calls_per_round=calls / len(windows) if calls else None,
                                    activated_recipe_nodes_per_round=len(item['active_nodes']) / len(windows),
                                    physical_descriptors_per_round=item['descriptors'] / len(windows),
                                    activity_union_ms_per_round=duration / len(windows),
                                    mean_call_activity_us=duration * 1000 / calls if calls else None,
                                    nominal_operand_bytes=payload,
                                    nominal_payload_TB_s=(payload * calls / (duration * 1e9)
                                                          if duration and calls else None),
                                    actual_HBM_TB_s=None,
                                    bandwidth_limit=('Operand bytes are not measured HBM traffic; '
                                                     'SRAM/slicing and repeat reads unknown'),
                                    operand_shapes=[dict(inputs=[dict(shape=t['shape'], dtype=t['dtype'],
                                                                     location=t.get('location'))
                                                               for t in row.get('inputs', [])],
                                                         outputs=[dict(shape=t['shape'], dtype=t['dtype'],
                                                                       location=t.get('location'))
                                                                  for t in row.get('outputs', [])])
                                                    for row in nodes[:3] if row]))
    summary = []
    for rank in range(4):
        keys = set().union(*(p['ranks'][rank]['allocation_ms'] for p in periods))
        categories = set().union(*(p['ranks'][rank]['semantic_allocation_ms'] for p in periods))
        summary.append(dict(rank=rank, mean_allocation_ms={key: statistics.mean(
            p['ranks'][rank]['allocation_ms'].get(key, 0) for p in periods) for key in sorted(keys)},
                            mean_semantic_allocation_ms={key: statistics.mean(
                                p['ranks'][rank]['semantic_allocation_ms'].get(key, 0)
                                for p in periods) for key in sorted(categories)},
                            recorded_device_recipe_starts_per_round=statistics.mean(
                                p['ranks'][rank]['recorded_device_recipe_starts'] for p in periods),
                            null_TPC_physical_descriptors_per_round=statistics.mean(
                                p['ranks'][rank]['physical_counts'].get('TPC_null_physical_descriptors', 0)
                                for p in periods)))
    result = dict(schema=3, source_capture=str(args.analysis.resolve()), rounds=len(windows),
                  mean_period_ms=ledger['mean_period_ms'], window_kind=ledger['method'],
                  completion_latency=False, Target_verification_latency=None,
                  actual_recipe_submissions_per_round=None,
                  kernel_grouping="Engine, physical kernel and semantic category; mixed GEMM roles are separate",
                  semantic_allocation_limit=('Exclusive category activity and explicit category overlaps partition '
                                             'each rank window; they are not a dependency critical path. '
                                             'No-useful-compute includes communication and descriptors; '
                                             'see engine allocation for true idle.'),
                  recipe_count_limit="Recorded device recipe starts omit cached native replay invocations",
                  per_round=periods, summary=summary, kernels=kernel_rows, nodes=node_rows, metadata=metadata,
                  host_sync_limit=('Named host API events only; '
                                   'uninstrumented Python .item/.cpu/branches remain unknown'),
                  count_limit=('Null/descriptor counts are hardware records; '
                               'Node calls require recipe starts; missing native markers give unknown calls. '
                               'activated recipe nodes and ROI/descriptors are reported separately'),
                  accuracy_ms=0.5)
    if args.serving_run:
        snapshots = []
        for rank in range(4):
            paths = [args.serving_run / 'traces' / f'rank{rank}-native-profile-{phase}.json'
                     for phase in ('start', 'stop')]
            counts = (native_submission_counts(*(json.loads(path.read_text()) for path in paths))
                      if all(path.is_file() for path in paths) else dict(status='snapshots unavailable'))
            snapshots.append(dict(rank=rank, **counts))
        result['native_submission_counter_scope'] = snapshots
    if args.reference:
        reference = json.loads(args.reference.read_text())
        before = defaultdict(list)
        for parent in reference['kernels']:
            before[(parent['rank'], parent['engine'], parent['kernel'])].append(parent)
        for row in kernel_rows:
            costs = [parent['activity_union_ms_per_round'] for parent in
                     before[(row['rank'], row['engine'], row['kernel'])]]
            row['C6_over_C1_activity_ratio_lower'] = (row['activity_union_ms_per_round'] / sum(costs)
                                                     if sum(costs) else None)
            row['C6_over_C1_activity_ratio_upper'] = (row['activity_union_ms_per_round'] / max(costs)
                                                     if costs and max(costs) else None)
    if args.reference_kernels:
        with gzip.open(args.reference_kernels, 'rt') as stream:
            reference = json.load(stream)
        before = defaultdict(list)
        for row in reference:
            before[(row['rank'], row['engine'], row['kernel'])].append(row)
        for row in kernel_rows:
            parents = before[(row['rank'], row['engine'], row['kernel'])]
            costs = [p['activity_ms_per_token'] for p in parents]
            row['C1_role_activity_lower_ms'] = max(costs) if costs else None
            row['C1_role_activity_upper_ms'] = sum(costs) if costs else None
            row['C6_over_C1_ratio_lower'] = row['activity_union_ms_per_round'] / sum(costs) if sum(costs) else None
            row['C6_over_C1_ratio_upper'] = (row['activity_union_ms_per_round'] / max(costs)
                                                if costs and max(costs) else None)
            row['C1_roles'] = [p['purpose'] for p in parents]
        result['C1_reference'] = str(args.reference_kernels.resolve())
        result['C1_ratio_limit'] = 'Role activity can overlap; bounds only, different captures; screening, not savings'
    (args.output / 'ROUND_ANATOMY.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'KERNELS.csv').open('w') as stream:
        fields = [key for key in kernel_rows[0] if key != 'operand_shapes']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(kernel_rows)
    print(json.dumps(dict(mean_period_ms=result['mean_period_ms'], ranks=summary), indent=2))


if __name__ == '__main__':
    main()
