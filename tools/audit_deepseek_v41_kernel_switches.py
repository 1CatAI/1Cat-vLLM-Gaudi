# SPDX-License-Identifier: Apache-2.0
"""Audit saved decode activities and compiler dependencies without a new capture.

Activity bouts are connected intervals, not recipe calls or hardware lane packets.
Export both quantities explicitly: sliced nodes can have many activity bouts.
"""
import argparse
import ast
import bisect
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re


def merged(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(spans):
    return sum(end-start for start, end in spans)


def recipe_occurrences(frames, slots):
    """Join each device recipe start to its ordered stage slot, never its name.

    A cached recipe can occur in several layers/groups. The trace's recipe ID
    and invocation time choose the occurrence; the stage slot chooses its
    group. Cold template recipe IDs are deliberately not used as trace IDs.
    """
    occurrences = defaultdict(list)
    for token, frame in enumerate(frames):
        if len(frame) != len(slots):
            raise ValueError('Native frame and stage slot count differ')
        for slot, (entry, (group, node)) in enumerate(zip(frame, slots)):
            occurrences[entry[1]].append(dict(start_us=entry[0], token=token, slot=slot,
                                              group=group, stage_node=node['name']))
    for entries in occurrences.values():
        entries.sort(key=lambda entry: entry['start_us'])
        if any(a['start_us'] >= b['start_us'] for a, b in zip(entries, entries[1:])):
            raise ValueError('Recipe starts are not unique and increasing')
    return occurrences


def packet_occurrence(start, end, entries, starts):
    index = bisect.bisect_right(starts, start)-1
    if index < 0:
        return None
    if index+1 < len(entries) and end > starts[index+1]:
        # Do not silently split a hardware execution between two invocations.
        return None
    return entries[index]


def layer_call_audit(root, output, rank, rows, inventory, frames, slots, windows, peer_points):
    """Attribute raw lane packets per invocation without prorating averages.

    One observed physical node execution means a serialized compiler node in
    one recipe occurrence. ROI slices and connected activity bouts are counted
    separately. Neither lane packets nor active-time sums are kernel counts or
    additive critical-path latency.
    """
    occurrences = recipe_occurrences(frames, slots)
    starts = {recipe: [entry['start_us'] for entry in entries]
              for recipe, entries in occurrences.items()}
    keys = {}
    for index, row in enumerate(rows):
        key = row['recipe'], row['engine'], row['source_node']
        if key in keys:
            raise ValueError('Ambiguous serialized physical node identity')
        keys[key] = index
    node_rows = {index: keys.get((node['recipe'].rstrip(':'), node['engine'], node['node']))
                 for index, node in enumerate(inventory['nodes'])}
    spans, packets, ignored = defaultdict(list), Counter(), Counter()
    window_starts = [start for start, _ in windows]
    with gzip.open(root / 'hardware.jsonl.gz', 'rt') as stream:
        for line in stream:
            stamp, length, lane, node_id, *_ = json.loads(line)
            index = node_rows.get(node_id)
            if index is None:
                continue
            row = rows[index]
            if row['engine'] not in ('TPC', 'MME') or row['kernel'] == 'null':
                continue
            wi = bisect.bisect_right(window_starts, stamp)-1
            if wi < 0 or stamp+length > windows[wi][1]:
                ignored['window_boundary_packets'] += 1
                continue
            recipe = row['recipe']
            if recipe not in occurrences:
                ignored['outside_native_stage_packets'] += 1
                continue
            entry = packet_occurrence(stamp, stamp+length, occurrences[recipe], starts[recipe])
            if entry is None:
                ignored['ambiguous_recipe_boundary_packets'] += 1
                continue
            key = entry['token'], entry['slot'], index
            spans[key].append((stamp, stamp+length))
            packets[key] += 1
    reductions = sorted(point['producer'] for point in peer_points if point['bytes'] == 10240)
    if len(reductions) != 81 or reductions[0] != 0:
        raise ValueError('Expected embedding plus two 5120-BF16 reductions per layer')
    # A cached compiler name can retain layer0 even when bound to layer2.
    # Preserve it as a source candidate; attach the runtime reduction context
    # separately. This context does not prove ownership of cross-layer fusions.
    output_producers = reductions[1:]
    totals = defaultdict(lambda: dict(calls=0, active_us=0., bouts=0, packets=0))
    for (token, slot, index), intervals in spans.items():
        key = slot, index
        merged_intervals = merged(intervals)
        total = totals[key]
        total['calls'] += 1
        total['active_us'] += duration(merged_intervals)
        total['bouts'] += len(merged_intervals)
        total['packets'] += packets[token, slot, index]
    result = []
    for (slot, index), total in sorted(totals.items()):
        group = slots[slot][0]
        source = rows[index]
        origins = [source['source_node'], *(node['name'] for node in source['fused_operations'])]
        local = sorted({int(layer) for name in origins for layer in re.findall(r'/layers/(\d+)/', name)})
        if any(layer >= 4 for layer in local):
            raise ValueError('Stage-local layer exceeds four-layer group')
        layers = [4*group+layer for layer in local]
        phase_index = bisect.bisect_left(output_producers, slot)
        next_layer = phase_index//2 if phase_index < len(output_producers) else None
        next_phase = ('attention' if phase_index % 2 == 0 else 'ffn') if next_layer is not None else 'tail'
        result.append(dict(rank=rank, group=group, compute_slot=slot,
                           next_reduction_layer=next_layer, next_reduction_phase=next_phase,
                           cached_source_layer_candidates=layers,
                           ownership='Runtime reduction context; cached source scope alone cannot establish layer ownership',
                           kernel=source['kernel'], engine=source['engine'], category=source['category'],
                           recipe=source['recipe'], source_node=source['source_node'],
                           physical_node_executions_per_token=total['calls']/len(frames),
                           connected_activity_bouts_per_token=total['bouts']/len(frames),
                           active_ms_per_token=total['active_us']/len(frames)/1000,
                           mean_observed_execution_us=total['active_us']/total['calls'],
                           observed_lane_packets=total['packets'],
                           predecessors=source['predecessors'], successors=source['successors'],
                           python_origins=source['python_saved_fx_origins'],
                           native_call_sites=source['python_lexical_calls']))
    (output / f'rank{rank}-layer-calls.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (output / f'rank{rank}-layer-calls.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in result)
    return dict(rows=len(result), source_single_layer_rows=sum(len(row['cached_source_layer_candidates']) == 1 for row in result),
                physical_node_executions_per_token=sum(row['physical_node_executions_per_token'] for row in result),
                connected_activity_bouts_per_token=sum(row['connected_activity_bouts_per_token'] for row in result),
                excluded_packets=dict(ignored),
                source_limit='FX names reused across saved graphs; all ambiguous Python lines remain candidates')


def python_calls(source):
    """Exact lexical custom-op call sites; fused arithmetic remains explicit."""
    calls = defaultdict(list)
    for path in source.rglob('*.py'):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name.startswith('custom_deepseek_'):
                    calls[name].append(dict(file=str(path), line=node.lineno))
    return calls


def fx_sources(directory):
    """Retain saved stacktrace lines, including ambiguity across reused graphs."""
    sources = defaultdict(dict)
    for path in directory.glob('overlap-input-*.py'):
        text = path.read_text()
        # Only the actual small decode root; no prefill provenance borrowing.
        header = text.splitlines()[1] if len(text.splitlines()) > 1 else ''
        if 'bf16[1, 4, 5120]' not in header and 'i32[1]' not in header:
            continue
        current = None
        for line in text.splitlines():
            comment = re.search(r'# File: (.+\.py):(\d+) in (\w+), code: (.*)', line)
            if comment:
                filename, number, function, expression = comment.groups()
                current = dict(file=filename, line=int(number), function=function, expression=expression)
            elif '# No stacktrace found' in line:
                current = None
            assignment = re.match(r'\s+(\w+)(?::[^=]+)?\s*=.*torch\.ops\.', line)
            if assignment and current:
                key = (current['file'], current['line'], current['function'])
                sources[assignment[1]][key] = dict(current, fx_graph=str(path), fx_node=assignment[1])
    return sources


def audit(capture, output, ranks, *, calls_by_layer=False):
    output.mkdir(parents=True, exist_ok=True)
    template = json.loads((capture / 'FULL_STAGE_PLAN_TEMPLATE.json').read_text())
    slots = [(group['group'], node) for group in template for node in group['nodes'] if node['kind'] == 'compute']
    call_sites = python_calls(capture / 'source' / 'vllm_gaudi')
    results = []
    for rank in ranks:
        root = capture / 'analysis-decode' / f'rank{rank}'
        with gzip.open(root / 'activity-intervals.json.gz', 'rt') as stream:
            activity = json.load(stream)
        windows = activity['windows_us']
        scale = len(windows)
        categories = [dict(engine=g['engine'], category=g['category'],
                           bouts_per_token=len(g['intervals_us'])/scale,
                           active_ms_per_token=duration(g['intervals_us'])/scale/1000)
                      for g in activity['groups']]
        compute = merged(span for g in activity['groups'] if g['engine'] in ('TPC', 'MME')
                         and g['category'] != '设备调度' for span in g['intervals_us'])
        bins = defaultdict(lambda: [0, 0.0])
        window_starts = [w[0] for w in windows]
        for left, right in zip(compute, compute[1:]):
            # Do not count a gap spanning two token accounting windows.
            wi = bisect.bisect_right(window_starts, left[1])-1
            if wi < 0 or right[0] > windows[wi][1]:
                continue
            gap = right[0]-left[1]
            limits = (2, 5, 10, 15, 20, 50, 100, 1000)
            k = bisect.bisect_right(limits, gap)
            label = f'{(0, *limits)[k]}–{(*limits, "inf")[k]} us'
            bins[label][0] += 1
            bins[label][1] += gap
        inv = json.loads((root / 'inventory.json').read_text())
        stacktraces = fx_sources(capture / 'plans' / f'rank{rank}')
        nodes = json.loads((root / 'node-breakdown.json').read_text())
        frames = json.loads((root / 'native-invocations.json').read_text())['frames']
        if any(len(frame) != len(slots) for frame in frames):
            raise ValueError('Native compute template does not match the captured frames')
        owners = defaultdict(set)
        calls = Counter()
        for frame in frames:
            for entry, (group, node) in zip(frame, slots):
                owners[entry[1]].add(group)
                calls[entry[1]] += 1
        # Post-graph tensor names establish exact in-recipe data/control dependencies.
        graphs = {}
        rows = []
        for row in nodes:
            contract = row.get('compiler_contract') or {}
            raw = contract.get('raw_node') or {}
            graph_path = (contract.get('graph') or {}).get('path')
            if graph_path and graph_path not in graphs:
                graph = json.loads(Path(graph_path).read_text())['graphs'][0]
                producers, consumers = defaultdict(list), defaultdict(list)
                for n in graph['nodes']:
                    for t in n.get('output_tensors', []) + n.get('output_ctrl_tensors', []):
                        producers[t].append(n['name'])
                    for t in n.get('input_tensors', []) + n.get('input_ctrl_tensors', []):
                        consumers[t].append(n['name'])
                graphs[graph_path] = producers, consumers
            producers, consumers = graphs.get(graph_path, ({}, {}))
            origins = (raw.get('fused_node_graph') or {}).get('nodes', [])
            names = [row['source_node'], *(n['name'] for n in origins)]
            local_layers = sorted({int(m) for name in names for m in re.findall(r'/layers/(\d+)/', name)})
            group_ids = sorted(owners[row['recipe_id']])
            layers = sorted({4*g+l for g in group_ids for l in local_layers})
            inputs = raw.get('input_tensors', []) + raw.get('input_ctrl_tensors', [])
            outputs = raw.get('output_tensors', []) + raw.get('output_ctrl_tensors', [])
            custom_names = {m for name in names for m in re.findall(r'custom_deepseek_\w+?_gaudi2', name)}
            custom_names.add(row['kernel'])
            lexical_calls = [site for name in sorted(custom_names) for site in call_sites.get(name, [])]
            fx_calls = {}
            for name in names:
                # Synapse retains original FX scope components before its GUID.
                for component in name.split('/'):
                    for key, origin in stacktraces.get(component, {}).items():
                        fx_calls[key] = origin
            rows.append(dict(rank=rank, recipe=row['recipe_id'], groups=group_ids,
                             layer_candidates=layers,
                             layer_scope='Reused recipe scopes; global ownership needs call-site join',
                             kernel=row['kernel'], engine=row['engine'], category=row['category'],
                             source_node=row['source_node'], fused_operations=origins,
                             active_ms_per_token=row['activity_ms_per_token'],
                             observed_calls_per_token=row.get('calls_per_token'),
                             native_recipe_calls_per_token=calls[row['recipe_id']]/scale,
                             lane_packets=row['observed_lane_packets'],
                             predecessors=sorted({p for t in inputs for p in producers.get(t, [])}),
                             successors=sorted({c for t in outputs for c in consumers.get(t, [])}),
                             compiler_graph=graph_path,
                             python_lexical_calls=lexical_calls,
                             python_saved_fx_origins=list(fx_calls.values()),
                             python_line_status='Saved FX name/stacktrace join; multiple matches remain explicit'))
        (output / f'rank{rank}-nodes.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
        with (output / f'rank{rank}-nodes.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                              for k, v in row.items()} for row in rows)
        peer_points = json.loads((capture / 'NATIVE_POINT_SKEW.json').read_text())['per_point'] if calls_by_layer else []
        layer_calls = layer_call_audit(root, output, rank, rows, inv, frames, slots, windows, peer_points) if calls_by_layer else None
        # Keep expert overlap distinct from total compute overlap.
        expert = {e: merged(s for g in activity['groups'] if g['category'] == '路由专家' and g['engine'] == e
                            for s in g['intervals_us']) for e in ('TPC', 'MME')}
        overlap = duration(expert['TPC'])+duration(expert['MME'])-duration(merged(expert['TPC']+expert['MME']))
        results.append(dict(rank=rank, tokens=scale, trace_sha256=activity['trace_sha256'], categories=categories,
                            category_bouts_sum_per_token=sum(g['bouts_per_token'] for g in categories
                                                           if g['engine'] in ('TPC', 'MME')
                                                           and g['category'] != '设备调度'),
                            nonnull_compute_bouts_per_token=len(compute)/scale,
                            nonnull_compute_active_ms_per_token=duration(compute)/scale/1000,
                            internal_compute_gaps={k: dict(count_per_token=v[0]/scale, ms_per_token=v[1]/scale/1000)
                                                   for k, v in bins.items()},
                            expert_tpc_ms=duration(expert['TPC'])/scale/1000,
                            expert_mme_ms=duration(expert['MME'])/scale/1000,
                            expert_overlap_ms=overlap/scale/1000,
                            recipe_count_per_token=len(slots),
                            layer_call_audit=layer_calls,
                            source_identity=str(root), base_time_nanoseconds=inv['base_time_nanoseconds']))
    report = dict(capture=str(capture), status='offline audit; no performance gain credited', ranks=results,
                  count_contract='Connected activity bouts exclude null; distinct from kernel invocations and recipes',
                  dependency_contract='Saved post-graph tensor/control edges; peer edges in NATIVE_POINT_SKEW')
    (output / 'SUMMARY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--calls-by-layer', action='store_true', help='Join raw lane packets to native stage occurrences')
    args = parser.parse_args()
    result = audit(args.capture, args.output, args.ranks, calls_by_layer=args.calls_by_layer)
    print(json.dumps([dict(rank=r['rank'], bouts=r['nonnull_compute_bouts_per_token'],
                           active_ms=r['nonnull_compute_active_ms_per_token'], gaps=r['internal_compute_gaps'],
                           expert_overlap_ms=r['expert_overlap_ms']) for r in result['ranks']], ensure_ascii=False))
