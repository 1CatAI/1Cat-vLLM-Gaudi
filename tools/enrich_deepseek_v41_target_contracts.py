# SPDX-License-Identifier: Apache-2.0
"""Recover fused expressions from this acquisition's saved pre/post graphs.

Boundary traversal adapted from archived decompose_23ms_kernels.py (20260910).
No old timing, shape, layer count or expression hash is transferred.
"""
import argparse
import collections
import json
from pathlib import Path
import re
from map_deepseek_v41_trace_contracts import graph_nodes, tensor


def io(node, prefix):
    return [
        tensor(value)
        for key, value in sorted(node['attrs'].items(),
                                 key=lambda x: int(x[0].rsplit(':', 1)[1]) if x[0].startswith(prefix) else -1)
        if key.startswith(prefix)
    ]


def base_name(name):
    name = name.split('_slice_')[0]
    return re.sub(r'^(\d+)_\d+$', r'\1', name)


def load_graph(path):
    nodes = graph_nodes(path)
    alias = {}
    producers = {}
    for n in nodes:
        for t in io(n, 'inputTensor:') + io(n, 'outputTensor:'):
            if t['alias']:
                alias[t['name']] = t['alias']
        for t in io(n, 'outputTensor:'):
            producers[t['name']] = n
    pre = path.with_name(path.name.replace('-PostGraph-', '-PreGraph-'))
    original = graph_nodes(pre) if pre.exists() else []
    consumers = collections.defaultdict(list)
    for n in original:
        for t in io(n, 'inputTensor:'):
            consumers[t['name']].append(n)
    engram = set()
    pending = []
    for n in original:
        if any(t['shape'] == [25600, 6144] for t in io(n, 'inputTensor:')):
            engram.add(n['name'])
            pending.extend(t['name'] for t in io(n, 'outputTensor:'))
    visited = set()
    while pending:
        tname = pending.pop()
        if tname in visited:
            continue
        visited.add(tname)
        for n in consumers[tname]:
            outputs = io(n, 'outputTensor:')
            # Stop at the restored Engram residual boundary. Later mHC and
            # attention consumers are separate functions.
            engram.add(n['name'])
            if n['op'] == 'cast_f32_to_bf16' and any(t['shape'][-2:] == [4, 5120] for t in outputs):
                continue
            if n['op'].startswith('linear_fwd'):
                continue
            pending.extend(t['name'] for t in outputs)
    return dict(nodes=nodes,
                alias=alias,
                producers=producers,
                pre=original,
                engram=engram,
                pre_producers={t['name']: n
                               for n in original
                               for t in io(n, 'outputTensor:')})


def roots(g, name):
    result = set()
    while name not in result:
        result.add(name)
        short = base_name(name)
        result.add(short)
        if name in g['alias']:
            name = g['alias'][name]
        elif short != name:
            name = short
        else:
            break
    return result


def expression(g, c):
    inputs = set().union(*(roots(g, t['name']) for t in c['inputs']))
    pending = []
    for t in c['outputs']:
        pending.extend(x for x in roots(g, t['name']) if x in g['pre_producers'] and x not in inputs)
    found = {}
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited or name in inputs:
            continue
        visited.add(name)
        n = g['pre_producers'].get(name)
        if n is None:
            continue
        if (n['op'] in ('GEMM', 'BatchGemm') or n['op'].startswith(('custom_', 'linear_fwd', 'batch_gemm'))):
            return [], 'unresolved_boundary'
        found[n['name']] = n
        for t in io(n, 'inputTensor:'):
            if not roots(g, t['name']) & inputs:
                pending.append(t['name'])
    ns = sorted(found.values(), key=lambda n: int(n['attrs'].get('Exec_idx', '0')))
    return ns, 'pregraph_boundary_slice' if ns else 'unresolved_output'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metadata', type=Path)
    args = p.parse_args()
    cache = {}
    for rank in range(4):
        path = args.metadata / f'rank{rank}/node-contracts.json'
        data = json.loads(path.read_text())
        counts = collections.Counter()
        for c in data:
            if not c['matched']:
                continue
            gp = Path(c['graph']['path'])
            if gp not in cache:
                cache[gp] = load_graph(gp)
            graph = cache[gp]
            if c['symbol']['kernel'].startswith('fused_kernel'):
                ns, method = expression(graph, c)
                c['fused_original_nodes'] = [
                    dict(name=n['name'], op=n['op'], semantic_group='Engram' if n['name'] in graph['engram'] else None)
                    for n in ns
                ]
                c['origin_method'] = method
                counts[method] += 1
            # Matrix inputs retain producer identity for W13/W2 and mixed chains.
            c['input_producers'] = [graph['producers'].get(t['name'], {}).get('name') for t in c['inputs']]
        (path.parent / 'enriched-contracts.json').write_text(json.dumps(data, indent=2) + '\n')
        print(rank, dict(counts), flush=True)


if __name__ == '__main__':
    main()
