# SPDX-License-Identifier: Apache-2.0
"""Join cached native hardware contexts to the same worker's cold SDK graphs.

Native replay does not publish a recipe name on every profiler event. Preserve
all measured timestamps and node indices, and restore names only when one cold
compiled recipe matches every observed (engine, context) for that recipe ID.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil

from normalize_deepseek_v41_raw_trace import engine, tensor_contract


def physical_nodes(graph):
    return { (engine(node['engine']), int(node['context_id'])): node
             for node in graph['nodes']
             if not node.get('is_logical') and engine(node['engine']) in ('TPC', 'MME', 'DMA') }


def restore(run, analysis, post_base, ranks=None):
    collection_path = analysis / 'collection.json'
    if collection_path.exists():
        collection = json.loads(collection_path.read_text())
    else:
        capture = json.loads((run / 'trace-decode/capture/manifest.json').read_text())
        collection = dict(ranks=[dict(rank=worker['rank'], pid=worker['metadata']['pid'])
                                 for worker in capture['hardware']])
    report = []
    for worker in collection['ranks']:
        rank, pid = worker['rank'], worker['pid']
        if ranks is not None and rank not in ranks:
            continue
        directory = analysis / f'rank{rank}'
        proof = directory / 'cached-native-metadata-proof.json'
        if proof.exists():
            report.append(json.loads(proof.read_text()))
            continue
        path = directory / 'inventory.json'
        inventory = json.loads(path.read_text())
        assert inventory['format'] == 'raw_hltv_cpu' and inventory['hardware_events'] > 0
        needed = collections.defaultdict(set)
        pending = []
        for index, node in enumerate(inventory['nodes']):
            raw_id, _, raw_name = node['recipe'].removesuffix(':').partition('@')
            context = node.get('raw_context_id', '')
            if raw_id.isdigit() and not raw_name and context.isdigit() and node['engine'] in ('TPC', 'MME'):
                pair = (node['engine'], int(context))
                needed[int(raw_id)].add(pair)
                pending.append((index, int(raw_id), pair))
        root = post_base / str(pid) / str(run / 'graphs' / f'rank{rank}').lstrip('/')
        candidates = collections.defaultdict(list)
        for cold in root.glob('graph_*_syn_*.post.json'):
            if f'.{pid}.' in cold.name:
                continue
            data = cold.read_bytes()
            match = re.search(rb'"recipe_debug_id"\s*:\s*(\d+)', data)
            if match is None or int(match[1]) not in needed:
                continue
            for graph in json.loads(data)['graphs']:
                raw_id = graph['recipe_debug_id']
                nodes = physical_nodes(graph)
                if raw_id in needed and needed[raw_id] <= nodes.keys():
                    candidates[raw_id].append((cold, graph, nodes))
        selected = {}
        for raw_id, observed in needed.items():
            matches = candidates[raw_id]
            if len(matches) != 1:
                raise ValueError(f'rank{rank} recipe{raw_id}: {len(matches)} cold matches for {sorted(observed)}')
            selected[raw_id] = matches[0]
        backup = directory / 'before-cached-metadata'
        backup.mkdir(exist_ok=False)
        for name in ('inventory.json', 'recipe-symbols.json', 'node-contracts.json', 'raw-graph-manifest.json'):
            shutil.copy2(directory / name, backup / name)
        recipes_doc = json.loads((directory / 'recipe-symbols.json').read_text())
        contracts = json.loads((directory / 'node-contracts.json').read_text())
        graphs = json.loads((directory / 'raw-graph-manifest.json').read_text())
        output = directory / 'cached-native-graphs'
        output.mkdir(exist_ok=False)
        symbols = {}
        evidence = []
        for raw_id, (cold, graph, nodes) in selected.items():
            data = cold.read_bytes()
            destination = output / cold.name
            destination.write_bytes(data)
            record = dict(path=str(destination.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                          format='Synapse cold post-graph JSON', original_path=str(cold), owned_pid=pid,
                          provenance='same-worker cold compiled graph; unique recipeID/engine/context match')
            graphs.append(record)
            identity = f"{raw_id}@{graph['name']}"
            tensors = {tensor['name']: tensor_contract(tensor) for tensor in graph['tensors']}
            recipe_nodes = []
            for (kind, context), node in nodes.items():
                symbol = dict(device_type={'TPC': 1, 'MME': 0, 'DMA': 8}[kind], context_id=context,
                              full_context_id=context, node=node['name'], kernel=node['guid'],
                              working_engines=node.get('tpc_working_engines', []),
                              roi_count=node.get('num_of_ROIs'), unique_node_id=node['id'])
                symbols[(raw_id, kind, context)] = symbol
                recipe_nodes.append(symbol)
                contracts.append(dict(recipe_id=identity, raw_recipe_id=raw_id, symbol=symbol,
                                      graph=record, matched=True,
                                      inputs=[tensors[name] for name in node['input_tensors'] if name in tensors],
                                      outputs=[tensors[name] for name in node['output_tensors'] if name in tensors],
                                      attributes={}, raw_node=node,
                                      provenance=record['provenance']))
            recipes_doc['recipes'].append(dict(recipe_id=identity, raw_recipe_id=raw_id,
                                              path=record['path'], sha256=record['sha256'], nodes=recipe_nodes))
            evidence.append(dict(recipe_id=raw_id, identity=identity, observed=sorted(needed[raw_id]), graph=record))
        for index, raw_id, (kind, context) in pending:
            node = inventory['nodes'][index]
            symbol = symbols[(raw_id, kind, context)]
            graph = selected[raw_id][1]
            node['cached_original'] = dict(node)
            node.update(recipe=f"{raw_id}@{graph['name']}:", node=symbol['node'], kernel=symbol['kernel'],
                        raw_unique_node_id=str(symbol['unique_node_id']),
                        metadata_provenance='unique same-worker cold recipeID/engine/context match')
        inventory['kernel_counts_before_cached_metadata'] = inventory.pop('kernel_counts', [])
        inventory['cached_native_metadata'] = dict(restored_nodes=len(pending), restored_recipes=len(selected),
                                                   timestamps_unchanged=True, node_indices_unchanged=True)
        path.write_text(json.dumps(inventory, indent=2) + '\n')
        (directory / 'recipe-symbols.json').write_text(json.dumps(recipes_doc, indent=2) + '\n')
        (directory / 'node-contracts.json').write_text(json.dumps(contracts, indent=2) + '\n')
        (directory / 'raw-graph-manifest.json').write_text(json.dumps(graphs, indent=2) + '\n')
        row = dict(rank=rank, pid=pid, restored_nodes=len(pending), restored_recipes=len(selected), evidence=evidence)
        proof.write_text(json.dumps(row, indent=2) + '\n')
        report.append(row)
        print(f'rank{rank}: restored {len(pending)} native nodes from {len(selected)} exact cold recipes', flush=True)
    merged = {row['rank']: row for row in report}
    for proof in analysis.glob('rank*/cached-native-metadata-proof.json'):
        row = json.loads(proof.read_text())
        merged[row['rank']] = row
    (analysis / 'cached-native-metadata-proof.json').write_text(json.dumps(list(merged.values()), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--post-base', type=Path, required=True)
    parser.add_argument('--rank', type=int, action='append')
    args = parser.parse_args()
    restore(args.run.resolve(), args.analysis.resolve(), args.post_base.resolve(), args.rank)
