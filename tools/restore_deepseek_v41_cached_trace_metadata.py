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
from collect_deepseek_v41_trace import recipe_symbols


def physical_nodes(graph):
    return { (engine(node['engine']), int(node['context_id'])): node
             for node in graph['nodes']
             if not node.get('is_logical') and engine(node['engine']) in ('TPC', 'MME', 'DMA') }


def restore(run, analysis, post_base, ranks=None, recipe_cache=None):
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
        root = (recipe_cache / f'rank{rank}' if recipe_cache is not None else
                post_base / str(pid) / str(run / 'graphs' / f'rank{rank}').lstrip('/'))
        candidates = collections.defaultdict(list)
        pattern = '*.recipe_debug_files/graph.post.json' if recipe_cache is not None else 'graph_*_syn_*.post.json'
        cache_proofs = {}
        for cold in root.glob(pattern):
            if f'.{pid}.' in cold.name:
                continue
            data = cold.read_bytes()
            match = re.search(rb'"recipe_debug_id"\s*:\s*(\d+)', data)
            if match is None or int(match[1]) not in needed:
                continue
            if recipe_cache is not None:
                binary = cold.parent.with_name(cold.parent.name.removesuffix('_debug_files'))
                symbol_table = recipe_symbols(binary)
                if symbol_table['recipe_id'] != int(match[1]):
                    raise ValueError(f'Cached binary/graph recipe identity differs: {binary}')
                cache_proofs[str(cold)] = dict(path=str(binary.resolve()), sha256=symbol_table['sha256'])
            for graph in json.loads(data)['graphs']:
                raw_id = graph['recipe_debug_id']
                nodes = physical_nodes(graph)
                # Recipe IDs are short and can collide across the shared warmup
                # cache. A one-TPC decode recipe must not match the first node
                # of an unrelated, much larger prefill recipe with that ID.
                observed_tpc = {pair for pair in needed.get(raw_id, ()) if pair[0] == 'TPC'}
                compiled_tpc = {pair for pair in nodes if pair[0] == 'TPC'}
                if (raw_id in needed and needed[raw_id] <= nodes.keys()
                        and (not observed_tpc or observed_tpc == compiled_tpc)):
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
            destination = output / (f'recipe-{raw_id}.post.json' if recipe_cache is not None else cold.name)
            destination.write_bytes(data)
            record = dict(path=str(destination.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                          format='Synapse cold post-graph JSON', original_path=str(cold), owned_pid=pid,
                          provenance=('cached binary debug table and graph; unique recipeID/engine/context match'
                                      if recipe_cache is not None else
                                      'same-worker cold compiled graph; unique recipeID/engine/context match'))
            if recipe_cache is not None:
                record['cached_binary'] = cache_proofs[str(cold)]
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
                          metadata_provenance=record['provenance'])
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--post-base', type=Path)
    source.add_argument('--recipe-cache', type=Path, help='Exact archived per-rank recipe binaries and debug graphs')
    parser.add_argument('--rank', type=int, action='append')
    args = parser.parse_args()
    restore(args.run.resolve(), args.analysis.resolve(), args.post_base.resolve() if args.post_base else None,
            args.rank, args.recipe_cache.resolve() if args.recipe_cache else None)
