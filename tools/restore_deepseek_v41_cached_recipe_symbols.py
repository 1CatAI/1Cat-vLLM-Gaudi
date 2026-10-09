# SPDX-License-Identifier: Apache-2.0
"""Restore names from the acquisition's recipe cache without exporting graphs.

Only an unambiguous recipe ID and complete observed engine/context match is
admitted. Binary debug symbols supply names and dtypes, never tensor placement.
Hardware timestamps, indices and recorded unique-node IDs remain unchanged.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from collect_deepseek_v41_trace import recipe_symbols


def restore(run, rank):
    directory = run / 'analysis' / f'rank{rank}'
    proof = directory / 'cached-binary-symbol-proof.json'
    if proof.exists():
        raise FileExistsError('Reuse the saved symbol restoration')
    inventory = json.loads((directory / 'inventory.json').read_text())
    needed = defaultdict(set)
    for node in inventory['nodes']:
        rid, _, name = node['recipe'].rstrip(':').partition('@')
        if rid.isdigit() and not name and node['engine'] in ('TPC', 'MME'):
            needed[int(rid)].add((dict(TPC=1, MME=0)[node['engine']], int(node['raw_context_id'])))
    cache = (run / 'recipe_cache' / f'rank{rank}').resolve(strict=True)
    candidates, failures = defaultdict(list), []
    for binary in cache.rglob('*.recipe'):
        try:
            row = recipe_symbols(binary)
        except ValueError as error:
            failures.append(dict(path=str(binary), error=str(error)))
            continue
        rid = row['recipe_id']
        if rid not in needed:
            continue
        symbols = {(s['device_type'], s['full_context_id']): s for s in row['nodes']}
        if needed[rid] <= symbols.keys():
            candidates[rid].append((row, symbols))
    selected, unresolved = {}, []
    for rid, observed in needed.items():
        matches = candidates[rid]
        # Byte-identical duplicate cache entries describe the same recipe.
        matches = list({row['sha256']: (row, symbols) for row, symbols in matches}.values())
        if len(matches) > 1:
            matches = [(row, symbols) for row, symbols in matches
                       if observed == {key for key in symbols if key[0] in (0, 1)}]
        if len(matches) == 1:
            selected[rid] = matches[0]
        else:
            unresolved.append(dict(recipe_id=rid, matching_binaries=len(matches)))
    backup = directory / 'before-cached-binary-symbols'
    backup.mkdir()
    names = ('inventory.json', 'recipe-symbols.json', 'node-contracts.json')
    for name in names:
        shutil.copy2(directory / name, backup / name)
    recipes = json.loads((directory / 'recipe-symbols.json').read_text())
    contracts = json.loads((directory / 'node-contracts.json').read_text())
    evidence = []
    for rid, (row, symbols) in selected.items():
        identity = f'{rid}@cached-binary-debug-{rid}'
        recipes['recipes'].append({**row, 'recipe_id': identity, 'raw_recipe_id': rid})
        evidence.append(dict(recipe_id=rid, path=row['path'], sha256=row['sha256'],
                             observed_contexts=sorted(needed[rid])))
        for symbol in symbols.values():
            contracts.append(dict(recipe_id=identity, raw_recipe_id=rid, symbol=symbol,
                                  matched=False, graph=None, raw_node=None, inputs=[], outputs=[], attributes={},
                                  provenance='Exact cached binary debug symbols; shapes/placement unobserved'))
    restored = 0
    for node in inventory['nodes']:
        rid, _, name = node['recipe'].rstrip(':').partition('@')
        if not rid.isdigit() or name or int(rid) not in selected or node['engine'] not in ('TPC', 'MME'):
            continue
        symbol = selected[int(rid)][1][dict(TPC=1, MME=0)[node['engine']], int(node['raw_context_id'])]
        node['before_cached_binary_symbols'] = dict(node)
        node.update(recipe=f'{rid}@cached-binary-debug-{rid}:', node=symbol['node'], kernel=symbol['kernel'],
                    reported_dtype=symbol['dtype'], metadata_provenance='Exact acquisition cache binary debug symbols')
        restored += 1
    (directory / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (directory / 'recipe-symbols.json').write_text(json.dumps(recipes, indent=2) + '\n')
    (directory / 'node-contracts.json').write_text(json.dumps(contracts, indent=2) + '\n')
    result = dict(rank=rank, cache=str(cache), restored_recipes=len(selected), restored_nodes=restored,
                  evidence=evidence, unresolved=unresolved, parser_failures=failures,
                  timestamps_unchanged=True, hardware_indices_unchanged=True, shapes_placement_unobserved=True)
    proof.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    restore(args.run.resolve(), args.rank)
