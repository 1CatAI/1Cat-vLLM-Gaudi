# SPDX-License-Identifier: Apache-2.0
"""Count final compiler execution nodes and retain their dependencies.

Bridge PreGraph/Python call counts are intentionally excluded. Logical views,
placeholders and graph outputs do not launch device work. Null descriptors
are not in the symbol graph and must be counted in the combined service trace.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

LOGICAL = {'Placeholder', 'OutputTensor', 'Reshape', 'StaticReshape', 'Slice', 'Split',
           'Concatenate', 'Squeeze', 'ExpandDims', 'Identity', 'Flatten', 'Broadcast',
           'LogicalTranspose', 'LogicalBroadcast', 'ReinterpretCast',
           'TransposedShape', 'Reduction'}


def logical_stages(nodes):
    """Collapse only compiler fragments with the same explicit source node.

    Never merge by GUID or bundle alone: two independent GEMMs/quantizers can
    share either. Source identity before `_bundle_N/op_M` distinguishes them.
    Compiler-created nodes without that provenance remain independent.
    """
    groups = {}
    for node in nodes:
        source = re.sub(r'_bundle_\d+/op_\d+.*$', '', node['name'])
        key = (source, node['op'])
        group = groups.setdefault(key, dict(source=source, operation=node['op'],
                                            fragments=[], execution_indices=[]))
        group['fragments'].append(node['name'])
        group['execution_indices'].append(node['execution_index'])
    stages = list(groups.values())
    return dict(logical_stage_count=len(stages), stages=stages,
                pipeline_fragment_excess=len(nodes)-len(stages),
                logical_count_scope='same explicit compiler source node only; not GUID/bundle deduplication')


def audit(path):
    raw = path.read_text()
    nodes = []
    ignored = Counter()
    for block in raw.split('node {'):
        name, op = re.search(r'\n  name: "([^"]+)"', block), re.search(r'\n  op: "([^"]+)"', block)
        if not name or not op:
            continue
        if op[1] in LOGICAL:
            ignored[op[1]] += 1
            continue
        attrs = dict(re.findall(r'key: "([^"]+)"\s+value \{\s+s: "([^"]*)"', block))
        nodes.append(dict(name=name[1], op=op[1], inputs=re.findall(r'\n  input: "([^"]+)"', block),
                          execution_index=attrs.get('Exec_idx'), bundle=attrs.get('Bundle_idx'),
                          tensors={k:v for k,v in attrs.items() if k.startswith(('inputTensor:', 'outputTensor:'))}))
    return dict(graph=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                physical_nodes=len(nodes), operations=dict(Counter(n['op'] for n in nodes)),
                **logical_stages(nodes),
                logical_nodes_excluded=dict(ignored), nodes=nodes,
                count_scope='post-compiler TPC/MME/DMA nodes; excludes logical views and null descriptors')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graphs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = [audit(p) for p in sorted(args.graphs.rglob('*PostGraph-symbol.pbtxt'))]
    if not results:
        raise RuntimeError('No final compiler symbol graphs; physical count cannot be inferred from recipes')
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    for r in results:
        print(json.dumps({k:v for k,v in r.items() if k != 'nodes'}))


if __name__ == '__main__':
    main()
