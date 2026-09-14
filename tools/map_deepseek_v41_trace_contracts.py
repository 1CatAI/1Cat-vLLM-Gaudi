# SPDX-License-Identifier: Apache-2.0
"""Join exact recipe node identities to archived compiler tensor contracts."""

import argparse
import collections
import json
from pathlib import Path
import re


def graph_nodes(path):
    result = []
    for block in re.split(r"\n(?=node \{)", path.read_text()):
        name = re.search(r'^  name: "([^"]+)"', block, re.M)
        op = re.search(r'^  op: "([^"]+)"', block, re.M)
        if name and op:
            attrs = dict(re.findall(r'key: "([^"]+)"\s+value \{\s+s: "([^"\n]+)"', block))
            result.append({
                "name": name[1],
                "op": op[1],
                "attrs": attrs,
                "predecessors": re.findall(r'^  input: "([^"]+)"', block, re.M)
            })
    return result


def tensor(text):
    shape = re.search(r"Sizes = (\[[^]]+\])", text)
    width = re.search(r"sizeInBytes = (\d+)", text)
    location = re.search(r"location = in (\w+)", text)
    strides = re.search(r"strides = (\[[^]]+\])", text)
    alias = re.search(r"isAliased = ([^,|]+)", text)
    fields = text.split("|")
    return {
        "name": fields[0].strip(),
        "shape": json.loads(shape[1]) if shape else None,
        "dtype": fields[2].strip() if len(fields) > 2 else "unknown",
        "bytes": int(width[1]) if width else None,
        "location": location[1] if location else "unknown",
        "strides": json.loads(strides[1]) if strides else None,
        "alias": alias[1].strip() if alias else None,
        "description": text
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.analysis / "graph-manifest.json").read_text())
    graphs, index = {}, collections.defaultdict(set)
    for record in manifest:
        if "PostGraph" not in record["path"]:
            continue
        path = Path(record["path"])
        nodes = graph_nodes(path)
        key = str(path)
        graphs[key] = {"record": record, "nodes": {(node["name"], node["op"]): node for node in nodes}}
        for pair in graphs[key]["nodes"]:
            index[pair].add(key)
    for rank in range(4):
        root = args.analysis / f"rank{rank}"
        recipes = json.loads((root / "recipe-symbols.json").read_text())["recipes"]
        contracts, unresolved = [], []
        for recipe in recipes:
            compute = [node for node in recipe["nodes"] if node["device_type"] in (0, 1, 8)]
            scores = collections.Counter(path for node in compute for path in index[(node["node"], node["kernel"])])
            complete = [path for path, score in scores.items() if score == len(compute)]
            own_rank = [path for path in complete if f"/rank{rank}/" in path]
            if own_rank:
                complete = own_rank
            # Repeated archives may hold the exact same graph. Compare every
            # selected node contract before choosing a representative source.
            graph = None
            if complete:
                ordered = sorted(complete)
                keys = [(node["node"], node["kernel"]) for node in compute]
                signatures = [[graphs[path]["nodes"][key] for key in keys] for path in ordered]
                if all(signature == signatures[0] for signature in signatures):
                    graph = graphs[ordered[0]]
            for symbol in compute:
                node = graph["nodes"].get((symbol["node"], symbol["kernel"])) if graph else None
                entry = {
                    "recipe_id": recipe["recipe_id"],
                    "symbol": symbol,
                    "graph": graph["record"] if graph else None,
                    "matched": node is not None
                }
                if node:
                    attrs = node["attrs"]
                    ordered = sorted(attrs.items(),
                                     key=lambda item: (item[0].split(":")[0], int(item[0].rsplit(":", 1)[1])
                                                       if item[0].startswith(
                                                           ("inputTensor:", "outputTensor:")) else -1))
                    entry.update(inputs=[tensor(value) for key, value in ordered if key.startswith("inputTensor:")],
                                 outputs=[tensor(value) for key, value in ordered if key.startswith("outputTensor:")],
                                 attributes=attrs)
                else:
                    unresolved.append([recipe["recipe_id"], symbol["full_context_id"], symbol["node"]])
                contracts.append(entry)
        (root / "node-contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
        (root / "unresolved-contracts.json").write_text(json.dumps(unresolved, indent=2) + "\n")
        print(json.dumps({
            "rank": rank,
            "compute_nodes": len(contracts),
            "matched": len(contracts) - len(unresolved),
            "unresolved": len(unresolved)
        }),
              flush=True)


if __name__ == "__main__":
    main()
