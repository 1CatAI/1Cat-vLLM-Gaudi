# SPDX-License-Identifier: Apache-2.0
"""Follow compiled packed-key producers to their first physical MME consumers."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re

from audit_deepseek_v41_sram import tensor_info


def audit(path):
    raw = path.read_text()
    key_ops = {"custom_deepseek_v41_index_keys_gaudi2", "custom_deepseek_v41_index_keys_tiled_gaudi2"}
    if not any(f'op: "{name}"' in raw for name in key_ops):
        return None
    nodes, users = {}, defaultdict(list)
    for block in raw.split("\nnode {"):
        name, op = re.search(r'  name: "([^"]+)"', block), re.search(r'  op: "([^"]+)"', block)
        if not name or not op:
            continue
        attrs = dict(re.findall(r'key: "([^"]+)"\s+value \{\s+s: "([^"]*)"', block))
        inputs = re.findall(r'  input: "([^"]+)"', block)
        nodes[name[1]] = {"name": name[1], "op": op[1], "attrs": attrs, "inputs": inputs}
        for source in inputs:
            users[source].append(name[1])
    producers = []
    all_mmes = set()
    for name, node in nodes.items():
        if node["op"] not in key_ops:
            continue
        pending, seen, matrices = deque(users[name]), set(), []
        while pending:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            consumer = nodes[current]
            if "gemm" in consumer["op"].lower():
                attrs = consumer["attrs"]
                matrices.append({
                    "name": current,
                    "op": consumer["op"],
                    "inputs": [tensor_info(attrs.get(f"inputTensor:{i}", "")) for i in (0, 1)],
                    "output": tensor_info(attrs.get("outputTensor:0", ""))
                })
                all_mmes.add(current)
            else:
                pending.extend(users[current])
        ancestors, pending = set(), deque(node["inputs"])
        while pending:
            current = pending.popleft()
            if current in ancestors or current not in nodes:
                continue
            ancestors.add(current)
            pending.extend(nodes[current]["inputs"])
        source = tensor_info(node["attrs"].get("inputTensor:0", ""))["name"]
        writes = [{
            "name": other_name,
            "execution_index": other["attrs"].get("Exec_idx"),
            "dependency_reaches_key_gather": other_name in ancestors
        } for other_name, other in nodes.items() if other["op"] == "custom_deepseek_v41_state_rows_write_u8_gaudi2"
                  and tensor_info(other["attrs"].get("inputTensor:0", ""))["name"] == source]
        producers.append({
            "name": name,
            "guid": node["op"],
            "output": tensor_info(node["attrs"].get("outputTensor:0", "")),
            "execution_index": node["attrs"].get("Exec_idx"),
            "same_graph_cache_writes": writes,
            "first_mme_consumers": matrices
        })
    return {
        "graph":
        str(path),
        "sha256":
        hashlib.sha256(raw.encode()).hexdigest(),
        "key_producers":
        producers,
        "first_mme_nodes":
        len(all_mmes),
        "unresolved_producers":
        sum(not p["first_mme_consumers"] for p in producers),
        "key_output_bytes":
        sum(p["output"]["bytes"] or 0 for p in producers),
        "key_locations":
        dict((location, sum(p["output"]["location"] == location for p in producers))
             for location in {p["output"]["location"]
                              for p in producers})
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = [item for path in sorted(args.graphs.rglob("*PostGraph*")) if (item := audit(path))]
    if not results:
        raise RuntimeError("No compiled index-key producer graph found")
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(json.dumps({key: value for key, value in item.items() if key != "key_producers"}))


if __name__ == "__main__":
    main()
