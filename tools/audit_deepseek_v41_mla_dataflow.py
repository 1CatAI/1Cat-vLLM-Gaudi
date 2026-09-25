# SPDX-License-Identifier: Apache-2.0
"""Inspect existing compiled MLA preparation and its first MME consumers."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re

from audit_deepseek_v41_sram import tensor_info


def audit(path):
    raw = path.read_text()
    prepare = {
        "custom_deepseek_v41_selected_kv_vector_scales_bf16_gaudi2",
        "custom_deepseek_v41_selected_mla_gather_gaudi2",
        "custom_deepseek_v41_selected_packed_mla_gather_gaudi2",
        "custom_deepseek_v41_selected_packed_mla_vector_gaudi2",
    }
    if not any(f'op: "{name}"' in raw for name in prepare):
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

    producers, matrices = [], set()
    for name, node in nodes.items():
        if node["op"] not in prepare:
            continue
        pending, seen, consumers = deque(users[name]), set(), []
        while pending:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            consumer = nodes[current]
            if "gemm" in consumer["op"].lower():
                consumers.append({
                    "name": current,
                    "guid": consumer["op"],
                    "inputs": [tensor_info(consumer["attrs"].get(f"inputTensor:{i}", "")) for i in (0, 1)],
                    "output": tensor_info(consumer["attrs"].get("outputTensor:0", "")),
                })
                matrices.add(current)
            else:
                pending.extend(users[current])
        ancestors, pending = set(), deque(node["inputs"])
        while pending:
            current = pending.popleft()
            if current in ancestors or current not in nodes:
                continue
            ancestors.add(current)
            pending.extend(nodes[current]["inputs"])
        writes = [{
            "name": key,
            "guid": nodes[key]["op"]
        } for key in sorted(ancestors) if "state_rows_write" in nodes[key]["op"]]
        outputs = [
            tensor_info(value) for key, value in sorted(node["attrs"].items()) if key.startswith("outputTensor:")
        ]
        producers.append({
            "name": name,
            "guid": node["op"],
            "outputs": outputs,
            "execution_index": node["attrs"].get("Exec_idx"),
            "ancestor_state_writes": writes,
            "first_mme_consumers": consumers,
        })
    counts = Counter(p["guid"] for p in producers)
    return {
        "graph": str(path),
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "preparation_nodes": dict(counts),
        "preparation_output_bytes": sum(t["bytes"] or 0 for p in producers for t in p["outputs"]),
        "preparation_locations": dict(Counter(t["location"] for p in producers for t in p["outputs"])),
        "first_mme_nodes": len(matrices),
        "unresolved_producers": sum(not p["first_mme_consumers"] for p in producers),
        "producers": producers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = [item for path in sorted(args.graphs.rglob("*PostGraph*")) if (item := audit(path))]
    if not results:
        raise RuntimeError("No compiled selected MLA producer graph found")
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(json.dumps({key: value for key, value in item.items() if key != "producers"}))


if __name__ == "__main__":
    main()
