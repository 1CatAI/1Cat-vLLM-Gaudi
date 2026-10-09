# SPDX-License-Identifier: Apache-2.0
"""Audit decoded expert operands in Synapse post-compile allocation graphs.

An SRAM tensor proves placement, not measured traffic or overlap. Eager FX IR
and pre-compile graphs cannot answer this question and are rejected here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit_graph(graph):
    tensors = {tensor["name"]: tensor for tensor in graph["tensors"]}
    consumers = {}
    for node in graph["nodes"]:
        for name in node["input_tensors"]:
            consumers.setdefault(name, []).append(node)
    records = []
    for node in graph["nodes"]:
        guid = node["guid"]
        if not ("expert" in guid and "fp8" in guid and node.get("engine") == "TPC"):
            continue
        for name in node["output_tensors"]:
            tensor = tensors[name]
            if tensor["dtype_bit_size"] != 8:
                continue
            if "allocation" not in tensor:
                raise ValueError("Physical allocation missing; pre-compile/FX graph is insufficient")
            destinations = consumers.get(name, [])
            records.append(dict(producer=node["name"], guid=guid, tensor=name,
                                shape=tensor["max_shape"], allocation=tensor["allocation"],
                                bytes=math.prod(tensor["max_shape"]),
                                persistent=tensor["persistent"], alias_of=tensor.get("alias_of"),
                                producer_bundle=node.get("bundle_index"),
                                consumers=[dict(name=other["name"], guid=other["guid"],
                                                engine=other.get("engine"),
                                                bundle=other.get("bundle_index"))
                                           for other in destinations]))
    return dict(graph=graph["name"], recipe_debug_id=graph.get("recipe_debug_id"),
                decoded_operands=records,
                allocation_bytes={location: sum(r["bytes"] for r in records if r["allocation"] == location)
                                  for location in ("SRAM", "DRAM")},
                all_decoded_operands_in_sram=bool(records) and all(r["allocation"] == "SRAM" for r in records),
                traffic_measured=False,
                warning="Static sliced tensor sizes are not dynamic traffic or invocation counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.directory.rglob("*.post.json")):
        payload = path.read_bytes()
        data = json.loads(payload)
        for graph in data["graphs"]:
            result = audit_graph(graph)
            if result["decoded_operands"]:
                records.append(dict(source=str(path), sha256=hashlib.sha256(payload).hexdigest(), **result))
    args.output.write_text(json.dumps(dict(status="matched" if records else "no_expert_allocation_graph",
                                         graphs=records, qualified_performance=False), indent=2) + "\n")
    print(json.dumps(dict(graphs=len(records), output=str(args.output))))


if __name__ == "__main__":
    main()
