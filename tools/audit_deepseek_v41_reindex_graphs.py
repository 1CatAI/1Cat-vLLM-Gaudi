# SPDX-License-Identifier: Apache-2.0
"""Archive compiled Reindex operands and placement, without inventing counters.

Counts describe compiled graph nodes. Combine with native replay counters to
show which tile programs were submitted; only a device trace measures actual
engine events. Tensor sizes are logical payloads, not hardware bus traffic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from audit_deepseek_v41_sram import tensor_info


def inspect_graph(path):
    matrices, decoded = [], []
    for node in path.read_text().split("\nnode {"):
        name, op = re.search(r'  name: "([^"]+)"', node), re.search(r'  op: "([^"]+)"', node)
        if not name or not op:
            continue
        attrs = dict(re.findall(r'key: "([^"]+)"\s+value \{\s+s: "([^"]*)"', node))
        tensors = {key: tensor_info(value) for key, value in attrs.items() if "Tensor:" in key}
        for value in tensors.values():
            fields = value["description"].split("|")
            value["dtype"] = fields[2].strip() if len(fields) > 2 else None
        record = dict(name=name[1], op=op[1], tensors=tensors)
        if "gemm" in op[1].lower():
            matrices.append(record)
        if "custom_deepseek_v41_index_keys_gaudi2" in op[1]:
            decoded.append(record)
    decoded_outputs = [entry["tensors"]["outputTensor:0"] for entry in decoded]
    names = {value["name"] for value in decoded_outputs}
    consumers = [entry for entry in matrices if entry["tensors"].get("inputTensor:1", {}).get("name") in names]
    return dict(graph=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                compiled_mme_nodes=len(matrices),
                compiled_key_decode_nodes=len(decoded),
                key_decode_logical_bytes=sum(value["bytes"] or 0 for value in decoded_outputs),
                all_key_blocks_in_sram=bool(decoded_outputs) and all(v["location"] == "SRAM" for v in decoded_outputs),
                key_mme_consumers=len(consumers),
                matrices=matrices,
                key_decode=decoded)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    graphs = sorted(args.directory.glob("*PostGraph*"))
    if not graphs:
        raise RuntimeError("No compiled graph evidence")
    result = dict(scope=__doc__,
                  graphs=[inspect_graph(path) for path in graphs],
                  hardware_bandwidth_counters=None,
                  actual_tpc_core_occupancy=None,
                  effective_mme_utilization=None)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for entry in result["graphs"]:
        print(json.dumps({key: value for key, value in entry.items() if key not in ("matrices", "key_decode")}))


if __name__ == "__main__":
    main()
