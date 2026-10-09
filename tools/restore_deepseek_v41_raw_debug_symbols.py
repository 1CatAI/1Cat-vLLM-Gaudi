# SPDX-License-Identifier: Apache-2.0
"""Recover raw SDK debug symbols, then join unambiguous same-launch graphs.

The raw debug binary differs from the serialized recipe debug table: strings
exclude NUL and the device kind follows the ROI list. Recipe IDs can collide;
context, node name and GUID must all match before using a tensor contract.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

from normalize_deepseek_v41_raw_trace import tensor_contract


def debug_symbols(path):
    data = path.read_bytes()
    major, minor, recipe, count = struct.unpack_from("<IIHI", data)
    assert (major, minor) == (1, 2) and 0 < count < 10000
    offset, nodes = 14, []
    for _ in range(count):
        context, full_context, descriptors, blob = struct.unpack_from("<HIII", data, offset)
        offset += 14
        strings = []
        for _ in range(3):
            length, = struct.unpack_from("<I", data, offset)
            offset += 4
            assert 0 < length < 16384
            strings.append(data[offset:offset + length].decode("utf-8"))
            offset += length
        roi_count, = struct.unpack_from("<H", data, offset)
        offset += 2
        engines = list(data[offset:offset + roi_count])
        offset += roi_count
        device, = struct.unpack_from("<I", data, offset)
        offset += 4
        assert device in (0, 1, 8) and all(n <= 64 for n in engines)
        nodes.append(dict(device_type=device, context_id=context, full_context_id=full_context,
                          descriptors=descriptors, kernel_blob_index=blob, node=strings[0], kernel=strings[1],
                          dtype=strings[2], working_engines=engines))
    assert offset == len(data)
    for device in (0, 1, 8):
        indices = [node["full_context_id"] for node in nodes if node["device_type"] == device]
        assert sorted(indices) == list(range(len(indices)))
    return dict(recipe_id=recipe, nodes=nodes, path=str(path), sha256=hashlib.sha256(data).hexdigest())


def restore(run, rank, allow_unresolved=False):
    path = run / f"analysis/rank{rank}"
    if (path / "raw-debug-symbol-proof.json").exists():
        raise ValueError("Raw symbols already restored; reuse the saved result")
    inventory = json.loads((path / "inventory.json").read_text())
    needed = collections.defaultdict(set)
    for node in inventory["nodes"]:
        rid, _, name = node["recipe"].rstrip(":").partition("@")
        if rid.isdigit() and int(rid) and not name and node["engine"] in ("TPC", "MME"):
            needed[int(rid)].add(({"TPC": 1, "MME": 0, "DMA": 8}[node["engine"]], int(node["raw_context_id"])))
    debug = {}
    for binary in (path / "raw-input/debug-info").glob("*_dbg.bin"):
        row = debug_symbols(binary)
        if row["recipe_id"] in needed:
            assert row["recipe_id"] not in debug
            debug[row["recipe_id"]] = row
    roots = run / f"graphs/rank{rank}/sdk-post-graphs"
    nested = roots / str(run / f"graphs/rank{rank}").lstrip("/")
    metadata_path = run / f"analysis/rank{rank}-metadata.json"
    if metadata_path.is_file():
        pid = json.loads(metadata_path.read_text())["pid"]
    else:
        collection = json.loads((run / "analysis/collection.json").read_text())
        pid = next(row["pid"] for row in collection["ranks"] if row["rank"] == rank)
    candidates = collections.defaultdict(list)
    files = ([Path(e.path) for e in os.scandir(nested)
              if e.name.startswith("graph_") and e.name.endswith(".post.json") and f".{pid}." not in e.name]
             if nested.is_dir() else [])
    files.extend(roots.glob("*.post.json"))
    for cold in files:
        data = cold.read_bytes()
        for graph in json.loads(data)["graphs"]:
            rid = graph.get("recipe_debug_id")
            if rid not in needed:
                continue
            physical = {(dict(TPC=1, MME=0, DMA=8)[node["engine"]], int(node["context_id"])): node
                        for node in graph["nodes"] if node["engine"] in ("TPC", "MME", "DMA")
                        and not node.get("is_logical")}
            if not needed[rid] <= physical.keys():
                continue
            if rid in debug:
                symbols = {(s["device_type"], s["full_context_id"]): s for s in debug[rid]["nodes"]}
                if any(physical[key]["name"] != symbols[key]["node"]
                       or physical[key]["guid"].lower() != symbols[key]["kernel"].lower() for key in needed[rid]):
                    continue
            candidates[rid].append((cold, graph, physical, hashlib.sha256(data).hexdigest()))
    backup = path / "before-raw-debug-symbols"
    backup.mkdir()
    for name in ("inventory.json", "recipe-symbols.json", "node-contracts.json", "raw-graph-manifest.json"):
        shutil.copy2(path / name, backup / name)
    # Cache restoration may already have resolved other recipes. Retain their
    # contracts and provenance when recovering the remaining SDK symbols.
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    contracts = json.loads((path / "node-contracts.json").read_text())
    graphs = json.loads((path / "raw-graph-manifest.json").read_text())
    symbol_map, records, unresolved = {}, [], []
    for rid in sorted(needed):
        matches = candidates[rid]
        assert len(matches) <= 1, (rid, [str(row[0]) for row in matches])
        cold, graph, physical, digest = matches[0] if matches else (None, None, {}, None)
        if rid not in debug and graph is None and allow_unresolved:
            unresolved.append(rid)
            continue
        if rid not in debug:
            assert graph is not None, (rid, "neither raw debug nor unambiguous cold graph")
            debug_row = dict(recipe_id=rid, path=str(cold), sha256=digest,
                             nodes=[dict(device_type=device, context_id=context, full_context_id=context,
                                         node=node["name"], kernel=node["guid"],
                                         working_engines=node.get("tpc_working_engines", []))
                                    for (device, context), node in physical.items()])
        else:
            debug_row = debug[rid]
        identity = f"{rid}@{graph['name'] if graph else 'SDK-raw-debug-' + str(rid)}"
        tensors = {t["name"]: tensor_contract(t) for t in graph["tensors"]} if graph else {}
        record = dict(path=str(cold), sha256=digest, format="same-launch Synapse post-graph JSON",
                      provenance="raw debug recipe/context/node/GUID match") if graph else None
        if record:
            graphs.append(record)
        recipes.append({**debug_row, "recipe_id": identity, "raw_recipe_id": rid})
        for symbol in debug_row["nodes"]:
            key = (symbol["device_type"], symbol["full_context_id"])
            symbol_map[(rid, *key)] = (symbol, identity)
            node = physical.get(key)
            contracts.append(dict(recipe_id=identity, symbol=symbol, matched=node is not None, graph=record,
                                  inputs=[tensors[n] for n in node["input_tensors"]] if node else [],
                                  outputs=[tensors[n] for n in node["output_tensors"]] if node else [],
                                  raw_node=node, attributes={},
                                  provenance="raw SDK debug symbols; tensor placement unobserved" if not node
                                  else record["provenance"]))
        records.append(dict(raw_recipe_id=rid, debug=debug_row["path"], debug_sha256=debug_row["sha256"],
                            observed_contexts=len(needed[rid]), matched_graph=record))
    for node in inventory["nodes"]:
        rid, _, name = node["recipe"].rstrip(":").partition("@")
        if not rid.isdigit() or not int(rid) or name or node["engine"] not in ("TPC", "MME", "DMA"):
            continue
        key = (int(rid), dict(TPC=1, MME=0, DMA=8)[node["engine"]], int(node["raw_context_id"]))
        if key not in symbol_map:
            if allow_unresolved:
                continue
            assert node["engine"] == "DMA", key
            continue
        symbol, identity = symbol_map[key]
        node["before_raw_debug"] = dict(node)
        node.update(recipe=identity + ":", node=symbol["node"], kernel=symbol["kernel"],
                    raw_unique_node_id="", metadata_provenance="same-acquisition SDK raw debug binary")
    (path / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (path / "recipe-symbols.json").write_text(json.dumps(dict(recipes=recipes), indent=2) + "\n")
    (path / "node-contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
    (path / "raw-graph-manifest.json").write_text(json.dumps(graphs, indent=2) + "\n")
    result = dict(rank=rank, raw_debug_recipes=len(debug), matched_graphs=len(graphs), records=records,
                  unresolved_raw_recipes=unresolved,
                  timestamps_unchanged=True, hardware_node_indices_unchanged=True)
    (path / "raw-debug-symbol-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--allow-unresolved", action="store_true",
                        help="Restore only current-acquisition SDK symbols; leave absent metadata explicitly unknown")
    args = parser.parse_args()
    restore(args.run.resolve(), args.rank, args.allow_unresolved)
