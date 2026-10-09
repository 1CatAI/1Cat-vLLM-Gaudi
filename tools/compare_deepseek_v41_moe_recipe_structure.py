# SPDX-License-Identifier: Apache-2.0
"""Compare saved compound recipes without acquiring a device or profiler trace."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from collect_deepseek_v41_trace import recipe_symbols


def compare(cache, candidate, rank=0):
    result = defaultdict(list)
    for path in sorted(cache.rglob("*.recipe")):
        if path.parent.name != f"rank{rank}":
            continue
        symbols = recipe_symbols(path)
        nodes = symbols.get("nodes", [])
        is_candidate = any(candidate in node.get("node", "") for node in nodes)
        is_parent = any("moe_token_wide_sat_fp8_gaudi2" in node.get("node", "") for node in nodes)
        if not (is_candidate or is_parent):
            continue
        layers = defaultdict(Counter)
        engines = defaultdict(Counter)
        for node in nodes:
            name = node.get("node", "")
            match = re.search(r"/layers/(\d+)/moe/", name)
            if not match:
                continue
            layer = match[1]
            layers[layer][node.get("kernel", "unknown")] += 1
            engines[layer].update(node.get("working_engines", []))
        result["candidate" if is_candidate else "parent"].append({
            "recipe": str(path), "source_sha256": symbols.get("sha256"),
            "nodes": len(nodes), "moe_kernels_by_relative_layer": dict(layers),
            "roi_engine_counts_by_relative_layer": {layer: dict(counts) for layer, counts in engines.items()},
            "DMA_nodes": dict(Counter(node.get("kernel") for node in nodes
                                      if node.get("kernel") in ("DmaMemcpy", "DmaTranspose", "memcpy_dma"))),
        })
    return dict(rank=rank, recipes=dict(result),
                limitation="Static sliced-node and ROI counts; not invocation timing, K/N slice geometry, "
                           "physical tensor placement or measured overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    result = compare(args.cache, args.candidate, args.rank)
    if not result["recipes"].get("parent") or not result["recipes"].get("candidate"):
        raise ValueError("Both actual captured arms are required")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: len(rows) for key, rows in result["recipes"].items()}))


if __name__ == "__main__":
    main()
