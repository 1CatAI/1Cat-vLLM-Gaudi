# SPDX-License-Identifier: Apache-2.0
"""Summarise compiler DMA placement for the V4.1 graph dumps.

This is an offline compiler report.  It intentionally reports bytes and
invocation counts, not device time: graph symbols do not contain the runtime
packet timestamps.  Use it to choose a DMA candidate, then join the selected
recipe with a hardware trace before making a latency claim.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

_ATTR = re.compile(r'key: "(inputTensor:\d+|outputTensor:\d+)"\s+value \{\s+s: "([^"]*)"')
_SIZE = re.compile(r"Sizes = (\[[^\]]+\])")
_BYTES = re.compile(r"sizeInBytes = (\d+)")
_DTYPE = re.compile(r"\|\s+([a-z0-9]+)\s+\|")
_LOC = re.compile(r"location = in (\w+)")


def tensor_info(value):
    shape = _SIZE.search(value)
    size = _BYTES.search(value)
    dtype = _DTYPE.search(value)
    location = _LOC.search(value)
    return {
        "shape": json.loads(shape.group(1)) if shape else None,
        "bytes": int(size.group(1)) if size else 0,
        "dtype": dtype.group(1) if dtype else "unknown",
        "location": location.group(1) if location else "unknown",
    }


def iter_nodes(raw):
    for block in raw.split("\nnode {")[1:]:
        name = re.search(r'\n  name: "([^"]+)"', block)
        op = re.search(r'\n  op: "([^"]+)"', block)
        if not name or not op or op.group(1) not in {"DmaMemcpy", "DmaTranspose"}:
            continue
        attrs = {key: tensor_info(value) for key, value in _ATTR.findall(block)}
        source = attrs.get("inputTensor:0", {})
        destination = attrs.get("outputTensor:0", {})
        yield {
            "node": name.group(1),
            "op": op.group(1),
            "src": source.get("location", "unknown"),
            "dst": destination.get("location", "unknown"),
            "dtype": destination.get("dtype", source.get("dtype", "unknown")),
            "bytes": destination.get("bytes", source.get("bytes", 0)),
            "shape": destination.get("shape", source.get("shape")),
        }


def report(graphs):
    groups = defaultdict(lambda: {"count": 0, "bytes": 0, "graphs": set()})
    for path in sorted(graphs.glob("*PostGraph-symbol.pbtxt")):
        for row in iter_nodes(path.read_text(errors="replace")):
            key = (row["op"], row["src"], row["dst"], row["dtype"])
            groups[key]["count"] += 1
            groups[key]["bytes"] += row["bytes"]
            groups[key]["graphs"].add(path.name)
    return [{
        "op": op,
        "src": src,
        "dst": dst,
        "dtype": dtype,
        "count": value["count"],
        "bytes": value["bytes"],
        "graphs": len(value["graphs"])
    } for (op, src, dst, dtype), value in sorted(groups.items(), key=lambda item: item[1]["bytes"], reverse=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = report(args.graphs)
    if not rows:
        raise SystemExit("No compiler DMA nodes found")
    args.output.write_text(json.dumps({"units": "bytes", "rows": rows}, indent=2) + "\n")
    for row in rows[:20]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
