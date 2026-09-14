# SPDX-License-Identifier: Apache-2.0
"""Retain compact, named hardware intervals without loading full trace JSON."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

import ijson

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
open_trace = gzip.open if args.trace.suffix == ".gz" else open
node_ids, nodes = {}, []
hw_ids, hw_names = {}, []
kernel_counts = collections.Counter()
recipe_names = collections.defaultdict(set)
host_enqueues, markers, modules = [], [], set()
metadata, examples = [], []
total = hardware = 0
first, last = float("inf"), float("-inf")
with open_trace(args.trace, "rt") as stream:
    header = stream.read(1024)
base = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
with (open_trace(args.trace, "rb") as src, gzip.open(args.output / "hardware.jsonl.gz", "wt", compresslevel=1) as
      dst, gzip.open(args.output / "host.jsonl.gz", "wt", compresslevel=1) as host):
    for event in ijson.items(src, "traceEvents.item", use_float=True):
        total += 1
        name = event.get("name", "")
        # Hardware exports contain metadata/flow events with a null args
        # field.  Treat those as an empty argument map; they still count in
        # the total event inventory but cannot carry recipe or engine data.
        a = event.get("args") or {}
        rid, handle = str(a.get("recipeId", "")), str(a.get("recipeHandle", ""))
        rkey = rid + ":" + handle
        if a.get("Module id"):
            modules.add(a["Module id"])
        if a.get("recipeName"):
            recipe_names[rkey].add(a["recipeName"])
        if event.get("ph") == "M":
            metadata.append(event)
        if "enqueue" in name.lower() and a.get("recipeName"):
            host_enqueues.append([event["ts"], event.get("dur", 0), rkey, a["recipeName"]])
        if event.get("cat") == "cpu_op" and ("Compiled Region" in name or "execute_model" in name
                                             or "prepared_moe" in name or "native_decoder" in name):
            markers.append([event["ts"], event.get("dur", 0), name])
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        hw = a.get("HW event name", "").upper()
        if not hw and event.get("cat") in ("cpu_op", "hpu_op", "user_annotation", "privateuse1_runtime"):
            host.write(
                json.dumps(
                    [event['ts'], event['dur'],
                     str(event.get('pid')),
                     str(event.get('tid')),
                     event.get('cat'), name],
                    separators=(',', ':')) + '\n')
        engine = next((x for x in ("TPC", "MME", "DMA", "NIC") if x in hw), None)
        if engine is None and re.match(r"STM_[01]_(RX|TX|QPC|QMAN)", hw):
            engine = "NIC"
        if engine is None:
            continue
        key = (engine, name, a.get("EventName", ""), rkey, str(a.get("Original Nodes", "")))
        if key not in node_ids:
            node_ids[key] = len(nodes)
            nodes.append({
                "engine": engine,
                "kernel": name,
                "node": key[2],
                "recipe": rkey,
                "original_nodes": key[4],
                "reported_dtype": a.get("dataType", "")
            })
        if hw not in hw_ids:
            hw_ids[hw] = len(hw_names)
            hw_names.append(hw)
        row = [event["ts"], event["dur"], str(event.get("tid")), node_ids[key], hw_ids[hw]]
        dst.write(json.dumps(row, separators=(",", ":")) + "\n")
        hardware += 1
        first = min(first, row[0])
        last = max(last, row[0] + row[1])
        kernel_counts[(engine, name)] += 1
        if "prepared_" in name and len(examples) < 6:
            examples.append(event)
digest = hashlib.file_digest(args.trace.open("rb"), "sha256").hexdigest()
result = {
    "trace":
    str(args.trace),
    "trace_sha256":
    digest,
    "events":
    total,
    "schema_version":
    2,
    "hardware_columns": ["ts_us", "dur_us", "lane", "node", "hw_kind"],
    "hw_event_names":
    hw_names,
    "base_time_nanoseconds":
    int(base[1]) if base else None,
    "hardware_events":
    hardware,
    "first_us":
    first,
    "last_us":
    last,
    "modules":
    sorted(modules),
    "nodes":
    nodes,
    "recipe_names": {
        key: sorted(value)
        for key, value in recipe_names.items()
    },
    "host_enqueues":
    host_enqueues,
    "cpu_markers":
    markers,
    "metadata":
    metadata,
    "decoder_examples":
    examples,
    "kernel_counts": [{
        "engine": key[0],
        "kernel": key[1],
        "lane_events": count
    } for key, count in kernel_counts.most_common()]
}
(args.output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("trace", "hardware_events", "modules", "decoder_examples")}, indent=2))
