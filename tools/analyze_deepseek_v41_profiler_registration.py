# SPDX-License-Identifier: Apache-2.0
"""Audit the diagnostic recorder's call concurrency and SDK map control words."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def analyze(path):
    active, calls, findings = set(), {}, []
    maximum = 0
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        fields = line.split("\t")
        kind, key = fields[:2]
        if kind == "enter":
            if active:
                findings.append({
                    "line": number,
                    "kind": "overlapping registration calls",
                    "entered": key,
                    "already_active": sorted(active)
                })
            active.add(key)
            maximum = max(maximum, len(active))
            calls[key] = {"owner": fields[2], "recipe_info": fields[3], "debug_pointer": fields[4]}
        elif kind == "exit":
            active.discard(key)
            calls[key]["returned"] = True
        elif kind == "map":
            bucket_ptr, buckets, first, count = int(fields[2], 16), int(fields[3]), int(fields[4], 16), int(fields[5])
            calls[key]["map"] = dict(bucket_pointer=bucket_ptr, buckets=buckets, first_node=first, count=count)
            if not bucket_ptr or not buckets or buckets > 1000000 or count > 8190 or bool(first) != bool(count):
                findings.append({
                    "line": number,
                    "kind": "implausible SDK unordered-map control words",
                    "call": key,
                    "map": calls[key]["map"]
                })
        elif kind == "debug":
            major, minor, recipe, count = map(int, fields[2:6])
            calls[key]["debug"] = dict(major=major, minor=minor, recipe=recipe, count=count, nodes=fields[6])
        elif kind == "name":
            calls[key].setdefault("nodes", []).append({
                "index": int(fields[2]),
                "name": fields[3],
                "operation": fields[4],
                "dtype": fields[5]
            })
    ids = [call["debug"]["recipe"] for call in calls.values() if "debug" in call]
    result = {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "registrations": len(calls),
        "maximum_simultaneous_calls": maximum,
        "unfinished_calls": {
            key: calls[key]
            for key in active
        },
        "findings": findings,
        "unique_debug_recipe_ids": len(set(ids)),
        "duplicate_id_frequencies": {
            key: value
            for key, value in collections.Counter(ids).items() if value > 1
        },
        "limitation": "Recorder changes timing; control words cannot prove pointer ownership or causality."
    }
    path.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        key: value
        for key, value in result.items() if key not in ("unfinished_calls", "findings")
    }),
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recorders", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.recorders:
        analyze(path)
