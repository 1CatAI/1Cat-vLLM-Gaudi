# SPDX-License-Identifier: Apache-2.0
"""Collect four-rank trace intervals and exact serialized recipe identities.

The streaming extractor and recipe parser reuse the preserved V4 native-joint
analysis. No old layer counts, packet clustering or timing windows are reused.
"""

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


def parse_symbols(data, start):
    major, minor, rid, count = struct.unpack_from("<IIHI", data, start)
    assert major == 1 and minor == 2 and 0 < count < 10000
    offset = start + 14
    nodes, contexts = [], collections.defaultdict(list)
    for _ in range(count):
        device, context, full_context, descriptors, blob = struct.unpack_from("<IHIII", data, offset)
        offset += 18
        assert device < 16 and full_context < count and descriptors < 10000000
        strings = []
        for _ in range(3):
            length, = struct.unpack_from("<I", data, offset)
            offset += 4
            assert 0 < length < 16384
            raw = data[offset:offset + length]
            assert len(raw) == length and raw[-1] == 0 and b"\0" not in raw[:-1]
            strings.append(raw[:-1].decode("utf-8"))
            offset += length
        rois, = struct.unpack_from("<H", data, offset)
        offset += 2
        assert rois < 10000
        engines = list(data[offset:offset + rois])
        assert len(engines) == rois and all(value <= 64 for value in engines)
        offset += rois
        contexts[device].append(full_context)
        nodes.append({
            "device_type": device,
            "context_id": context,
            "full_context_id": full_context,
            "descriptors": descriptors,
            "kernel_blob_index": blob,
            "node": strings[0],
            "kernel": strings[1],
            "dtype": strings[2],
            "working_engines": engines
        })
    assert all(sorted(values) == list(range(len(values))) for values in contexts.values())
    return {"major": major, "minor": minor, "recipe_id": rid, "offset": start, "end_offset": offset, "nodes": nodes}


def recipe_symbols(path):
    data = path.read_bytes()
    target = struct.pack("<II", 1, 2)
    offset, matches = 0, []
    while (offset := data.find(target, offset)) >= 0:
        with contextlib.suppress(AssertionError, struct.error, UnicodeDecodeError, IndexError):
            matches.append(parse_symbols(data, offset))
        offset += 1
    if len(matches) != 1:
        raise ValueError(f"Recipe debug section is not unique: {path}, candidates={len(matches)}")
    return {**matches[0], "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs",
                        type=int,
                        choices=range(1, 5),
                        default=1,
                        help="Independent rank extraction workers; does not acquire another trace")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run / "trace-analysis"
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("extract_deepseek_v41_trace.py")
    shutil.copy2(source, output / source.name)
    shutil.copy2(__file__, output / Path(__file__).name)
    log = (run / "run.log").read_text(errors="replace")
    record = {
        "run": str(run),
        "timing_units_raw": "us",
        "report_units": "ms",
        "ranks": [],
        "extractor_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reuse": "20260909_dsv4-native-joint-decode/extract_native_trace.py and decode_native_recipe_symbols.py",
        "status": "collecting; physical calls and complete token windows not yet reconstructed"
    }

    def extract_rank(rank):
        pp, tp = divmod(rank, 2)
        stop = run / f"traces/rank{rank}-native-profile-stop.json"
        if not stop.is_file():
            raise ValueError(f"Rank {rank} profiler export has not completed")
        pids = set(re.findall(rf"Worker_PP{pp}_TP{tp} pid=(\d+)", log))
        if len(pids) != 1:
            raise ValueError(f"Rank {rank} has ambiguous process identity: {pids}")
        pid = pids.pop()
        traces = list((run / "traces").glob(f"*_{pid}.*.pt.trace.json.gz"))
        if len(traces) != 1:
            raise ValueError(f"Rank {rank} has {len(traces)} worker traces")
        with (output / f"extract-rank{rank}.log").open("w") as destination:
            subprocess.run(
                [sys.executable, str(source),
                 str(traces[0]), "--output",
                 str(output / f"rank{rank}")],
                stdout=destination,
                stderr=subprocess.STDOUT,
                check=True)
        recipes = [recipe_symbols(path) for path in sorted((run / f"recipes/rank{rank}").glob("*.recipe"))]
        if not recipes:
            raise ValueError(f"Rank {rank} has no private serialized recipes")
        (output / f"rank{rank}/recipe-symbols.json").write_text(
            json.dumps({
                "recipes": recipes,
                "source_runtime_profile": str(run / "runtime-profile.json")
            }, indent=2) + "\n")
        return {
            "rank": rank,
            "pp": pp,
            "tp": tp,
            "pid": int(pid),
            "trace": str(traces[0]),
            "serialized_recipes": len(recipes),
            "stats": json.loads(stop.read_text())
        }

    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(extract_rank, rank): rank for rank in range(4)}
        for future in as_completed(futures):
            rank = futures[future]
            try:
                item = future.result()
                record["ranks"].append(item)
                record["ranks"].sort(key=lambda item: item["rank"])
                print(f"Rank {rank}: extracted trace and {item['serialized_recipes']} exact recipe records", flush=True)
            except Exception as error:
                errors.append({"rank": rank, "error": repr(error)})
            record["errors"] = errors
            (output / "collection.json").write_text(json.dumps(record, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"Rank extraction failed; partial artifacts retained: {errors}")
    graph_roots = [run / "graphs"]
    seed = run / "recipe-seed-manifest.json"
    if seed.exists():
        graph_roots.append(Path(json.loads(seed.read_text())["compiled_graphs"]))
    graph_manifest = []
    for root in graph_roots:
        for path in sorted(root.rglob("*-symbol.pbtxt")):
            if "PostGraph" not in path.name and "PreGraph" not in path.name and "eager_final_graph" not in path.name:
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            graph_manifest.append({"path": str(path), "sha256": digest, "mtime_ns": path.stat().st_mtime_ns})
    (output / "graph-manifest.json").write_text(json.dumps(graph_manifest, indent=2) + "\n")
    record["status"] = "collected; token reconstruction and full per-kernel report pending"
    (output / "collection.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
