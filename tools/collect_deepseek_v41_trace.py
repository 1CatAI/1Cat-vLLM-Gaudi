# SPDX-License-Identifier: Apache-2.0
"""Collect four-rank trace intervals and exact serialized recipe identities.

The streaming extractor and recipe parser reuse the preserved V4 native-joint
analysis. No old layer counts, packet clustering or timing windows are reused.
"""

import argparse
import collections
from contextlib import suppress
from concurrent.futures import ProcessPoolExecutor
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
        with suppress(AssertionError, struct.error, UnicodeDecodeError, IndexError):
            matches.append(parse_symbols(data, offset))
        offset += 1
    if len(matches) != 1:
        raise ValueError(f"Recipe debug section is not unique: {path}, candidates={len(matches)}")
    return {**matches[0], "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def worker_coordinates(rank, stats):
    # Old archives predate explicit topology and were all TP2 x PP2.
    topology = stats.get("topology", {"tensor_parallel_size": 2, "pipeline_parallel_size": 2})
    tp_size, pp_size = topology["tensor_parallel_size"], topology["pipeline_parallel_size"]
    if (tp_size, pp_size) not in ((2, 2), (4, 1)) or not 0 <= rank < tp_size * pp_size:
        raise ValueError("Trace topology must describe TP4 x PP1 or legacy TP2 x PP2")
    return divmod(rank, tp_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-normalized",
                        action="store_true",
                        help="Assemble an existing complete normalization after verifying every raw/CPU identity")
    parser.add_argument("--raw-workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--raw-chunk-ms", type=float, default=200)
    parser.add_argument("--capture-dir",
                        type=Path,
                        help="One archived acquisition with its own worker traces and start/stop counters")
    args = parser.parse_args()
    run = args.run.resolve()
    trace_dir = args.capture_dir.resolve() if args.capture_dir else run / "traces"
    output = args.output or run / "trace-analysis"
    output.mkdir(parents=True, exist_ok=args.reuse_normalized)
    if args.reuse_normalized:
        versions = output / "analysis-source-versions"
        for previous in [*output.glob("*.py"), output / "analysis-sources.json"]:
            if previous.is_file():
                digest = hashlib.sha256(previous.read_bytes()).hexdigest()
                destination = versions / digest / previous.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(previous, destination)
    source = Path(__file__).with_name("extract_deepseek_v41_trace.py")
    shutil.copy2(source, output / source.name)
    shutil.copy2(__file__, output / Path(__file__).name)
    analysis_sources = {}
    for name in ("normalize_deepseek_v41_raw_trace.py", "analyze_deepseek_v41_trace.py",
                 "map_deepseek_v41_trace_contracts.py", "report_deepseek_v41_trace.py", "render_deepseek_v41_trace.py",
                 "deepseek_v41_trace_accounting.py", "report_deepseek_v41_latency_ledger.py"):
        script = Path(__file__).with_name(name)
        shutil.copy2(script, output / name)
        analysis_sources[name] = hashlib.sha256(script.read_bytes()).hexdigest()
    (output / "analysis-sources.json").write_text(json.dumps(analysis_sources, indent=2) + "\n")
    log = (run / "run.log").read_text(errors="replace")
    record = {
        "run": str(run),
        "capture_dir": str(trace_dir),
        "timing_units_raw": "us",
        "report_units": "ms",
        "ranks": [],
        "extractor_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reuse": "20260909_dsv4-native-joint-decode/extract_native_trace.py and decode_native_recipe_symbols.py",
        "status": "collecting; physical calls and complete token windows not yet reconstructed"
    }
    active_graphs = {}
    raw_manifest = trace_dir / "manifest.json"
    raw_manifest = json.loads(raw_manifest.read_text()) if raw_manifest.exists() else {}
    raw_ranks = {row["rank"]: row for row in raw_manifest.get("hardware", []) if row.get("format") == "raw_hltv_cpu"}
    if raw_ranks:
        if set(raw_ranks) != set(range(4)):
            raise ValueError("Raw capture requires all four rank manifests")
        sys.path.insert(0, str(output.resolve()))
        from normalize_deepseek_v41_raw_trace import normalize
        # Export/normalize independent owned ranks concurrently on the caller's
        # analysis CPU affinity. Each SDK parser is bounded by its time window.
        if args.reuse_normalized:
            for rank, row in sorted(raw_ranks.items()):
                directory = output / f"rank{rank}"
                inventory = json.loads((directory / "inventory.json").read_text())
                provenance = json.loads((directory / "raw-provenance.json").read_text())
                assert inventory.get("format") == "raw_hltv_cpu" and inventory["hardware_events"] > 0
                assert provenance["metadata"] == row["metadata"], "Capture clock/ownership identity changed"
                for name, key in (("bundle", "raw_sha256"), ("cpu_trace", "cpu_sha256")):
                    with (trace_dir / row[name]).open("rb") as stream:
                        assert hashlib.file_digest(stream, "sha256").hexdigest() == provenance[key]
        else:
            with ProcessPoolExecutor(max_workers=args.raw_workers) as pool:
                futures = [
                    pool.submit(normalize, trace_dir / row["bundle"], trace_dir / row["cpu_trace"], row["metadata"],
                                output / f"rank{rank}", run / "profiler-config.json", args.raw_chunk_ms)
                    for rank, row in sorted(raw_ranks.items())
                ]
                for future in futures:
                    future.result()
    for rank in range(4):
        stop = trace_dir / f"rank{rank}-native-profile-stop.json"
        if not stop.is_file():
            raise ValueError(f"Rank {rank} profiler export has not completed")
        stats = json.loads(stop.read_text())
        pp, tp = worker_coordinates(rank, stats)
        label = (rf"Worker_(?:PP0_)?TP{tp}"
                 if stats.get("topology", {}).get("pipeline_parallel_size") == 1 else rf"Worker_PP{pp}_TP{tp}")
        pids = set(re.findall(label + r" pid=(\d+)", log))
        if len(pids) != 1:
            raise ValueError(f"Rank {rank} has ambiguous process identity: {pids}")
        pid = pids.pop()
        traces = list(trace_dir.glob(f"*_{pid}.*.pt.trace.json.gz"))
        if len(traces) != 1:
            raise ValueError(f"Rank {rank} has {len(traces)} worker traces")
        raw = raw_ranks.get(rank)
        if raw:
            inventory_path = output / f"rank{rank}/inventory.json"
            assert inventory_path.is_file(), "Raw worker did not finish normalization"
        else:
            with (output / f"extract-rank{rank}.log").open("w") as destination:
                subprocess.run(
                    [sys.executable,
                     str(source), str(traces[0]), "--output",
                     str(output / f"rank{rank}")],
                    stdout=destination,
                    stderr=subprocess.STDOUT,
                    check=True)
        inventory = json.loads((output / f"rank{rank}/inventory.json").read_text())
        if not inventory["hardware_events"]:
            raise ValueError(f"Rank {rank} capture contains no device events; CPU-only data cannot qualify kernels")
        active_graphs[rank] = {Path(row[3]).name for row in inventory["host_enqueues"]}
        if raw:
            recipes = json.loads((output / f"rank{rank}/recipe-symbols.json").read_text())["recipes"]
        else:
            recipes = [recipe_symbols(path) for path in sorted((run / f"recipes/rank{rank}").glob("*.recipe"))]
        if not recipes:
            raise ValueError(f"Rank {rank} has no private serialized recipes")
        (output / f"rank{rank}/recipe-symbols.json").write_text(
            json.dumps({
                "recipes": recipes,
                "source_runtime_profile": str(run / "runtime-profile.json")
            }, indent=2) + "\n")
        record["ranks"].append({
            "rank": rank,
            "pp": pp,
            "tp": tp,
            "pid": int(pid),
            "trace": str(traces[0]),
            "serialized_recipes": 0 if raw else len(recipes),
            "captured_recipe_graphs": len(recipes) if raw else 0,
            "stats": stats
        })
        (output / "collection.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Rank {rank}: extracted trace and {len(recipes)} exact recipe records", flush=True)
    if raw_manifest.get("status") == "raw_pending_offline_validation":
        graph_manifest = [
            graph for rank in range(4)
            for graph in json.loads((output / f"rank{rank}/raw-graph-manifest.json").read_text())
        ]
        (output / "graph-manifest.json").write_text(json.dumps(graph_manifest, indent=2) + "\n")
        record.update(format="raw_hltv_cpu", status="normalized; complete phase and kernel accounting pending")
        (output / "collection.json").write_text(json.dumps(record, indent=2) + "\n")
        return
    graph_roots = [run / "graphs"]
    seed = run / "recipe-seed-manifest.json"
    if seed.exists():
        graph_roots.append(Path(json.loads(seed.read_text())["compiled_graphs"]))
    graph_manifest = []
    excluded_inactive_eager_graphs = 0
    for root in graph_roots:
        for path in sorted(root.rglob("*-symbol.pbtxt")):
            if not any(
                    stage in path.name
                    for stage in ("PostGraph", "PreGraph", "eager_final_graph", "eager-finalgraph", "eager-pregraph")):
                continue
            # Preserve every static graph, but avoid parsing the unbounded
            # sequence of eager dumps from requests outside this acquisition.
            # The full dumps remain in the run archive for further research.
            if "-eager" in path.name:
                rank_match = re.search(r"/rank(\d+)/", str(path))
                prefix = path.name.split("-eager", 1)[0]
                if rank_match and prefix not in active_graphs[int(rank_match[1])]:
                    excluded_inactive_eager_graphs += 1
                    continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            graph_manifest.append({"path": str(path), "sha256": digest, "mtime_ns": path.stat().st_mtime_ns})
    (output / "graph-manifest.json").write_text(json.dumps(graph_manifest, indent=2) + "\n")
    record["status"] = "collected; token reconstruction and full per-kernel report pending"
    record["graph_selection"] = dict(rule="all static graphs plus exact enqueued eager graph names per rank",
                                     excluded_inactive_eager_graphs=excluded_inactive_eager_graphs,
                                     original_graph_roots=[str(path) for path in graph_roots])
    (output / "collection.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
