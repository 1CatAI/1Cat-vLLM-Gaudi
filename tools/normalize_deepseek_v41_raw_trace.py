# SPDX-License-Identifier: Apache-2.0
"""Normalize SDK raw CSV, captured post-graphs and CPU scopes without live expansion."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


NORMALIZER_SHA256 = digest(Path(__file__))


class Clock:
    def __init__(self, metadata, base):
        first, last = metadata["clock_samples"]
        self.raw0 = (first["monotonic_raw_before_ns"] + first["monotonic_raw_after_ns"]) // 2
        raw1 = (last["monotonic_raw_before_ns"] + last["monotonic_raw_after_ns"]) // 2
        wall0 = (first["wall_before_ns"] + first["wall_after_ns"]) // 2
        wall1 = (last["wall_before_ns"] + last["wall_after_ns"]) // 2
        wall_scale = (wall1 - wall0) / (raw1 - self.raw0)
        raw_scopes = metadata.get("scope_clock_domain") == "CLOCK_MONOTONIC_RAW"
        self.scale = 1. if raw_scopes else wall_scale
        if not raw_scopes and not .999 < self.scale < 1.001:
            raise ValueError("Raw device/CPU clock calibration changed rate unexpectedly")
        self.relative0 = (self.raw0 if raw_scopes else wall0) - base
        self.synapse0 = first["synapse_clock_ns"]
        span = raw1 - self.raw0 if raw_scopes else wall1 - wall0
        self.synapse_scale = span / (last["synapse_clock_ns"] - self.synapse0)
        self.proof = dict(output_clock_domain="CLOCK_MONOTONIC_RAW" if raw_scopes else "wall",
                          raw_to_output_scale=self.scale, synapse_to_output_scale=self.synapse_scale,
                          observed_raw_to_wall_scale=wall_scale,
                          samples=metadata["clock_samples"], base_time_nanoseconds=base,
                          bracket_uncertainty_ns=max((row["wall_after_ns"] - row["wall_before_ns"] +
                                                      row["monotonic_raw_after_ns"] -
                                                      row["monotonic_raw_before_ns"]) / 2
                                                     for row in metadata["clock_samples"]))

    def raw(self, microseconds):
        return ((float(microseconds) * 1000 - self.raw0) * self.scale + self.relative0) / 1000

    def synapse(self, microseconds):
        return ((float(microseconds) * 1000 - self.synapse0) * self.synapse_scale + self.relative0) / 1000


def engine(name):
    if "TPC" in name:
        return "TPC"
    if "MME" in name:
        return "MME"
    if "DMA" in name:
        return "DMA"
    if "NIC" in name or "STM_" in name:
        return "NIC"
    return None


def tensor_contract(value):
    # Synapse stores dimensions in fastest-to-slowest order. Preserve that
    # original layout alongside the conventional row-major presentation.
    return dict(name=value["name"], shape=list(reversed(value["max_shape"])), dtype=value["dtype"],
                bytes=math.prod(value["max_shape"]) * value.get("dtype_bit_size", 0) // 8,
                location=value.get("allocation", "unknown"), strides=value.get("strides"),
                alias=value.get("alias"), raw_synapse_tensor=value)


def owned_csv_rows(parts):
    """Pair each padded export independently; give each real BEGIN one owner.

    SDK time filtering can synthesize truncated MME boundaries. Padding keeps
    those outside the owned core. Never splice BEGIN/END from separate exports.
    """
    for path, low, high, parse_high in parts:
        pending = collections.defaultdict(collections.deque)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                stamp = float(row["Timestamp[usec]"]) * 1000
                kind = engine(row["Engine"])
                if kind is None or row["Event type"] not in ("BEGIN", "END"):
                    if low <= stamp < high:
                        yield row, None
                    continue
                # Name is presentation text, not identity. In unnamed SDK
                # contexts BEGIN and END are named after different HW ports.
                key = tuple(row[field] for field in ("Engine", "Recipe ID", "Context ID", "Unique Node ID"))
                if row["Event type"] == "BEGIN":
                    pending[key].append((stamp, row))
                elif pending[key]:
                    begin, first = pending[key].popleft()
                    if low <= begin < high:
                        if stamp >= parse_high - 1000 and high < parse_high:
                            raise ValueError("Owned interval reaches parser padding boundary; increase overlap")
                        yield first, row
                elif low <= stamp < high:
                    raise ValueError(f"Missing BEGIN inside owned raw chunk: {row}")
        for rows in pending.values():
            for stamp, row in rows:
                if low <= stamp < high:
                    raise ValueError(f"Missing END inside owned raw chunk: {row}")


def resolve_recipe_starts(pending, names, recipe_ids_at_start, observed_recipe_ids):
    recipe_starts, unresolved_recipe_starts = [], []
    for stamp, raw_stamp, name in pending:
        # ENG ARC labels omit the finite-width recipe ID. First recover it
        # from the exact same hardware start timestamp and graph name. An
        # otherwise unique observed ID is also unambiguous. Never overwrite a
        # graph-name collision with the last archived recipe.
        candidates = recipe_ids_at_start.get((raw_stamp, name), set())
        if not candidates:
            candidates = observed_recipe_ids.get(name, set()) or names[name]
        if len(candidates) == 1:
            rid = next(iter(candidates))
            recipe_starts.append([stamp, 0, f"{rid}@{name}:", name])
        else:
            unresolved_recipe_starts.append(dict(stamp_us=stamp, raw_stamp=raw_stamp,
                                                graph_name=name, candidate_ids=sorted(candidates)))
    return recipe_starts, unresolved_recipe_starts


def normalize(bundle, cpu_trace, metadata, output, config, chunk_ms=200, overlap_ms=50, csv_source=None):
    subprocess.run([sys.executable, str(Path(__file__).with_name("extract_deepseek_v41_trace.py")),
                    str(cpu_trace), "--output", str(output)], check=True, stdout=subprocess.DEVNULL)
    inventory_path = output / "inventory.json"
    inv = json.loads(inventory_path.read_text())
    clock = Clock(metadata, inv["base_time_nanoseconds"])
    graphs_dir = output / "raw-graphs"
    graphs_dir.mkdir()
    graphs, contracts, recipes = [], [], []
    by_unique, names = {}, collections.defaultdict(set)
    raw_input = output / "raw-input"
    debug_dir = raw_input / "debug-info"
    debug_dir.mkdir(parents=True)
    binaries, host, state = [], None, None
    with tarfile.open(bundle, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name == "prof-data/profile_state.json":
                state = json.load(archive.extractfile(member))
                continue
            if member.name == "prof-data/host/host_trace.json":
                host = json.load(archive.extractfile(member))
                continue
            if member.name.startswith(("prof-data/bin/", "prof-data/debug-info/")):
                destination = (debug_dir if "/debug-info/" in member.name else raw_input) / Path(member.name).name
                with destination.open("wb") as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream, length=1 << 20)
                if "/bin/" in member.name:
                    binaries.append(destination)
                continue
            if not member.name.startswith("prof-data/post-graph/") or not member.name.endswith(".json"):
                continue
            path = graphs_dir / Path(member.name).name
            path.write_bytes(archive.extractfile(member).read())
            record = dict(path=str(path.resolve()), sha256=digest(path), format="Synapse post-graph JSON")
            graphs.append(record)
            for graph in json.loads(path.read_text())["graphs"]:
                rid = graph["recipe_debug_id"]
                identity = f"{rid}@{graph['name']}"
                names[graph["name"]].add(rid)
                tensors = {tensor["name"]: tensor_contract(tensor) for tensor in graph["tensors"]}
                nodes = []
                for node in graph["nodes"]:
                    kind = engine(node["engine"])
                    if node.get("is_logical") or kind not in ("TPC", "MME", "DMA"):
                        continue
                    symbol = dict(device_type={"TPC": 1, "MME": 0, "DMA": 8}[kind],
                                  context_id=node["context_id"], full_context_id=node["context_id"],
                                  node=node["name"], kernel=node["guid"],
                                  working_engines=node.get("tpc_working_engines", []),
                                  roi_count=node.get("num_of_ROIs"), unique_node_id=node["id"])
                    inputs = [tensors[name] for name in node["input_tensors"] if name in tensors]
                    outputs = [tensors[name] for name in node["output_tensors"] if name in tensors]
                    contracts.append(dict(recipe_id=identity, raw_recipe_id=rid, symbol=symbol,
                                          graph=record, matched=True,
                                          inputs=inputs, outputs=outputs, attributes={}, raw_node=node,
                                          provenance="same HLTV recipe_debug_id and Unique Node ID"))
                    key = (str(rid), graph["name"], str(node["id"]))
                    if key in by_unique and by_unique[key] != symbol:
                        raise ValueError(f"Ambiguous raw recipe/node identity: {key}")
                    by_unique[key] = symbol
                    nodes.append(symbol)
                recipes.append(dict(recipe_id=identity, raw_recipe_id=rid, path=str(path.resolve()),
                                    sha256=record["sha256"], nodes=nodes))
    if state is None or len(binaries) != 1:
        raise ValueError("Raw capture requires one binary and its exact profile_state")
    (output / "recipe-symbols.json").write_text(json.dumps(dict(recipes=recipes), indent=2) + "\n")
    (output / "node-contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
    (output / "raw-graph-manifest.json").write_text(json.dumps(graphs, indent=2) + "\n")
    (output / "raw-profile-state.json").write_text(json.dumps(state, indent=2) + "\n")
    parser_dir = output / "raw-parser"
    parser_dir.mkdir()
    parse_config = json.loads(config.read_text())
    hardware_config = next(plugin["values"] for plugin in parse_config["Plugins"] if plugin["name"] == "HwTrace")
    options = hardware_config["parseOptions"]
    options.update(skipParse={"value": False}, mergeWithHost={"value": False}, addEnqueuesFlow={"value": False},
                   traceAnalyzer={"enable": {"value": False}, "traceAnalyzerJson": {"value": False}})
    options["outputPerInvocation"] = {name: {"value": name == "csv"} for name in
                                      ("csv", "binary", "dbgInfo", "hltv", "hltvWithHost", "json", "text", "log")}
    parse_config_path = output / "offline-parser-config.json"
    parse_config_path.write_text(json.dumps(parse_config, indent=2) + "\n")
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = "/usr/lib/habanatools/habana_plugins:/usr/lib/habanalabs"
    calibrations = [die for item in state["host_device_time_diff"] for die in item["die"]]
    low = min(item["first"]["host"] for item in calibrations)
    high = max(item["second"]["host"] for item in calibrations)
    step = max(1, int(chunk_ms * 1e6)) if chunk_ms > 0 else high - low + 1
    padding = int(overlap_ms * 1e6)
    parts = []
    if csv_source is not None:
        source_root = csv_source.parent
        saved = json.loads((source_root / "csv-reuse-contract.json").read_text())
        assert saved["chunk_ms"] == chunk_ms and saved["overlap_ms"] == overlap_ms
        assert saved["binary_sha256"] == digest(binaries[0])
        assert saved["profile_state"] == state and saved["parser_config"] == parse_config
    commands, files = [], []
    for index, begin in enumerate(range(low, high, step)):
        directory = parser_dir / f"chunk{index:04d}"
        directory.mkdir()
        end = min(high, begin + step)
        parse_low, parse_high = max(low, begin - padding), min(high, end + padding)
        command = ["/opt/habanalabs/bin/synprof_parser", str(binaries[0]), "--gaudi2", "--csv",
                   "--conf", str(parse_config_path), "--outdir", str(directory),
                   "--dbg_dir", str(debug_dir), "--post_graph_dir", str(graphs_dir),
                   "--profile_state", str(output / "raw-profile-state.json"),
                   "--time-ranges", f"{parse_low}-{parse_high}"]
        commands.append(command)
        if csv_source is not None:
            cached = list((csv_source / directory.name).glob("*.csv.gz"))
            if len(cached) != 1 or list((csv_source / directory.name).glob("*.csv")):
                raise ValueError("Only closed, complete SDK CSV exports can be reused")
            compressed = cached[0]
        else:
            with (directory / "parser.log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            generated = list(directory.glob("*.csv"))
            if len(generated) != 1:
                raise ValueError(f"Expected one SDK CSV in {directory}, found {generated}")
            # CSV expands raw data by orders of magnitude. Preserve its exact
            # bytes compressed before starting another chunk; raw HLTV stays intact.
            compressed = generated[0].with_suffix(".csv.gz")
            with generated[0].open("rb") as source, gzip.open(compressed, "wb", compresslevel=1) as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            generated[0].unlink()
        files.append(compressed)
        parts.append((compressed, begin, end + int(end == high), parse_high))
        elapsed_ms = (min(high, begin + step) - low) / 1e6
        print(f"{output.name}: parsed raw time chunk {index}: {elapsed_ms:.3f} ms", flush=True)
    nodes, node_ids, unmatched, recipe_starts = [], {}, [], []
    pending_recipe_starts = []
    recipe_ids_at_start = collections.defaultdict(set)
    observed_recipe_ids = collections.defaultdict(set)
    counts, all_rows, hardware, zero_intervals = collections.Counter(), 0, 0, 0
    first, last = float("inf"), float("-inf")
    with gzip.open(output / "hardware.jsonl.gz", "wt", compresslevel=1) as out:
        for begin_row, end_row in owned_csv_rows(parts):
            row = end_row or begin_row
            all_rows += 2 if end_row is not None else 1
            kind = engine(row["Engine"])
            stamp = clock.raw(row["Timestamp[usec]"])
            if row["Trace Source"] == "SW" and row["Engine"].startswith("ENG ARC") and row["Name"] in names:
                pending_recipe_starts.append((stamp, row["Timestamp[usec]"], row["Name"]))
            if end_row is None:
                continue
            begin = clock.raw(begin_row["Timestamp[usec]"])
            raw_name, raw_id = begin_row["Recipe name"], begin_row["Recipe ID"]
            if raw_id and len(names.get(raw_name, ())) > 1:
                recipe_ids_at_start[(begin_row["Timestamp[usec]"], raw_name)].add(int(raw_id))
                observed_recipe_ids[raw_name].add(int(raw_id))
            if stamp < begin:
                raise ValueError("Negative device event duration")
            if stamp == begin:
                zero_intervals += 1
                continue
            rid = row["Recipe ID"]
            symbol = by_unique.get((rid, row["Recipe name"], row["Unique Node ID"]))
            if symbol is not None and symbol["device_type"] != {"TPC": 1, "MME": 0, "DMA": 8}.get(kind):
                raise ValueError("Device event engine differs from its captured post-graph node")
            recipe = f"{rid}@{row['Recipe name']}:"
            kernel = symbol["kernel"] if symbol else row["Operation"] or row["Name"]
            if kind == "DMA" and symbol is None:
                # SDK export-local command indices are metadata, not distinct
                # hardware kernels. The original name remains on the node.
                kernel = re.sub(r" apiId=\d+ chunk=\d+", "", kernel)
            if kind == "MME" and not row["Recipe ID"] and not row["Recipe name"]:
                kernel = "unresolved_mme_recipe0"
            name = symbol["node"] if symbol else row["Name"]
            identity = (kind, recipe, name, kernel)
            if identity not in node_ids:
                node_ids[identity] = len(nodes)
                nodes.append(dict(engine=kind, kernel=kernel, node=name, recipe=recipe,
                                  original_nodes="", reported_dtype=row["Data Type"],
                                  raw_unique_node_id=row["Unique Node ID"], raw_context_id=row["Context ID"],
                                  raw_event_name=row["Name"]))
            out.write(json.dumps([begin, stamp - begin, row["Engine"], node_ids[identity], 0, ""],
                                 separators=(",", ":")) + "\n")
            hardware += 1
            counts[(kind, kernel)] += 1
            first, last = min(first, begin), max(last, stamp)
    recipe_starts, unresolved_recipe_starts = resolve_recipe_starts(
        pending_recipe_starts, names, recipe_ids_at_start, observed_recipe_ids)
    (output / "raw-recipe-start-ambiguities.json").write_text(json.dumps(unresolved_recipe_starts, indent=2) + "\n")
    host_events = 0
    host_domain = None
    if host is not None:
        events = [event for event in host["traceEvents"] if event.get("ph") == "X" and event.get("dur", 0) > 0]
        # HLTV-with-host normally already converts host TSC timestamps to the
        # hardware monotonic-raw clock. Select only an explicitly calibrated
        # domain whose captured range overlaps this acquisition.
        center = sum(row[0] for row in recipe_starts) / max(1, len(recipe_starts))
        sample = max(events, key=lambda event: event["ts"]) if events else None
        if sample:
            host_domain = min(("raw", "synapse"), key=lambda name: abs(getattr(clock, name)(sample["ts"]) - center))
        convert = getattr(clock, host_domain or "raw")
        scale = clock.scale if host_domain == "raw" else clock.synapse_scale
        with gzip.open(output / "host.jsonl.gz", "at", compresslevel=1) as stream:
            for event in events:
                start, duration = convert(event["ts"]), event["dur"] * scale
                args = event.get("args") or {}
                stream.write(json.dumps([start, duration, str(event.get("pid")), str(event.get("tid")),
                                         event.get("cat"), event["name"], args], separators=(",", ":")) + "\n")
                host_events += 1
                if "enqueue" in event["name"].lower() and args.get("recipeName") and args.get("recipeId"):
                    inv["host_enqueues"].append([start, duration, f"{args['recipeId']}@{args['recipeName']}:",
                                                 args["recipeName"]])
    if not hardware or not any(key[0] == "MME" for key in counts) or not any(key[0] == "TPC" for key in counts):
        raise ValueError("Raw normalization did not recover complete TPC and MME intervals")
    identity = dict(raw_bundle=str(bundle.resolve()), raw_sha256=digest(bundle), cpu_trace=str(cpu_trace.resolve()),
                    cpu_sha256=digest(cpu_trace), metadata=metadata, parser_commands=commands,
                    parser_csv_files=[dict(path=str(path), sha256=digest(path)) for path in files],
                    normalization_sha256=NORMALIZER_SHA256, chunk_ms=chunk_ms, overlap_ms=overlap_ms,
                    reused_csv_source=str(csv_source) if csv_source else None,
                    chunk_ownership="BEGIN in half-open core; complete pair in independently padded CSV")
    (output / "raw-provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "raw-unmatched-events.json").write_text(json.dumps(unmatched, indent=2) + "\n")
    clock.proof.update(host_event_clock_domain=host_domain, hardware_clock_domain="CLOCK_MONOTONIC_RAW")
    (output / "clock-alignment.json").write_text(json.dumps(clock.proof, indent=2) + "\n")
    inv.update(trace_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
               format="raw_hltv_cpu", nodes=nodes, hardware_events=hardware, first_us=first, last_us=last,
               raw_owned_csv_rows=all_rows, raw_zero_duration_pairs=zero_intervals,
               raw_unmatched_events=len(unmatched), raw_host_events=host_events,
               device_recipe_starts=sorted(recipe_starts), raw_unresolved_recipe_starts=len(unresolved_recipe_starts),
               modules=[state.get("Module id")],
               kernel_counts=[dict(engine=kind, kernel=kernel, lane_events=count)
                              for (kind, kernel), count in counts.most_common()],
               hw_event_names=["paired SDK raw CSV BEGIN/END; includes hidden MME START_EVENT"])
    inventory_path.write_text(json.dumps(inv, indent=2) + "\n")
    summary = dict(status="passed", hardware_events=hardware, recipe_starts=len(recipe_starts),
                   unmatched=len(unmatched), host_events=host_events, kernels=len(nodes),
                   clock_alignment=clock.proof, output=str(output.resolve()))
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("cpu_trace", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=float, default=200)
    parser.add_argument("--overlap-ms", type=float, default=50)
    parser.add_argument("--csv-source", type=Path)
    args = parser.parse_args()
    normalize(args.bundle, args.cpu_trace, json.loads(args.metadata.read_text()), args.output,
              args.config, args.chunk_ms, args.overlap_ms, args.csv_source)
