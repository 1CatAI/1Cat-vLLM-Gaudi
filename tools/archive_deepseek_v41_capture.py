# SPDX-License-Identifier: Apache-2.0
"""Freeze one completed four-rank acquisition before another profiler session."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

import ijson


def hardware_events(path):
    counts = collections.Counter()
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") != "X" or event.get("dur", 0) <= 0:
                continue
            name = (event.get("args") or {}).get("HW event name", "").upper()
            engine = next((engine for engine in ("TPC", "MME", "DMA", "NIC") if engine in name), None)
            if engine is not None:
                counts[engine] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    result = json.loads((args.request / "result.json").read_text())
    profile = json.loads((args.request / "profile.json").read_text())
    assert result["status"] == "passed" and result["profile"] in ("prefill", "decode")
    assert [event["action"] for event in profile] == ["start", "stop"]
    assert all(event["status"] == 200 for event in profile)
    start_ns = profile[0]["start_ns"]
    traces = args.run / "traces"
    output = args.request / "capture"
    output.mkdir(exist_ok=False)
    log = (args.run / "run.log").read_text(errors="replace")
    manifest = dict(phase=result["profile"], request=str(args.request.resolve()), files=[], hardware=[])
    for rank in range(4):
        pids = set(re.findall(rf"Worker_(?:PP0_)?TP{rank} pid=(\d+)", log))
        assert len(pids) == 1, (rank, pids)
        pid = pids.pop()
        stop = json.loads((traces / f"rank{rank}-native-profile-stop.json").read_text())
        raw = stop.get("raw_trace")
        matches = [path for path in traces.glob(f"*_{pid}.*.pt.trace.json.gz")
                   if path.stat().st_mtime_ns >= start_ns]
        assert len(matches) == 1, (rank, matches)
        paths = [matches[0], *(traces / f"rank{rank}-native-profile-{phase}.json"
                              for phase in ("start", "stop"))]
        if raw:
            assert raw["capture_cpu"] and raw["pid"] == int(pid)
            paths.extend(Path(path) for path in raw["raw_files"])
        engram = traces / f"rank{rank}-engram-profile.json"
        if engram.exists() and engram.stat().st_mtime_ns >= start_ns:
            paths.append(engram)
        for path in paths:
            assert path.stat().st_mtime_ns >= start_ns, path
            destination = output / path.name
            # Stats are overwritten by the next capture. Copy these rather
            # than sharing their inodes; immutable trace files can be linked.
            if path.name.endswith(".gz"):
                destination.hardlink_to(path)
            else:
                shutil.copy2(path, destination)
            with destination.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"].append(dict(rank=rank, path=destination.name,
                                           bytes=destination.stat().st_size, sha256=digest))
        if raw:
            bundles = []
            for path in raw["raw_files"]:
                source = output / Path(path).name
                if "_accel" not in source.name:
                    continue
                with tarfile.open(source, "r:gz") as archive:
                    members = archive.getnames()
                    owners = {int(match[1]) for name in members
                              if (match := re.fullmatch(r"prof-data/gcfg/gcfg_(\d+)\.txt", name))}
                    assert owners == {int(pid)}, (source, owners)
                    assert any(name.startswith("prof-data/bin/") for name in members)
                    assert "prof-data/host/host_trace.json" in members
                bundles.append(source.name)
            assert len(bundles) == 1, bundles
            manifest["hardware"].append(dict(rank=rank, format="raw_hltv_cpu", bundle=bundles[0],
                                              cpu_trace=matches[0].name, metadata=raw))
        else:
            manifest["hardware"].append(dict(rank=rank, engines=hardware_events(output / matches[0].name)))
    if all(row.get("format") == "raw_hltv_cpu" for row in manifest["hardware"]):
        manifest["status"] = "raw_pending_offline_validation"
    else:
        manifest["status"] = ("passed" if all(row["engines"].get("TPC") and row["engines"].get("MME")
                                              for row in manifest["hardware"]) else "failed_no_device_compute")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if manifest["status"] == "failed_no_device_compute":
        raise RuntimeError("Acquisition lacks TPC/MME events on one or more ranks; preserved for diagnosis")


if __name__ == "__main__":
    main()
