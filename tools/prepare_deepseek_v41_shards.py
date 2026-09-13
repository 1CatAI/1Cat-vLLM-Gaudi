# SPDX-License-Identifier: Apache-2.0
"""Prepare four immutable TP2/PP2 files, with Engram retained in its source mmap."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_gaudi.ops.deepseek_v41_weights import (  # noqa: E402
    LAYOUT_VERSION, MAX_TEMPORARY_BYTES, RankWriter, build_plan, canonical_hash, checkpoint_catalog,
    copy_experts, copy_plain, file_hash, host_manifest, publish_json, stage_for,
)
from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--upstream-lock", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage-ready-shards", action="store_true",
                        help="Prepare verified backbone files while Engram downloads; publish only after all hashes pass")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)


def prepare(args):
    started = time.monotonic()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("A published prepared checkpoint is immutable; use another output directory")
    if args.stage_ready_shards:
        remote = json.loads((args.checkpoint_audit / "remote-model.json").read_text())
        records = {}
        for entry in remote["siblings"]:
            name = entry["rfilename"]
            path = args.checkpoint_audit / "files" / (name.replace("/", "_") + ".json")
            if path.exists():
                records[name] = json.loads(path.read_text())
            elif entry.get("lfs"):
                records[name] = {"file": name, "bytes": entry["size"], "sha256": entry["lfs"]["sha256"]}
            else:
                raise RuntimeError(f"Wait for pinned metadata verification before preparing shards: {name}")
        checkpoint = {"revision": remote["sha"], "files": list(records.values())}
    else:
        checkpoint = json.loads(args.checkpoint_audit.read_text())
    sources = {record["file"]: record["sha256"] for record in checkpoint["files"]}
    config = json.loads((args.model / "config.json").read_text())
    if config["architectures"] != ["DeepseekV41ForCausalLM"] or config["text_config"]["num_hidden_layers"] != 40:
        raise ValueError("This preparation profile requires the 40-layer V4.1 Flash checkpoint")
    metadata_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    for name in metadata_files:
        if file_hash(args.model / name) != sources[name]:
            raise ValueError(f"Metadata changed after checkpoint verification: {name}")
    upstream = json.loads(args.upstream_lock.read_text())
    if not upstream.get("pull_requests"):
        raise ValueError("Missing upstream PR lock")
    catalog = checkpoint_catalog(args.model, allow_download_headers=args.stage_ready_shards)
    ranks, groups, tables = build_plan(catalog)
    if len(groups) != 43 or len(tables) != 4:
        raise ValueError("Expected 40 backbone MoEs, 3 draft MoEs and two Engram tables with scales")
    for source in catalog.values():
        if source.file.name not in sources:
            raise ValueError(f"Unverified source shard: {source.file.name}")
    hash_layout = EngramHashLayout.from_config(config["text_config"])
    quantization = {"checkpoint": config["quantization_config"], "prepared_layout_version": LAYOUT_VERSION,
                    "q16_row_block": 128, "q16_k_block": 128, "scale_group": 32,
                    "s16_encoding": "raw_e8m0_bits_shift_left_7", "dense_scale_encoding": "raw_u8",
                    "expert_tp": "intermediate", "w13_order": ["gate", "up"]}
    plan = {"format_version": 1, "model_revision": checkpoint["revision"], "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2, "pp_layer_ranges": [[0, 20], [20, 40]],
            "source_tensor_count": len(catalog), "source_shard_count": len({s.file for s in catalog.values()}),
            "metadata_sha256": {name: sources[name] for name in metadata_files},
            "encoding_sha256": {name: value for name, value in sources.items() if name.startswith("encoding/")},
            "prepared_layout_version": LAYOUT_VERSION, "engram_sharding": "complete_hash_heads",
            "quantization": quantization,
            "quantization_fingerprint": canonical_hash(quantization), "upstream_pr_lock": upstream,
            "upstream_lock_sha256": file_hash(args.upstream_lock),
            "source_file_sha256": sources,
            "ranks": {f"pp{pp}-tp{tp}": specs for (pp, tp), specs in ranks.items()}}
    fingerprint = canonical_hash(plan)
    plan_path = args.output / "preparation-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The existing preparation plan is different; refusing to reuse partial files")
    publish_json(plan_path, plan)
    from vllm_gaudi.ops.deepseek_v41_weights import ITEM_BYTES
    import math
    payload_bytes = sum(math.prod(spec["shape"]) * ITEM_BYTES[spec["dtype"]]
                        for specs in ranks.values() for spec in specs.values())
    print(json.dumps({"stage": "planned", "output_payload_gib": payload_bytes / 2**30,
                      "temporary_limit_gib": MAX_TEMPORARY_BYTES / 2**30, "plan_fingerprint": fingerprint}), flush=True)
    if args.plan_only:
        return
    # Existing partial files are sparse. Count allocated blocks when resuming,
    # not their logical lengths, so a truncated file cannot mask ENOSPC.
    allocated = sum(path.stat().st_blocks * 512 for path in args.output.glob("pp*-tp*.safetensors*"))
    if shutil.disk_usage(args.output).free < max(0, payload_bytes - allocated) + 8 * 2**30:
        raise RuntimeError("Prepared checkpoint destination lacks payload space plus an 8 GiB reserve")
    publish_json(args.output / "owned-process.json", {"pid": os.getpid(), "pgid": os.getpgrp(),
                 "started_at": datetime.now(timezone.utc).isoformat(), "plan_fingerprint": fingerprint})
    progress_path = args.output / "preparation-progress.json"
    progress = json.loads(progress_path.read_text()) if args.resume and progress_path.exists() else {
        "plan_fingerprint": fingerprint, "complete": [], "normal_scales": {}}
    if progress["plan_fingerprint"] != fingerprint:
        raise ValueError("Resume progress belongs to another plan")
    completed = set(progress["complete"])
    writers = {}
    def verified_source(source):
        if not source.file.exists():
            return False
        if not args.stage_ready_shards:
            return True
        path = args.checkpoint_audit / "files" / (source.file.name + ".json")
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        stat = source.file.stat()
        if (record["sha256"] != sources[source.file.name] or record.get("inode") != stat.st_ino
                or record.get("mtime_ns") != stat.st_mtime_ns):
            raise ValueError(f"Source file changed after verification: {source.file.name}")
        return True
    try:
        for (pp, tp), specs in ranks.items():
            path = args.output / f"pp{pp}-tp{tp}.safetensors.partial"
            final = path.with_suffix("")
            if args.resume and final.exists():
                if path.exists():
                    raise ValueError("Both partial and final rank files exist; refusing ambiguous recovery")
                path = final
            writers[pp, tp] = RankWriter(path, specs, {"format": "pt", "plan_fingerprint": fingerprint,
                "model_revision": checkpoint["revision"], "prepared_layout_version": str(LAYOUT_VERSION),
                "tp_rank": str(tp), "pp_rank": str(pp)}, resume=args.resume)
        def save_progress(key, selected):
            for writer in selected:
                writer.sync()
            completed.add(key)
            progress["complete"] = sorted(completed)
            publish_json(progress_path, progress)

        for (pp, tp), specs in ranks.items():
            key = f"plain-pp{pp}-tp{tp}"
            if key in completed:
                continue
            writer = writers[pp, tp]
            pending = False
            for name, spec in specs.items():
                if "source" in spec:
                    source = catalog[spec["source"]]
                    if verified_source(source):
                        copy_plain(source, writer, name, tp)
                    else:
                        pending = True
            if not pending:
                save_progress(key, [writer])
            print(json.dumps({"stage": key, "elapsed_s": time.monotonic() - started}), flush=True)
        for prefix, experts in sorted(groups.items()):
            if prefix in completed:
                continue
            pp = stage_for(prefix)
            if not all(verified_source(source) for matrices in experts.values() for source in matrices.values()):
                raise RuntimeError(f"Backbone shard verification is still pending for {prefix}")
            selected = [writers[pp, tp] for tp in range(2)]
            progress["normal_scales"][prefix] = copy_experts(prefix, experts, selected)
            save_progress(prefix, selected)
            print(json.dumps({"stage": prefix, "elapsed_s": time.monotonic() - started,
                              "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)
        if args.stage_ready_shards:
            complete_path = args.checkpoint_audit / "complete.json"
            print(json.dumps({"stage": "waiting_for_complete_checkpoint_audit"}), flush=True)
            while not complete_path.exists():
                time.sleep(10)
            verified = json.loads(complete_path.read_text())
            if verified["revision"] != checkpoint["revision"] or {
                    item["file"]: item["sha256"] for item in verified["files"]} != sources:
                raise ValueError("Final checkpoint audit differs from the preparation plan")
            final_catalog = checkpoint_catalog(args.model)
            if final_catalog != catalog:
                raise ValueError("Completed checkpoint headers differ from the staged preparation headers")
            for (pp, tp), specs in ranks.items():
                key = f"plain-pp{pp}-tp{tp}"
                if key in completed:
                    continue
                for name, spec in specs.items():
                    if "source" in spec:
                        source = final_catalog[spec["source"]]
                        if not verified_source(source):
                            raise RuntimeError("Checkpoint audit is complete but a source file has changed")
                        copy_plain(source, writers[pp, tp], name, tp)
                save_progress(key, [writers[pp, tp]])
    finally:
        for writer in writers.values():
            writer.close()
    rank_files = {}
    for (pp, tp), writer in writers.items():
        path = writer.path
        rank_files[f"pp{pp}-tp{tp}"] = {"file": path.name.removesuffix(".partial"), "sha256": file_hash(path),
                                         "bytes": path.stat().st_size, "tensors": len(writer.specs),
                                         "inode": path.stat().st_ino, "mtime_ns": path.stat().st_mtime_ns}
    host_files = {}
    for tp in range(2):
        path = args.output / f"engram-tp{tp}.json"
        publish_json(path, host_manifest(tables, tp, checkpoint["revision"], sources, hash_layout=hash_layout))
        host_files[str(tp)] = {"file": path.name, "sha256": file_hash(path)}
    for relative in sources:
        if relative.endswith(".safetensors") or relative.startswith("inference/"):
            continue
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.model / relative, destination)
    for writer in writers.values():
        if writer.path.suffix == ".partial":
            writer.path.replace(writer.path.with_suffix(""))
    manifest = {key: value for key, value in plan.items() if key != "ranks"}
    manifest.update({"plan_fingerprint": fingerprint, "rank_files": rank_files, "engram_host_shards": host_files,
                     "normal_scales": progress["normal_scales"], "temporary_limit_bytes": MAX_TEMPORARY_BYTES,
                     "preparation_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "elapsed_s": time.monotonic() - started, "created_at": datetime.now(timezone.utc).isoformat()})
    publish_json(args.output / "manifest.json", manifest)
    print(json.dumps({"stage": "published", "manifest": str(args.output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
