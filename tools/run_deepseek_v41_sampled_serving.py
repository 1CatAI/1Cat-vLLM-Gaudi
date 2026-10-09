# SPDX-License-Identifier: Apache-2.0
"""Leased official EOS + companion trace using the established DSpark profile."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def align_deployment_artifacts(profile, baseline):
    """Keep candidate operators while pinning the qualified deployment libraries."""
    candidate_root = Path(profile["environment"]["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"]).resolve()
    baseline_root = Path(baseline["environment"]["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"]).resolve()
    replacements = []
    for group in ("additional_libraries", "configuration_files"):
        external = {Path(item["path"]).name: item for item in baseline.get(group, [])
                    if not Path(item["path"]).resolve().is_relative_to(baseline_root)}
        for index, item in enumerate(profile.get(group, [])):
            path = Path(item["path"]).resolve()
            replacement = external.get(path.name)
            if replacement is None or path.is_relative_to(candidate_root):
                continue
            target = Path(replacement["path"])
            if hashlib.sha256(target.read_bytes()).hexdigest() != replacement["sha256"]:
                raise ValueError(f"Qualified deployment artifact changed: {target}")
            if item != replacement:
                replacements.append(dict(previous=item, qualified=replacement))
                profile[group][index] = dict(replacement)
    profile["deployment_artifact_alignment"] = replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path,
                        help="Run a frozen DSpark-private adapter without modifying shared sources")
    parser.add_argument("--stream-filter", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=("native-bounded", "profile"),
        default="native-bounded",
        help="profile preserves the qualified baseline protocol and accumulated candidate flags",
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    storage = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving")
    profile = json.loads(args.runtime_profile.read_text())
    baseline = json.loads((workspace / "evidence/20260930_tp4_dspark/runtime-official-tp4-dspark-current.json")
                          .read_text())
    align_deployment_artifacts(profile, baseline)
    temporary = storage / "tmp" / hashlib.sha256(str(args.run).encode()).hexdigest()[:8]
    temporary.mkdir(parents=True, exist_ok=False)
    environment = profile["environment"]
    if args.source_snapshot is None and environment.get("VLLM_HPU_DSV41_DSPARK_LARGE_HCCL") == "1":
        snapshot = args.run.with_name(args.run.name + "-capacity-source")
        subprocess.run([sys.executable, "tools/prepare_deepseek_v41_dspark_capacity_snapshot.py", str(snapshot)],
                       cwd=workspace, check=True)
        args.source_snapshot = snapshot / "source"
    for key in tuple(environment):
        if key.startswith("DUMP_") or "GRAPH_VISUALIZATION" in key or key == "VLLM_HPU_TP2_PLAN_DUMP_DIR":
            del environment[key]
    for suffix in ("MERGED_MLA", "COHESIVE_MLA", "FEATURE_SILU", "PARTITION_RADIX_SAMPLING"):
        environment["VLLM_HPU_DSV41_DSPARK_" + suffix] = "0"
    if args.protocol == "native-bounded":
        environment.update(
            VLLM_HPU_DSV41_DSPARK_NATIVE_SAMPLED_PROTOCOL="1", VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR="1"
        )
    if args.stream_filter:
        environment["VLLM_HPU_DSV41_DSPARK_STREAM_FILTER_SAMPLING"] = "1"
    environment["TMPDIR"] = str(temporary)
    native_root = Path(environment["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
    for group in ("additional_libraries", "configuration_files"):
        for artifact in profile.get(group, []):
            installed = native_root / Path(artifact["path"]).name
            if installed.is_file() and hashlib.sha256(installed.read_bytes()).hexdigest() == artifact["sha256"]:
                artifact["path"] = str(installed)
    frozen_profile = args.run.with_name(args.run.name + "-profile.json")
    if frozen_profile.exists():
        raise FileExistsError("Use a new service evidence path")
    frozen_profile.write_text(json.dumps(profile, indent=2) + "\n")
    launch = [
        sys.executable,
        "tools/run_deepseek_v41.py",
        "--devices",
        "4",
        "--modules",
        "2,6,7,3",
        "--cpu-conflict-policy",
        "relocate-or-measure",
        "--preferred-cpus",
        "10-19,38-47",
        "--min-host-available-gib",
        "350",
        "--lock-dir",
        str(workspace.parent / "locks"),
        "--secondary-lock-dir",
        "/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks",
        "--runtime-profile",
        str(frozen_profile),
        "--engine-source",
        str(workspace.parent / "builds/dsv41-tp4-dspark-main-v1/engine"),
        "--recipe-cache-dir",
        str(storage / "official-native-repair-cache"),
        "--raw-profiler",
        str(args.run),
        "--",
        sys.executable,
        "-m",
        "vllm_gaudi.entrypoints.deepseek_v41",
        "/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2",
        "--tensor-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "18581",
        "--max-model-len",
        "1048576",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "32",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "1.0",
        "--num-gpu-blocks-override",
        "8193",
        "--additional-config",
        json.dumps({"dsv41_native_warmup_searches": [512, 1024, 32768, 65536]}),
    ]
    if args.source_snapshot is not None:
        slot = launch.index("--recipe-cache-dir")
        launch[slot:slot] = ["--source-snapshot", str(args.source_snapshot.resolve())]
    child_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DUMP_") and "GRAPH_VISUALIZATION" not in key and key != "VLLM_HPU_TP2_PLAN_DUMP_DIR"
    }
    launcher = None
    try:
        with args.run.with_name(args.run.name + "-launcher.log").open("xb") as log:
            launcher = subprocess.Popen(
                launch, cwd=workspace, env=child_environment, stdout=log, stderr=subprocess.STDOUT
            )
            client = [
                sys.executable,
                "tools/qualify_deepseek_v41_dspark_batch.py",
                "--run",
                str(args.run),
                "--launcher-pid",
                str(launcher.pid),
                "--request",
                str(workspace / "evidence/20260930_tp4_dspark/official-seed42-16k-request.json"),
                "--baseline-output",
                "/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1/serving-official-mme-router-12/"
                "official-16k-eos-diagnostic/token_ids.json",
                "--url",
                "http://127.0.0.1:18581",
                "--modules",
                "2,6,7,3",
            ]
            with args.run.with_name(args.run.name + "-client.log").open("xb") as client_log:
                subprocess.run(
                    client,
                    cwd=workspace,
                    env=child_environment,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            launcher.wait(timeout=120)
    finally:
        if launcher is not None and launcher.poll() is None:
            # The launcher owns and retires only the group recorded in this
            # run. Its handler releases card leases after the worker retires.
            launcher.terminate()
            launcher.wait(timeout=120)
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
