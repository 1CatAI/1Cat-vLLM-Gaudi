# SPDX-License-Identifier: Apache-2.0
"""Fixed leased real-input DSpark sampled-control gate; no service restart."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import shutil
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=(
            "full_repair",
            "large_hccl_capability",
            "full_hccl",
            "global_bounded_hccl",
            "global_bounded_hccl_filter",
            "global_bounded_hccl_lane",
            "global_bounded_hccl_fused",
            "global_bounded_hccl_journal",
            "bounded_target_exact_draft",
            "bounded_target_exact_draft_bf16",
            "bf16_hccl_head",
            "partition",
            "stream_filter",
            "bf16_head",
            "framework_head",
            "journal_coordinates",
            "nucleus_mass",
            "full_main",
            "global_bounded",
            "local_bounded",
            "round_input",
            "draft_body",
            "stochastic_only",
        ),
        required=True,
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path,
                        help="Immutable private capability source; shared C1 sources remain unchanged")
    parser.add_argument("--modules", default="2,6,7,3")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--queued-device-window", action="store_true",
                        help="Match production queue occupancy with a common actual native producer")
    parser.add_argument("--original-parent-head-norm", action="store_true",
                        help="Keep the timing parent on its original production norm implementation")
    parser.add_argument("--head-input-boundary", action="store_true",
                        help="Exercise the maintained BF16 norm value boundary in both projection arms")
    parser.add_argument("--production-plan", action="store_true",
                        help="Capture the serving graph without producer-debug copy side effects")
    parser.add_argument(
        "--repair-journal",
        action="store_true",
        help="Also test full native repair of overwritten Target, draft and cursor state",
    )
    parser.add_argument("--weighted-draft-ab", action="store_true",
                        help="Keep Target K64 and compare weighted draft nucleus against exact full q")
    parser.add_argument("--static-weighted-ab", action="store_true")
    parser.add_argument("--sparse-weighted-ab", action="store_true",
                        help="Exact block-bound skip vs existing weighted native probability protocol")
    parser.add_argument("--vocab-cdf-proof", type=Path)
    parser.add_argument("--vocab-cdf-ab", action="store_true", help="Exact q; blocked vocabulary-order proposal draw")
    parser.add_argument("--preserve-production-cdf", action="store_true",
                        help="Validate full repair with the installed production proposal draw map")
    parser.add_argument("--record-readback-ab", action="store_true")
    parser.add_argument("--protocol-writeback-ab", action="store_true",
                        help="Same production sampler; move three state writes into native tail")
    parser.add_argument("--journal-batch-direct-ab", action="store_true",
                        help="Direct typed journal outputs without packed slicing")
    parser.add_argument("--journal-batch-ab", action="store_true",
                        help="Same production protocol; eight-source byte snapshots")
    parser.add_argument("--batch-staging-ab", action="store_true",
                        help="Single transfer manifest in the isolated bridge")
    parser.add_argument("--mtp-cache-ab", action="store_true", help="Exact load-time draft weight banks")
    parser.add_argument("--mtp-fp8-ab", action="store_true", help="Same official protocol; N128 draft FP8 body only")
    parser.add_argument("--mtp-teacher-proof", type=Path)
    parser.add_argument("--bounded-draft-ab", action="store_true")
    args = parser.parse_args()
    if (args.mtp_fp8_ab or args.mtp_cache_ab or args.batch_staging_ab) and args.weighted_draft_ab:
        parser.error("Measure one independent change")
    if args.bounded_draft_ab and (args.mtp_fp8_ab or args.mtp_cache_ab or args.batch_staging_ab
                                 or args.weighted_draft_ab
                                 or args.sparse_weighted_ab or args.static_weighted_ab):
        parser.error("Bounded draft selection is an independent change")
    workspace = Path(__file__).resolve().parents[1]
    storage = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving")
    profile = json.loads(args.runtime_profile.read_text())
    production_environment = dict(profile["environment"])
    for key in tuple(profile["environment"]):
        if key.startswith("DUMP_") or "GRAPH_VISUALIZATION" in key or key == "VLLM_HPU_TP2_PLAN_DUMP_DIR":
            del profile["environment"][key]
    for suffix in ("MERGED_MLA", "COHESIVE_MLA", "FEATURE_SILU", "NATIVE_FULL_REPAIR", "NATIVE_SAMPLED_PROTOCOL"):
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_" + suffix] = "0"
    hccl_bounded = args.candidate in (
        "global_bounded_hccl", "global_bounded_hccl_filter", "global_bounded_hccl_lane",
        "global_bounded_hccl_fused", "global_bounded_hccl_journal", "bounded_target_exact_draft",
        "bounded_target_exact_draft_bf16"
    )
    if args.candidate in ("large_hccl_capability", "full_hccl", "bf16_hccl_head") or hccl_bounded:
        for suffix in ("LARGE_HCCL", "FULL_HCCL_MAIN"):
            profile["environment"]["VLLM_HPU_DSV41_DSPARK_" + suffix] = "1"
        if args.source_snapshot is None:
            # Reuse the proven private capacity adapter. The maintained C1
            # communicator and its default bounded guard stay unchanged.
            snapshot = args.run.with_name(args.run.name + "-capacity-source")
            subprocess.run([
                sys.executable, "tools/prepare_deepseek_v41_dspark_capacity_snapshot.py",
                str(snapshot),
            ], cwd=workspace, check=True)
            args.source_snapshot = snapshot / "source"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_PARTITION_RADIX_SAMPLING"] = (
        "1" if args.candidate == "partition" else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_REPAIR"] = (
        "1"
        if hccl_bounded or args.candidate
        in ("full_repair", "bf16_head", "nucleus_mass", "full_main", "global_bounded",
            "global_bounded_hccl", "local_bounded", "round_input")
        else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_STREAM_FILTER_SAMPLING"] = (
        "1" if args.candidate in ("stream_filter", "global_bounded_hccl_filter") else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_JOURNAL_COORDINATES"] = (
        "1" if args.candidate == "journal_coordinates" else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"] = "1" if args.candidate == "nucleus_mass" else "0"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING"] = (
        "1" if args.candidate == "global_bounded" or hccl_bounded else "0"
    )
    if args.candidate in ("global_bounded", "local_bounded"):
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_SAMPLING_WIDTH"] = "256"
    if hccl_bounded:
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_SAMPLING_WIDTH"] = "64"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_LANE_CANDIDATES"] = (
        "1" if args.candidate in ("global_bounded_hccl_lane", "global_bounded_hccl_fused",
                                  "global_bounded_hccl_journal", "bounded_target_exact_draft",
                                  "bounded_target_exact_draft_bf16") else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_FUSED_BOUNDED_NUCLEUS"] = (
        "1" if args.candidate in ("global_bounded_hccl_fused", "global_bounded_hccl_journal",
                                  "bounded_target_exact_draft", "bounded_target_exact_draft_bf16") else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_JOURNAL_COPY"] = (
        "1" if args.candidate in ("global_bounded_hccl_journal", "bounded_target_exact_draft",
                                  "bounded_target_exact_draft_bf16") else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_EXACT_DRAFT_SAMPLING"] = (
        "1" if args.candidate in ("bounded_target_exact_draft", "bounded_target_exact_draft_bf16") else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_WEIGHTED_DRAFT_NUCLEUS"] = (
        "1" if args.weighted_draft_ab or (args.sparse_weighted_ab or args.static_weighted_ab) else "0"
    )
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_WEIGHTED_SPARSE_BINS"] = "1" if args.sparse_weighted_ab else "0"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_WEIGHTED_STATIC_MASK"] = "1" if args.static_weighted_ab else "0"
    if args.weighted_draft_ab or (args.sparse_weighted_ab or args.static_weighted_ab):
        if args.candidate != "bounded_target_exact_draft":
            raise ValueError("Weighted draft uses an unchanged BF16 head profile and the exact-draft parent")
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_EXACT_DRAFT_SAMPLING"] = "0"
    if args.candidate in ("global_bounded_hccl_journal", "bounded_target_exact_draft",
                          "bounded_target_exact_draft_bf16"):
        # Serving records fallback causes; its certificate is a real state
        # allocation, so correctness must exercise the same capture contract.
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_COVERAGE_AUDIT"] = "1"
    if args.candidate in ("global_bounded_hccl_filter", "global_bounded_hccl_lane", "global_bounded_hccl_fused",
                          "global_bounded_hccl_journal", "bounded_target_exact_draft",
                          "bounded_target_exact_draft_bf16"):
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_CONSUMED_TARGET_CERTIFICATE"] = "1"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY"] = "1" if args.head_input_boundary else "0"
    if (args.mtp_fp8_ab or args.mtp_cache_ab or args.batch_staging_ab or args.bounded_draft_ab
            or args.vocab_cdf_ab or args.record_readback_ab or args.journal_batch_ab or args.journal_batch_direct_ab
            or args.protocol_writeback_ab):
        if args.candidate != "global_bounded_hccl_journal" or args.mtp_fp8_ab and args.mtp_teacher_proof is None:
            parser.error("MTP-only gate requires the production bounded/journal protocol and teacher proof")
        # The draft arithmetic is the sole experimental change. Preserve the
        # serving sampler, including weighted q, in both captured replay arms.
        for suffix in (
            "STOCHASTIC_ONLY", "WEIGHTED_DRAFT_NUCLEUS", "EXACT_DRAFT_SAMPLING",
            "CONSUMED_TARGET_CERTIFICATE", "FUSED_BOUNDED_NUCLEUS", "LANE_CANDIDATES",
            "JOURNAL_COPY", "COVERAGE_AUDIT", "HEAD_INPUT_BOUNDARY",
            "GLOBAL_BOUNDED_SAMPLING", "SAMPLING_WIDTH",
        ):
            key = "VLLM_HPU_DSV41_DSPARK_" + suffix
            if key not in production_environment:
                parser.error("MTP-only gate needs the explicit production sampling profile: " + key)
            profile["environment"][key] = production_environment[key]
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_MTP_FP8"] = "1" if args.mtp_fp8_ab else "0"
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_MTP_CACHE"] = "1" if args.mtp_cache_ab else "0"
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_BATCH_INPUT_STAGING"] = "1" if args.batch_staging_ab else "0"
        if args.bounded_draft_ab:
            if production_environment["VLLM_HPU_DSV41_DSPARK_WEIGHTED_DRAFT_NUCLEUS"] != "1":
                parser.error("Bounded draft gate needs the qualified weighted-q serving parent")
            profile["environment"]["VLLM_HPU_DSV41_DSPARK_WEIGHTED_DRAFT_NUCLEUS"] = "0"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_RECORD_READBACK"] = "1" if args.record_readback_ab else "0"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF"] = (
        production_environment.get("VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF", "0")
        if args.record_readback_ab or args.journal_batch_ab or args.journal_batch_direct_ab
        or args.protocol_writeback_ab or args.preserve_production_cdf else
        "1" if args.vocab_cdf_ab else "0")
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_JOURNAL_BATCH"] = "1" if args.journal_batch_ab else "0"
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_JOURNAL_BATCH_DIRECT"] = (
        "1" if args.journal_batch_direct_ab else "0")
    profile["environment"]["VLLM_HPU_DSV41_DSPARK_PROTOCOL_WRITEBACK"] = (
        "1" if args.protocol_writeback_ab else "0")
    temporary = storage / "tmp" / hashlib.sha256(str(args.run).encode()).hexdigest()[:8]
    temporary.mkdir(parents=True, exist_ok=False)
    profile["environment"]["TMPDIR"] = str(temporary)
    profile_path = args.run.with_name(args.run.name + "-profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    if profile_path.exists():
        raise FileExistsError("Use a new gate artifact path")
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    os.sched_setaffinity(0, set(range(10, 20)) | set(range(38, 48)))
    invocation = [
        sys.executable,
        "tools/run_deepseek_v41.py",
        "--devices",
        "4",
        "--modules",
        args.modules,
        "--cpu-conflict-policy",
        "relocate-or-measure",
        "--preferred-cpus",
        "10-19,38-47",
        "--lock-dir",
        str(workspace.parent / "locks"),
        "--secondary-lock-dir",
        "/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks",
        "--runtime-profile",
        str(profile_path),
        "--engine-source",
        str(workspace.parent / "builds/dsv41-tp4-dspark-main-v1/engine"),
        "--recipe-cache-dir",
        str(storage / "sampled-control-component-cache"),
        str(args.run),
        "--",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "4",
        "tools/check_deepseek_v41_sampled_native_control.py",
        "--prepared",
        "/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2",
        "--fixtures",
        "/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1/production-c6-request-fixtures-02-with-history",
        "--samples",
        "6",
        "--probability-diagnostic",
        "--producer-bf16-max-ulp",
        "2",
    ]
    if args.candidate == "large_hccl_capability":
        invocation[invocation.index("tools/check_deepseek_v41_sampled_native_control.py")] = (
            "tools/check_deepseek_v41_large_hccl_native.py")
        invocation = invocation[:-3]  # Sampling-specific diagnostic/ULP flags.
    elif args.candidate == "framework_head":
        invocation += ["--framework-head-ab"]
    elif args.candidate == "full_hccl":
        invocation += ["--native-full", "--native-full-main", "--stock-hccl-full-ab"]
    elif args.candidate == "bf16_hccl_head":
        invocation += ["--native-full", "--native-full-main", "--stock-hccl-full-ab",
                       "--native-parent", "--bf16-head-ab"]
    elif args.candidate == "stochastic_only":
        invocation += ["--stochastic-only-ab"]
    elif args.candidate == "draft_body":
        invocation += ["--draft-body-ab"]
    elif args.candidate == "round_input":
        invocation += ["--native-full", "--native-full-main", "--native-parent", "--round-input-ab"]
    elif args.candidate == "local_bounded":
        invocation += ["--native-parent", "--repair-journal", "--local-bounded-ab"]
    elif args.candidate == "global_bounded":
        invocation += ["--native-parent", "--repair-journal", "--global-bounded-ab"]
    elif hccl_bounded:
        invocation += ["--native-parent", "--repair-journal", "--global-bounded-ab", "--stock-hccl-bounded-ab"]
        if args.candidate == "bounded_target_exact_draft_bf16":
            invocation.append("--bf16-head-ab")
    elif args.candidate == "full_main":
        invocation += ["--native-full", "--native-full-main"]
    elif args.candidate == "nucleus_mass":
        invocation += ["--native-full", "--native-parent", "--nucleus-mass-ab"]
    elif args.candidate == "bf16_head":
        invocation += ["--native-full", "--native-parent", "--bf16-head-ab"]
    else:
        invocation += ["--native-full"] if args.candidate == "full_repair" else ["--repair-journal", "--native-parent"]
    if args.candidate == "full_repair" and args.repair_journal:
        invocation += ["--repair-journal"]
    if args.preserve_production_cdf:
        invocation.append("--production-stochastic")
    if args.candidate == "journal_coordinates":
        invocation += ["--journal-coordinate-ab"]
    if args.candidate == "stream_filter":
        invocation += ["--selection-flag", "STREAM_FILTER_SAMPLING"]
    if args.source_snapshot is not None:
        slot = invocation.index("--recipe-cache-dir")
        invocation[slot:slot] = ["--source-snapshot", str(args.source_snapshot.resolve())]
    if args.weighted_draft_ab:
        invocation.append("--weighted-draft-ab")
    if args.sparse_weighted_ab:
        invocation.append("--sparse-weighted-ab")
    if args.static_weighted_ab:
        invocation.append("--static-weighted-ab")
    if args.vocab_cdf_ab:
        if args.vocab_cdf_proof is None:
            parser.error("Proposal CDF needs its fixed-prefix native probability proof")
        invocation += ["--vocab-cdf-ab", "--vocab-cdf-proof", str(args.vocab_cdf_proof.resolve())]
    if args.record_readback_ab:
        invocation.append("--record-readback-ab")
    if args.protocol_writeback_ab:
        invocation.append("--protocol-writeback-ab")
    if args.journal_batch_direct_ab:
        invocation.append("--journal-batch-direct-ab")
    if args.journal_batch_ab:
        invocation.append("--journal-batch-ab")
    if args.batch_staging_ab:
        invocation.append("--batch-staging-ab")
    if args.mtp_cache_ab:
        invocation.append("--mtp-cache-ab")
    if args.mtp_fp8_ab:
        invocation.append("--mtp-fp8-ab")
        if args.mtp_teacher_proof is not None:
            invocation.extend(("--mtp-teacher-proof", str(args.mtp_teacher_proof.resolve())))
    if args.bounded_draft_ab:
        invocation.append("--bounded-draft-ab")
    if args.original_parent_head_norm:
        if not args.head_input_boundary or args.candidate != "bounded_target_exact_draft_bf16":
            raise ValueError("Original-parent norm comparison requires the explicit BF16 head candidate")
        invocation.append("--original-parent-head-norm")
    if args.correctness_only:
        invocation.append("--correctness-only")
    if args.queued_device_window:
        invocation.append("--queued-device-window")
    if args.production_plan and "--probability-diagnostic" in invocation:
        invocation.remove("--probability-diagnostic")
        slot = invocation.index("--producer-bf16-max-ulp")
        del invocation[slot:slot + 2]
    try:
        worker = subprocess.Popen(invocation, cwd=workspace)
        retired = False
        while worker.poll() is None:
            reports = list(args.run.glob("sampled-control-rank*.json"))
            try:
                rows = [json.loads(path.read_text()) for path in reports]
                failed = any(row.get("status") == "failed" for row in rows)
                done = len(rows) == 4 and all(
                    row.get("status") == ("correctness_passed" if args.correctness_only else "passed")
                    and len(row.get("checks", [])) >= 3
                    and (args.correctness_only or len(row.get("rounds", [])) == 6) for row in rows)
            except json.JSONDecodeError:
                failed = done = False
            identity = args.run / "process.json"
            if not retired and (failed or done) and identity.exists():
                process = json.loads(identity.read_text())
                pid, pgid = process["pid"], process["pgid"]
                command = Path(f"/proc/{pid}/cmdline")
                if command.exists() and os.getpgid(pid) == pgid:
                    words = command.read_bytes().split(b"\0")
                    if any(b"check_deepseek_v41_sampled_native_control.py" in word or
                           b"check_deepseek_v41_large_hccl_native.py" in word for word in words):
                        os.killpg(pgid, signal.SIGTERM)
                        retired = True
            time.sleep(1)
        if worker.returncode and not retired:
            raise subprocess.CalledProcessError(worker.returncode, invocation)
    finally:
        # The leased launcher retires its complete process group before
        # returning. Remove only this run's scratch, retaining all evidence.
        shutil.rmtree(temporary)
    ranks = [json.loads((args.run / f"sampled-control-rank{rank}.json").read_text()) for rank in range(4)]
    if any(not row["capability_passed"] for row in ranks):
        raise ValueError("All ranks must pass the real-input probability/RNG/cache gate")
    if args.correctness_only:
        result = dict(candidate=args.candidate, correctness_passed=True, performance_tested=False,
                      micro_passed=False, default_enabled=False, end_to_end_tested=False,
                      native_journal_checks=[row.get("native_journal_checks", []) for row in ranks])
        (args.run / "component-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    savings = []
    for iteration in range(3):
        slowest = [
            max(
                next(
                    sample["median_device_ms"]
                    for sample in rank["rounds"]
                    if sample["iteration"] == iteration and sample["arm"] == arm
                )
                for rank in ranks
            )
            for arm in ("parent", "native")
        ]
        savings.append(slowest[0] - slowest[1])
    result = dict(
        candidate=args.candidate,
        paired_saved_ms=savings,
        saved_ms_per_round=statistics.median(savings),
        micro_passed=all(value > 0 for value in savings) and statistics.median(savings) >= 0.3,
        default_enabled=False,
        end_to_end_tested=False,
        scope="complete sampled protocol and next embedding; full next Target is outside this component",
        queued_device_window=args.queued_device_window,
        timer_boundary=("behind common actual native producer; all repair cases retained"
                        if args.queued_device_window else "drained before each replay"),
        contains_production_repair_journal=hccl_bounded or args.candidate in ("global_bounded", "local_bounded")
        or (args.candidate in ("partition", "stream_filter", "journal_coordinates") or args.repair_journal),
    )
    if args.candidate == "large_hccl_capability":
        result.update(micro_passed=False, capability_passed=True,
                      scope=ranks[0]["scope"], full_protocol_qualified=False)
    (args.run / "component-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
