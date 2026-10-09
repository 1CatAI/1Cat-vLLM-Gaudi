# SPDX-License-Identifier: Apache-2.0
"""Fixed leased production C6 component gate; no full-service startup."""

import argparse
import hashlib
import json
import os
import re
import signal
import time
import shutil
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(run, candidate, *, output=None):
    ranks = [json.loads((run / f"request-c6-rank{rank}.json").read_text()) for rank in range(4)]
    if any(row["status"] != "component_passed" for row in ranks):
        raise ValueError("All four ranks must finish correctness and native A/B")
    if any(
        not (check["outputs_and_swa_exact"] or check.get("precision_oracle_qualified", False))
        for row in ranks
        for check in row["checks"]
    ):
        raise ValueError("Real request correctness gate failed")
    savings = []
    for iteration in range(3):
        slowest = [
            max(
                next(
                    round_["median_device_ms"]
                    for round_ in rank["rounds"]
                    if round_["iteration"] == iteration and round_["arm"] == arm
                )
                for rank in ranks
            )
            for arm in range(2)
        ]
        savings.append(slowest[0] - slowest[1])
    multiplier = ranks[0]["projection_multiplier"]
    if any(row["projection_multiplier"] != multiplier for row in ranks):
        raise ValueError("Ranks disagree about the eligible production layer count")
    projected = statistics.median(savings) * multiplier
    result = dict(
        candidate=candidate,
        correctness=(
            "3 actual request cases/four ranks; isolated operator precision proof; "
            "finite native-chain outputs, internal differences recorded"
            if any(row.get("precision_oracle") for row in ranks)
            else ("3 actual fixed prefixes/four ranks: protocol, q, RNG and next embedding exact; "
                  "SWA not separately compared")
            if candidate == "deep_queue" else "3 actual request cases, four ranks, outputs and SWA exact"
        ),
        paired_slowest_rank_saved_ms=savings,
        projected_saved_ms_per_round=projected,
        micro_passed=all(value > 0 for value in savings),
        rank_local_paired_saved_ms=[row["paired_saved_ms"] for row in ranks],
        timing_rule=("Three paired slowest-rank deltas positive; "
                     "forecast>=0.3ms is a selection gate, not a measured gate"),
        default_enabled=False,
        end_to_end_tested=False,
        credited_end_to_end_ms=0,
        extrapolation=("actual16layer tail plus complete sampled protocol; one join boundary, no layer multiplier"
                       if candidate == "joined_sampled_tail" else
                       "real16/protocol through next TP embedding; no layer multiplier; Engram staging checked only"
                       if candidate == "round_input_publication" else
                       "independent real16/C5/protocol retained-queue windows; no layer multiplier or serving IPC proof"
                       if candidate == "deep_queue" else
                       "one production index owner; project only its recorded geometry class"
                       if candidate in ("threshold_selection", "candidate_keys")
                       else "real16 source20 chain; project only ratio-one Reindex owners; source producer unchanged"
                       if candidate == "runtime_selection" else
                       "real16 source20 chain; layer projection is not a measured full-round gain"
                       if ranks[0].get("real_layers") == 16 else
                       "four-layer group; projected layer count is not a measured full-round gain"),
        projected_layer_count=ranks[0]["projected_layer_count"],
    )
    if candidate == "expert_consumer_stitch":
        markers = re.findall(r'DSPARK_EXPERT_CONSUMER_PIPELINE_READY [^\n]* chain=(\d+)',
                             (run / 'run.log').read_text())
        result['compiler_policy_hits'] = len(markers)
        result['eligible_compiler_consumer_chains'] = sum(int(value) > 0 for value in markers)
        result['compiler_consumer_placement_proved'] = False
        result['micro_passed'] &= bool(result['eligible_compiler_consumer_chains'])
        result['compiler_gate_limit'] = 'Eligible chain is not proof of final SRAM placement'
    unchanged_w2 = candidate == "w2_reduce_n256" and all(
        check["outputs_and_swa_exact"] for rank in ranks for check in rank["checks"])
    if unchanged_w2:
        result["micro_timing_passed"] = result["micro_passed"]
        result["teacher_forced_acceptance_qualified"] = False
        result["teacher_not_repeated_reason"] = (
            "Arithmetic unchanged: same products/scales, per-route RNE and route order; "
            "only vector layout/loads change. "
            "All actual chain hidden/pre/cache outputs byte-exact, accepted C1 operator gate passed.")
    elif candidate in ("qk_flat", "qk_flat_direct", "stream_exp", "mhc_post_stats", "main_single_bank", "coherent_swa",
                     "c1_dense_chain",
                     "runtime_selection", "mhc_producer", "mhc_native_dependencies",
                     "index_query_local", "norm_roundtrip",
                     "input_fp8", "output_fp8", "output_layout", "scaled_w13", "w2_k_pipeline", "w2_ready_scale",
                     "w2_reduce_n256",
                     "mhc_high_plane", "mhc_deferred", "control_fp8", "control_fp8_pair",
                     "input_norm", "mhc_statistics", "dense_kn", "dense_bits12",
                     "router_shared_fused", "hw_dense", "hw_dense_fused", "peer_post_collapse"):
        result["micro_timing_passed"] = result["micro_passed"]
        result["micro_passed"] = False
        result["teacher_forced_acceptance_qualified"] = False
        result["decision"] = (
            "Timing passed; whole-model fixed-prefix acceptance required before batch admission"
            if result["micro_timing_passed"] else
            "OFF: native timing gate failed; do not advance to whole-model acceptance or serving"
        )
    (output or run / "component-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=(
            "recipe_constants",
            "silu_decode_affine", "split_scale_planes", "w13_k_pipeline",
                                  "w2_k_pipeline", "w2_ready_scale", "w2_reduce_n256", "ordered_peer_sum",
                                  "peer_post_collapse", "mhc_gate_packet",
            "native_page_coalesce", "expert_consumer_stitch",
                                          "qk_flat", "qk_flat_direct", "stream_exp",
            "mhc_post_stats",
            "main_single_bank",
            "coherent_swa",
            "main_adjacent_pv",
            "swa_source_reuse",
            "joined_sampled_tail",
            "deep_queue",
            "round_input_publication",
            "silu_full_rows",
            "compressor_sequence",
            "pair_silu",
            "swa_cache",
            "split_feature_silu",
            "layer_main_split",
            "layer_main_reuse",
            "compact_stream_mme",
            "unpaired_feature_silu",
            "silu_scalar_cache",
            "q_bf16_rope",
            "query_norm",
            "full_row",
            "fp4_table",
            "row_cache",
            "shared_finalize",
            "wide_codec",
            "cohesive_mla",
            "merged_mla",
            "feature_silu",
            "w2_three_routes",
            "mhc_overlap", "mhc_producer", "mhc_native_dependencies", "index_query_local",
            "native_target_input",
            "runtime_selection",
            "c1_dense_chain",
            "norm_roundtrip", "recipe_constants", "silu_decode",
                                  "silu_decode_affine", "split_scale_planes", "w13_k_pipeline",
                                  "w2_k_pipeline", "w2_ready_scale", "w2_reduce_n256", "ordered_peer_sum",
                                  "peer_post_collapse", "mhc_gate_packet",
                                  "native_page_coalesce", "expert_consumer_stitch",
                                          "qk_flat", "qk_flat_direct", "stream_exp",
                                          "explicit_steps", "scaled_w13",
                                  "control_fp8", "control_fp8_pair", "moe_peer_post", "mhc_deferred", "mhc_high_plane",
                                  "input_fp8", "output_fp8", "output_layout",
            "input_norm", "n512_decode", "dense_kn", "dense_bits12", "scheduled_peer", "rope_coherent",
            "router_shared", "router_shared_fused", "hw_dense", "hw_dense_fused",
            "dense_restore",
            "peer_post_norm",
            "router_bf16",
            "silu_unroll",
            "physical_silu",
            "physical_role_silu",
            "exp_pv",
            "wo_handoff",
            "resident_constants",
            "shared_gate_up",
            "shared_fp8",
            "shared_prequant",
            "kv_publish",
            "pv_rope",
            "qkv_publish",
            "q_prologue",
            "mhc_mme_epilogue", "mhc_statistics",
            "hash_mla",
            "mla_coord_cache",
            "shared_kv_mme",
            "compact_kv_mme",
            "group_pipeline",
            "channel_silu",
            "w13_unroll",
            "main_mirror",
            "decoded_publish",
            "scale_cache",
            "silu_affine",
            "threshold_selection",
            "candidate_keys",
            "affine_route",
            "pair_pv",
            "stacked_pv",
            "batch6_kv",
            "transpose_sat",
            "k64_partition",
            "unpaired_w13",
            "merge_cache",
            "mhc_weight_reuse", "mhc_control_tiles",
        ),
        required=True,
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--modules", default="2,6,7,3")
    parser.add_argument("--preferred-cpus", default="10-19,38-47")
    parser.add_argument("--min-host-available-gib", type=float, default=100,
                        help="Host budget for the retained component working set; full serving uses its own budget")
    parser.add_argument("--group", type=int, default=5, choices=(0, 2, 5, 6))
    parser.add_argument("--real-layer-count", type=int, choices=(4, 16), default=4)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--baseline-profile", type=Path, help="Keep qualified serving precision and codec in both arms")
    parser.add_argument("--dense-bits12-proof", type=Path, help="Immutable native lane and actual-weight byte proof")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--attention-oracle", action="store_true")
    parser.add_argument("--peer-input-oracle", action="store_true")
    parser.add_argument("--constant-oracle", action="store_true")
    parser.add_argument("--c1-projection-oracle", action="store_true")
    parser.add_argument("--selection-oracle", action="store_true")
    parser.add_argument("--fixed-prefix-acceptance", action="store_true")
    parser.add_argument("--stack-candidates", nargs="*", default=[],
                        choices=("norm_roundtrip", "w13_k_pipeline", "mhc_gate_packet",
                                 "w2_reduce_n256", "index_query_local"))
    parser.add_argument("--precision-proof", type=Path)
    parser.add_argument("--input-quant-remat", action="store_true")
    args = parser.parse_args()
    if args.stack_candidates and (not args.fixed_prefix_acceptance or args.candidate in args.stack_candidates
                                  or len(set(args.stack_candidates)) != len(args.stack_candidates)):
        parser.error("Stack distinct qualified candidates only in the combined full40 teacher gate")
    if args.input_quant_remat and args.candidate != "input_fp8":
        parser.error("Rematerialized producer requires the input_fp8 complete chain")
    run = args.run.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(run, args.candidate), indent=2))
        return
    workspace = Path(__file__).resolve().parents[1]
    root = workspace.parent
    storage = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving")
    fixtures = (
        Path("/opt/ssd960/builds/dsv41-tp4-dspark-round-chain-v1") / "production-c6-request-fixtures-02-with-history"
    )
    profile_path = (args.runtime_profile or args.baseline_profile
                    or workspace / "evidence/20260930_tp4_dspark/runtime-request-c6-batch-v29.json")
    profile = json.loads(profile_path.read_text())
    if args.candidate == "dense_bits12":
        if args.dense_bits12_proof is None:
            parser.error("Lossless dense component requires --dense-bits12-proof")
        proof_path = args.dense_bits12_proof.resolve()
        proof = json.loads(proof_path.read_text())
        if proof.get("status") != "WEIGHT_BYTES_FUNCTIONAL_ONLY":
            parser.error("The native lane/weight binding must pass before producer-consumer timing")
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_DENSE_BITS12_MAP"] = str(proof_path)
        profile.setdefault("configuration_files", []).append(dict(
            path=str(proof_path), sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest()))
    for key in tuple(profile["environment"]):
        if key.startswith("DUMP_") or "GRAPH_VISUALIZATION" in key or key == "VLLM_HPU_TP2_PLAN_DUMP_DIR":
            del profile["environment"][key]
    temporary = storage / "tmp" / hashlib.sha256(str(run).encode()).hexdigest()[:8]
    profile["environment"]["TMPDIR"] = str(temporary)
    # This component contains Target only. Avoid accidentally enabling a
    # draft protocol, exporter or another accumulating candidate.
    for flag in (
        "MHC_POST_STATS",
        "MAIN_SINGLE_BANK",
        "CONTROL_FP8", "CONTROL_FP8_PAIR",
        "SCALED_W13", "MOE_PEER_POST", "INPUT_QUANT_REMAT",
        "EXPLICIT_STEPS",
        "INPUT_FP8",
        "OUTPUT_LAYOUT",
        "OUTPUT_FP8",
        "MHC_HIGH_PLANE",
        "MHC_DEFERRED",
        "RECIPE_CONSTANTS",
        "MHC_GATE_PACKET",
        "W13_K_PIPELINE", "W2_K_PIPELINE", "W2_READY_SCALE", "W2_REDUCE_N256",
        "EXPERT_CONSUMER_STITCH",
        "QK_FLAT",
        "QK_FLAT_DIRECT",
        "STREAM_EXP",
        "SPLIT_SCALE_PLANES",
        "SILU_DECODE_AFFINE",
        "SILU_DECODE",
        "SILU_FULL_ROWS",
        "COMPRESSOR_SEQUENCE",
        "SWA_CACHE",
        "SPLIT_FEATURE_SILU",
        "LAYER_MAIN_SPLIT",
        "LAYER_MAIN_REUSE",
        "COMPACT_STREAM_MME",
        "UNPAIRED_FEATURE_SILU",
        "SILU_SCALAR_CACHE",
        "Q_BF16_ROPE",
        "QUERY_NORM",
        "FULL_ROW",
        "FP4_TABLE",
        "ROW_CACHE",
        "SHARED_FINALIZE",
        "WIDE_CODEC",
        "NATIVE_SAMPLED_PROTOCOL",
        "ROUND_INPUT_PUBLICATION",
        "NATIVE_PAGE_COALESCE",
        "COHESIVE_MLA",
        "MERGED_MLA",
        "FEATURE_SILU",
        "W2_THREE_ROUTES",
        "PEER_POST_NORM",
        "SILU_UNROLL",
        "PHYSICAL_SILU",
        "PHYSICAL_ROLE_SILU",
        "KV_PUBLISH",
        "QKV_PUBLISH",
        "Q_PROLOGUE",
        "C1_DENSE_CHAIN",
        "INPUT_NORM",
        "N512_DECODE",
        "DENSE_KN",
        "SCHEDULED_PEER",
        "ROPE_COHERENT",
        "ROUTER_SHARED",
        "ROUTER_SHARED_FUSED",
        "HW_DENSE",
        "HW_DENSE_FUSED_QUANT",
        "MHC_OVERLAP", "MHC_PRODUCER",
        "NATIVE_TARGET_INPUT",
        "RUNTIME_SELECTION",
        "MHC_MME_EPILOGUE",
        "HASH_MLA",
        "MLA_COORD_CACHE",
        "SHARED_KV_MME",
        "COMPACT_KV_MME",
        "GROUP_PIPELINE",
        "CHANNEL_SILU",
        "W13_UNROLL",
        "MAIN_MIRROR",
        "DECODED_PUBLISH",
        "SCALE_CACHE",
        "SILU_AFFINE",
        "SHARED_FP8",
        "SHARED_PREQUANT",
        "THRESHOLD_SELECTION",
        "CANDIDATE_KEYS",
        "AFFINE_ROUTE",
        "PAIR_PV",
        "STACKED_PV",
        "BATCH6_KV",
        "TRANSPOSE_SAT",
        "K64_PARTITION",
        "UNPAIRED_W13",
        "MERGE_CACHE",
        "MHC_WEIGHT_REUSE", "MHC_CONTROL_TILES",
        "ROUTER_BF16",
        "PV_ROPE",
        "EXP_PV",
        "WO_HANDOFF",
        "RESIDENT_CONSTANTS",
        "DENSE_BITS12",
    ):
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_" + flag] = "0"
    profile["environment"]["VLLM_HPU_DSV41_BF16_ROUTER_GATE"] = "0"
    profile["environment"]["VLLM_HPU_DSV41_SHARED_GATE_UP"] = "0"
    if args.candidate in ("dense_restore", "qkv_publish"):
        profile["environment"]["VLLM_HPU_DSV41_ATTN_DENSE_FP8"] = "0"
    if args.baseline_profile:
        if args.candidate not in ("mhc_post_stats", "main_single_bank", "coherent_swa", "mhc_overlap",
                                  "mhc_producer", "mhc_native_dependencies", "index_query_local",
                                  "native_target_input", "runtime_selection",
                                  "c1_dense_chain",
                                  "norm_roundtrip", "recipe_constants", "silu_decode",
                                  "silu_decode_affine", "split_scale_planes", "w13_k_pipeline",
                                  "w2_k_pipeline", "w2_ready_scale", "w2_reduce_n256", "ordered_peer_sum",
                                  "peer_post_collapse", "mhc_gate_packet",
                                  "native_page_coalesce", "expert_consumer_stitch",
                                          "qk_flat", "qk_flat_direct", "stream_exp",
                                          "explicit_steps", "scaled_w13",
                                  "control_fp8", "control_fp8_pair", "moe_peer_post", "mhc_deferred", "mhc_high_plane",
                                  "input_fp8", "output_fp8", "output_layout",
                                  "input_norm", "mhc_statistics", "n512_decode",
                                  "dense_kn", "dense_bits12", "scheduled_peer",
                                  "rope_coherent", "router_shared", "router_shared_fused", "hw_dense", "hw_dense_fused",
                                  "main_adjacent_pv", "swa_source_reuse",
                                  "joined_sampled_tail", "deep_queue", "round_input_publication", "silu_full_rows",
                                  "compressor_sequence", "pair_silu",
                                  "swa_cache",
                                  "split_feature_silu", "layer_main_split",
                                  "layer_main_reuse",
                                  "compact_stream_mme",
                                  "unpaired_feature_silu",
                                  "q_bf16_rope", "query_norm",
                                  "silu_scalar_cache", "full_row",
                                  "fp4_table", "row_cache",
                                  "shared_finalize", "wide_codec",
                                  "threshold_selection", "candidate_keys", "affine_route", "router_bf16",
                                  "pair_pv",
            "stacked_pv",
            "batch6_kv", "transpose_sat", "k64_partition", "q_prologue", "mhc_mme_epilogue", "mhc_statistics",
            "unpaired_w13", "merge_cache", "mhc_weight_reuse", "mhc_control_tiles", "physical_role_silu", "exp_pv",
            "wo_handoff", "resident_constants", "decoded_publish"):
            raise ValueError("Qualified serving baseline currently applies to the index producer/consumer gates")
        baseline_profile = json.loads(args.baseline_profile.read_text())
        from run_deepseek_v41_sampled_serving import align_deployment_artifacts

        align_deployment_artifacts(profile, baseline_profile)
        baseline = baseline_profile["environment"]
        for name in ("DSPARK_SHARED_FP8", "DSPARK_SCALE_CACHE", "ATTN_DENSE_FP8_SIDECAR",
                     "DSPARK_THRESHOLD_SELECTION", "DSPARK_STOCHASTIC_ONLY", "DSPARK_LAYER_MAIN_SPLIT"):
            if name == "DSPARK_THRESHOLD_SELECTION" and args.candidate in ("threshold_selection", "candidate_keys"):
                continue
            if name == "DSPARK_LAYER_MAIN_SPLIT" and args.candidate in ("layer_main_split", "layer_main_reuse"):
                continue
            key = "VLLM_HPU_DSV41_" + name
            if key in baseline:
                profile["environment"][key] = baseline[key]
        profile["qualified_baseline_profile"] = dict(path=str(args.baseline_profile.resolve()),
                                                   sha256=hashlib.sha256(args.baseline_profile.read_bytes()).hexdigest())
    if args.candidate in ("peer_post_collapse", "ordered_peer_sum", "w2_ready_scale", "w2_reduce_n256", "mhc_producer",
                          "mhc_native_dependencies", "index_query_local", "router_shared_fused"):
        if not args.baseline_profile:
            raise ValueError("This candidate must compare against the qualified serving baseline")
        for suffix in ("NORM_ROUNDTRIP", "W13_K_PIPELINE", "MHC_HIGH_PLANE"):
            key = "VLLM_HPU_DSV41_DSPARK_" + suffix
            profile["environment"][key] = baseline[key]
        for suffix in ("W2_REDUCE_N256", "INDEX_QUERY_LOCAL"):
            key = "VLLM_HPU_DSV41_DSPARK_" + suffix
            if key in baseline and suffix.lower() != args.candidate:
                profile["environment"][key] = baseline[key]
        if args.candidate == "w2_reduce_n256":
            profile["environment"]["VLLM_HPU_DSV41_DSPARK_W2_REDUCE_N256"] = "1"
        if args.candidate == "w2_ready_scale":
            profile["environment"]["VLLM_HPU_DSV41_DSPARK_W2_READY_SCALE"] = "1"
    if args.input_quant_remat:
        profile["environment"]["VLLM_HPU_DSV41_DSPARK_INPUT_QUANT_REMAT"] = "1"
    admission_profile = run.with_name(run.name + "-profile.json")
    admission_profile.parent.mkdir(parents=True, exist_ok=True)
    if admission_profile.exists():
        raise FileExistsError("Use a new artifact path for each candidate gate")
    temporary.mkdir(parents=True, exist_ok=False)
    admission_profile.write_text(json.dumps(profile, indent=2) + "\n")
    from run_deepseek_v41 import cpuset

    os.sched_setaffinity(0, cpuset(args.preferred_cpus))
    invocation = [
        sys.executable,
        str(workspace / "tools/run_deepseek_v41.py"),
        "--devices",
        "4",
        "--modules",
        args.modules,
        "--cpu-conflict-policy",
        "relocate-or-measure",
        "--preferred-cpus",
        args.preferred_cpus,
        "--min-host-available-gib",
        str(args.min_host_available_gib),
        "--lock-dir",
        str(root / "locks"),
        "--secondary-lock-dir",
        "/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks",
        "--runtime-profile",
        str(admission_profile),
        "--engine-source",
        str(root / "builds/dsv41-tp4-dspark-main-v1/engine"),
        "--recipe-cache-dir",
        str(storage / ("request-c6-cache-" + run.name)),
        str(run),
        "--",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "4",
        "tools/check_deepseek_v41_request_c6_batch.py",
        "--prepared",
        "/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2",
        "--fixtures",
        str(fixtures),
        "--candidate",
        args.candidate,
        "--group",
        str(args.group),
        "--real-layer-count",
        str(args.real_layer_count),
        "--samples",
        "6",
    ]
    if args.candidate in ("joined_sampled_tail", "deep_queue", "round_input_publication"):
        snapshot = run.with_name(run.name + "-capacity-source")
        subprocess.run([sys.executable, "tools/prepare_deepseek_v41_dspark_capacity_snapshot.py", str(snapshot)],
                       cwd=workspace, check=True)
        index = invocation.index("--recipe-cache-dir")
        invocation[index:index] = ["--source-snapshot", str(snapshot / "source")]
    if args.fixed_prefix_acceptance:
        invocation.append("--fixed-prefix-acceptance")
    if args.stack_candidates:
        invocation.extend(("--stack-candidates", *args.stack_candidates))
    if args.peer_input_oracle:
        invocation.append("--peer-input-oracle")
    if args.attention_oracle:
        invocation.append("--attention-oracle")
    if args.constant_oracle:
        invocation.append("--constant-oracle")
    if args.c1_projection_oracle:
        invocation.append("--c1-projection-oracle")
    if args.candidate == "mhc_control_tiles" and not args.c1_projection_oracle and not args.precision_proof:
        invocation.append("--inline-control-oracle")
    if args.candidate == "input_norm":
        invocation.append("--inline-input-norm-oracle")
    if args.selection_oracle:
        invocation.append("--selection-oracle")
    if args.precision_proof:
        invocation.extend(("--precision-proof", str(args.precision_proof.resolve())))
    try:
        worker = subprocess.Popen(invocation, cwd=workspace)
        retired = False
        while worker.poll() is None:
            if not retired and (run / "process.json").exists():
                reports = [run / f"request-c6-rank{rank}.json" for rank in range(4)]
                try:
                    failed = any(path.exists() and json.loads(path.read_text()).get("status") == "failed"
                                 for path in reports)
                    done = all(
                        path.exists()
                        and (row := json.loads(path.read_text()))["status"]
                        == (
                            "diagnostic_completed"
                            if (args.constant_oracle or args.attention_oracle or args.c1_projection_oracle
                                or args.selection_oracle or args.peer_input_oracle
                                or args.fixed_prefix_acceptance)
                            else "component_passed"
                        )
                        and (args.constant_oracle or args.attention_oracle or args.c1_projection_oracle
                             or args.selection_oracle or args.peer_input_oracle
                             or args.fixed_prefix_acceptance or len(row["rounds"]) == 6)
                        and len(row["checks"]) >= 3
                        for path in reports
                    )
                except (json.JSONDecodeError, KeyError):
                    done, failed = False, False
                if done or failed:
                    identity = json.loads((run / "process.json").read_text())
                    pid, pgid = identity["pid"], identity["pgid"]
                    command = Path(f"/proc/{pid}/cmdline")
                    if command.exists():
                        words = command.read_bytes().split(b"\0")
                        if (
                            b"torch.distributed.run" in words
                            and b"tools/check_deepseek_v41_request_c6_batch.py" in words
                            and os.getpgid(pid) == pgid
                        ):
                            # Retire completed or failed owned peers immediately,
                            # before a blocked collective reaches its timeout.
                            # The leased launcher verifies idle cards before unlock.
                            os.killpg(pgid, signal.SIGTERM)
                            retired = True
                    else:
                        retired = True
            time.sleep(1)
        if worker.returncode and not retired:
            raise subprocess.CalledProcessError(worker.returncode, invocation)
        if args.fixed_prefix_acceptance:
            from deepseek_v41_fixed_prefix_target_gate import summarize as summarize_acceptance

            print(json.dumps(summarize_acceptance(run), indent=2))
        elif args.peer_input_oracle:
            ranks = [json.loads((run / f"request-c6-rank{rank}.json").read_text()) for rank in range(4)]
            if any(row["status"] != "diagnostic_completed" for row in ranks):
                raise ValueError("Four ranks must finish all three peer-input diagnostics")
            print(json.dumps(dict(status="diagnostic_completed", performance_measured=False)))
        elif args.constant_oracle or args.attention_oracle or args.c1_projection_oracle or args.selection_oracle:
            ranks = [json.loads((run / f"request-c6-rank{rank}.json").read_text()) for rank in range(4)]
            if any(row["status"] != "diagnostic_completed" for row in ranks):
                raise ValueError("Four ranks must finish all three numerical diagnostics")
            summary = dict(
                candidate=args.candidate, performance_qualified=False,
                ranks=[row["constant_oracle" if args.constant_oracle else
                           "selection_oracle" if args.selection_oracle else
                           "projection_oracle" if args.c1_projection_oracle else "attention_oracle"] for row in ranks]
            )
            (run / "attention-oracle-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
        else:
            print(json.dumps(summarize(run, args.candidate), indent=2))
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
