# SPDX-License-Identifier: Apache-2.0
"""Real request C6 group: production preparation and joint native replay A/B.

Loads one four-layer working set once. Inputs and physical KV state come
from saved 16K requests, rather than random residuals or zero KV. This is a
component gate; the extrapolation is never an achieved full-round result.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=(
            "shared_scale", "cooperative_silu", "router_shared_bf16", "router_batched_f32",
            "recipe_constants",
            "silu_decode_affine",
                                          "split_scale_planes", "w13_k_pipeline", "w2_k_pipeline", "w2_ready_scale",
                                          "w2_channels", "w2_reduce_n256", "ordered_peer_sum", "peer_post_collapse", "mhc_gate_packet",
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
            "mhc_overlap", "mhc_producer", "tensor_ready_peer", "mhc_native_dependencies", "index_query_local",
            "native_target_input",
            "runtime_selection",
            "c1_dense_chain", "fp8_prologue",
            "norm_roundtrip", "recipe_constants", "silu_decode", "silu_decode_affine",
                                          "split_scale_planes", "w13_k_pipeline", "w2_k_pipeline", "w2_ready_scale",
                                          "w2_channels", "w2_reduce_n256", "ordered_peer_sum", "peer_post_collapse", "mhc_gate_packet",
                                          "native_page_coalesce", "expert_consumer_stitch",
                                          "qk_flat", "qk_flat_direct", "stream_exp",
                                          "explicit_steps", "scaled_w13",
                                          "control_fp8",
                                          "control_fp8_pair",
                                          "moe_peer_post",
                                          "mhc_deferred", "mhc_high_plane", "input_fp8", "output_fp8", "output_layout",
            "input_norm", "n512_decode", "dense_kn", "dense_bits12", "scheduled_peer", "rope_coherent",
            "router_shared", "router_shared_fused", "router_ready_fp8", "hw_dense", "hw_dense_fused",
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
            "mhc_weight_reuse", "mhc_control_tiles", "startup_frontend",
        ),
        required=True,
    )
    parser.add_argument(
        "--dense-sidecar", type=Path, default=Path("/opt/optane/dsv41-builds/pr52-dense-fp8/input-shared-sidecar")
    )
    parser.add_argument("--group", type=int, default=5, choices=range(0, 9))
    parser.add_argument("--real-layer-count", type=int, choices=(4, 16), default=4)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--frontend-cache-dir", type=Path)
    parser.add_argument("--frontend-restore-only", action="store_true")
    parser.add_argument("--frontend-identity", help="Validated common runtime/model identity for fresh-process reuse")
    parser.add_argument("--frontend-rows", type=int, choices=(1, 6), default=6)
    parser.add_argument("--frontend-first", action="store_true",
                        help="Exercise cache capture before ordinary compilation, matching service startup")
    parser.add_argument(
        "--attention-oracle",
        action="store_true",
        help="Capture first-layer actual operands and audit FP64 attention; no performance score",
    )
    parser.add_argument("--peer-input-oracle", action="store_true")
    parser.add_argument("--constant-oracle", action="store_true")
    parser.add_argument("--c1-projection-oracle", action="store_true")
    parser.add_argument("--inline-control-oracle", action="store_true",
                        help="Qualify tiled control against C1 before native A/B in this process")
    parser.add_argument("--inline-input-norm-oracle", action="store_true",
                        help="Qualify Attention input norm and actual QKV consumers against C1 before native A/B")
    parser.add_argument("--selection-oracle", action="store_true")
    parser.add_argument("--fixed-prefix-acceptance", action="store_true")
    parser.add_argument("--quality-only", action="store_true",
                        help="Export the missing full Target p/q gate without repeating qualified native timing")
    parser.add_argument("--stack-candidates", nargs="*", default=[],
                        choices=("norm_roundtrip", "w13_k_pipeline", "mhc_gate_packet",
                                 "w2_reduce_n256", "index_query_local"))
    parser.add_argument("--precision-proof", type=Path)
    parser.add_argument("--independent-prologue", action="store_true")
    args = parser.parse_args()
    if args.candidate == "startup_frontend":
        if args.frontend_cache_dir is None or not args.frontend_identity or len(args.frontend_identity) != 64:
            parser.error("Startup frontend qualification requires a cache directory and complete runtime identity")
        os.environ["DSV41_SERVING_RUNTIME"] = args.frontend_identity
        os.environ["DSV41_SERVING_COMPILE_IDENTITY"] = hashlib.sha256(
            (args.prepared / "manifest.json").read_bytes() + args.frontend_identity.encode()).hexdigest()
    elif args.frontend_cache_dir or args.frontend_restore_only or args.frontend_rows != 6:
        parser.error("Frontend controls belong only to the startup_frontend candidate")
    if args.independent_prologue and args.candidate != "fp8_prologue":
        parser.error("Independent Q/KV readiness belongs to the FP8 prologue capability")
    if args.quality_only and not args.fixed_prefix_acceptance:
        parser.error("Quality-only mode requires the full fixed-prefix acceptance gate")
    if args.stack_candidates and (not args.fixed_prefix_acceptance or args.candidate in args.stack_candidates
                                  or len(set(args.stack_candidates)) != len(args.stack_candidates)):
        parser.error("Stack distinct qualified candidates only in the combined full40 teacher gate")
    if args.inline_control_oracle and args.candidate != "mhc_control_tiles":
        parser.error("Inline control oracle requires the tiled C2-C6 control candidate")
    if args.inline_input_norm_oracle and args.candidate != "input_norm":
        parser.error("Inline input norm oracle belongs to input_norm")
    if args.selection_oracle and args.candidate != "threshold_selection":
        parser.error("Selection precision audit belongs to threshold_selection")
    if args.c1_projection_oracle and args.candidate not in (
        "c1_dense_chain", "fp8_prologue", "dense_restore", "shared_gate_up", "shared_fp8", "shared_prequant",
        "qkv_publish", "q_prologue",
        "router_bf16", "mhc_mme_epilogue", "mhc_statistics", "mhc_weight_reuse", "mhc_control_tiles",
        "compressor_sequence"
    ):
        parser.error("C1 projection audit requires a supported projection candidate")
    rank, tp = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
        "{rank}", str(rank)
    )
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment

    prepare_environment(args.prepared, tensor_parallel_size=tp, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_replay import StageVariant, stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.tp2_prepared_plan import (
        _modules,
        _native_entries,
        collect_prepared_group_replays,
        shutdown_prepared_group_plans,
    )

    def weight_views(module):
        # Preparation may replace a weight's layout. Give each arm its own
        # module metadata while sharing the original immutable allocations.
        result = copy.copy(module)
        result._buffers = module._buffers.copy()
        result._parameters = module._parameters.copy()
        result._modules = {
            name: weight_views(child) if child is not None else None for name, child in module._modules.items()
        }
        return result

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(
        rank=rank,
        status="loading",
        candidate=args.candidate,
        fixtures=[],
        checks=[],
        rounds=[],
        qualified_gain_ms=0,
        formal_gain_ms=0,
        real_layers=4,
        group=args.group,
    )

    def save():
        (root / f"request-c6-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.candidate == "fp8_prologue":
        report["prologue_readiness"] = "independent" if args.independent_prologue else "joint"
    save()
    flag = (
        "VLLM_HPU_DSV41_DSPARK_FP8_QKV_PROLOGUE_SPLIT"
        if args.independent_prologue
        else
        "VLLM_HPU_DSV41_DSPARK_FP8_QKV_PROLOGUE"
        if args.candidate == "fp8_prologue"
        else "VLLM_HPU_DSV41_SHARED_GATE_UP"
        if args.candidate == "shared_gate_up"
        else "VLLM_HPU_DSV41_DSPARK_ROUTER_BF16"
        if args.candidate == "router_bf16"
        else "VLLM_HPU_DSV41_ATTN_DENSE_FP8"
        if args.candidate == "dense_restore"
        else "VLLM_HPU_DSV41_DSPARK_NATIVE_TARGET_INPUT"
        if args.candidate == "native_target_input"
        else "VLLM_HPU_DSV41_DSPARK_MHC_MME_EPILOGUE" if args.candidate == "mhc_statistics"
        else "VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT" if args.candidate == "hw_dense_fused"
        else "VLLM_HPU_DSV41_DSPARK_MHC_OVERLAP" if args.candidate == "mhc_native_dependencies"
        else "VLLM_HPU_DSV41_DSPARK_" + args.candidate.upper()
    )
    if args.precision_proof:
        proof = json.loads(args.precision_proof.read_text())
        reused_selection = (args.candidate == "runtime_selection" and proof.get("kind") == "c1_selection_reuse")
        if (args.candidate in ("router_shared", "router_ready_fp8")
                and proof.get("kind") == "conditional_router_acceptance"):
            from deepseek_v41_fixed_prefix_target_gate import reuse_router_acceptance
            report["precision_oracle"] = reuse_router_acceptance(
                proof, Path(__file__).resolve().parents[1], os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
        elif reused_selection:
            from reuse_deepseek_v41_c1_selection_proof import validate_current

            validate_current(proof, Path(os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"]))
            if args.group not in proof["eligible_groups"]:
                raise ValueError("C1 selection reuse does not cover this group")
            report["precision_oracle"] = dict(path=str(args.precision_proof), qualified=True,
                                              scope=proof["scope"], archived_reference=proof["reference"],
                                              current_case_checks_required=True)
        else:
            projection_proof = (args.candidate in (
                "c1_dense_chain", "fp8_prologue", "dense_restore", "shared_gate_up", "shared_fp8", "shared_prequant", "qkv_publish",
                "q_prologue", "router_bf16", "mhc_mme_epilogue", "mhc_statistics", "mhc_weight_reuse",
                "mhc_control_tiles", "compressor_sequence"
            ) and proof.get("kind") in ("c1_projection", "mhc_epilogue"))
            selection_proof = args.candidate == "threshold_selection" and proof.get("kind") == "c1_selection"
            peer_post_proof = (args.candidate == "peer_post_collapse"
                               and proof.get("kind") == "c1_peer_post_collapse")
            if not (projection_proof or selection_proof or peer_post_proof) and (
                args.candidate not in ("shared_kv_mme", "compact_kv_mme", "main_adjacent_pv", "pair_pv", "stacked_pv")
                or proof["candidate"] != args.candidate or proof["eligible_ratio"] != 1
            ):
                raise ValueError("Precision proof does not qualify this dispatch")
            if proof["candidate"] != args.candidate:
                raise ValueError("Precision proof belongs to another candidate")

            native = Path(os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
            for name, expected in proof["native_binaries"].items():
                if hashlib.sha256((native / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Precision proof belongs to another native binary")
            if peer_post_proof:
                checks = proof["oracle"]["checks"]
                if len(checks) != 3 or not all(all(e["exact"] for e in c["errors"]) for c in checks):
                    raise ValueError("Peer/post fusion changed the common C1 arithmetic boundary")
                if proof["oracle"].get("performance_measured"):
                    raise ValueError("Isolated C1 arithmetic proof must not claim communication performance")
            elif selection_proof:
                if args.group != proof.get("eligible_group", 5):
                    raise ValueError("Selection proof belongs to a different index source geometry")
                cases = proof["oracle"][rank]
                if len(cases) != 3 or not all(
                    v["passed"] and v["scores_have_bf16_boundary"] and v["selected_ids_exact"]
                    and v["candidate_blocks_exact"] for v in cases
                ):
                    raise ValueError("C6 selection disagrees with accepted C1 ordered selection")
            elif projection_proof:
                if len(proof["oracle"][rank]) != 3 or not all(
                    case["passed"] and all(row["passed"] and row["relative_l2"] <= 0.002
                                           for row in case["projections"].values())
                    for case in proof["oracle"][rank]
                ):
                    raise ValueError("C6 projection disagrees with the accepted C1 precision contract")
                if args.candidate == "qkv_publish" and not all(
                    case.get("canonical_swa_exact", False) for case in proof["oracle"][rank]
                ):
                    raise ValueError("QKV precision proof must preserve canonical SWA writes exactly")
            elif args.candidate in ("main_adjacent_pv", "pair_pv", "stacked_pv"):
                if proof.get("kind") != "attention_pair_pv" or len(proof["oracle"][rank]) != 3:
                    raise ValueError("Pair PV requires three actual-input official-tolerance cases")
                for case in proof["oracle"][rank]:
                    if not (all(case["inputs_exact"].values()) and case["candidate"]["finite"]
                            and case["candidate"]["official_tolerance_close"]
                            and case["candidate"]["relative_l2"] <= 0.002):
                        raise ValueError("Pair PV failed the official numerical tolerance")
            else:
                for case in proof["oracle"][rank]:
                    if not (
                        all(case["inputs_exact"].values())
                        and case["candidate"]["finite"]
                        and case["candidate"]["max_abs"] <= case["parent"]["max_abs"]
                        and case["candidate"]["rms"] <= case["parent"]["rms"] * (1 + 2**-23)
                    ):
                        raise ValueError("The actual-input FP64 precision proof failed")
            report["precision_oracle"] = dict(
                path=str(args.precision_proof),
                sha256=hashlib.sha256(args.precision_proof.read_bytes()).hexdigest(),
                qualified=True,
                scope=("Three real peer packets/gates vs C1 fused post/collapse; whole-model alpha pending"
                       if peer_post_proof else
                       "Actual C6 score/causal operands vs accepted per-row C1 ordered selector; full quality pending"
                       if selection_proof else
                       "Three actual-input C6 projections vs accepted C1 methods; full-service quality pending"
                       if projection_proof else "Three actual ratio-one attention cases; full-service quality pending"),
            )

    previous = os.environ.get(flag)
    native_dependency_flag = "VLLM_HPU_DSV41_TP_MHC_OVERLAP"
    previous_native_dependency = os.environ.get(native_dependency_flag)
    try:
        with (
            set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))),
            torch.inference_mode(),
        ):
            bind_worker_cpu(rank)
            torch.hpu.set_device(rank)
            # Register the candidate through the serving worker loader, then
            # keep dispatch disabled until the candidate arm is captured.
            os.environ[flag] = "1"
            load_native_operators()
            os.environ[flag] = "0"
            init_distributed_environment(
                world_size=tp, rank=rank, local_rank=rank, distributed_init_method="env://", backend="hccl"
            )
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            files = sorted((args.fixtures / f"rank{rank}").glob("c6-*.pt"))
            if not 3 <= len(files) <= 5 or args.samples < 3:
                raise ValueError("Need 3-5 actual request fixtures and at least three device samples")
            raw = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
            if args.candidate == "startup_frontend" and args.frontend_rows == 1:
                for data in raw:
                    data["positions"] = data["positions"][:1]
                    data["ids"] = data["ids"][:1]
                    for group in data["groups"].values():
                        for name in ("residual", "pre"):
                            if name in group:
                                group[name] = group[name][:1]
            for path, data in zip(files, raw, strict=True):
                if not data.get("request_context_qualified") or data["rank"] != rank:
                    raise ValueError("Only rank-matched actual request state is accepted")
                report["fixtures"].append(dict(name=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if (
                args.precision_proof
                and [row["sha256"] for row in report["fixtures"]] != proof["fixtures_by_rank"][rank]
            ):
                raise ValueError("Precision proof belongs to other actual inputs")
            start, stop = args.group * 4, args.group * 4 + args.real_layer_count
            if args.real_layer_count != 4:
                if (args.candidate not in ("startup_frontend", "mhc_post_stats", "main_single_bank", "coherent_swa", "c1_dense_chain", "fp8_prologue",
                                          "runtime_selection", "mhc_producer", "tensor_ready_peer",
                                          "mhc_native_dependencies", "index_query_local", "norm_roundtrip",
                                          "recipe_constants", "silu_decode", "silu_decode_affine",
                                          "split_scale_planes", "w13_k_pipeline", "w2_k_pipeline", "w2_ready_scale",
                                          "w2_channels", "w2_reduce_n256", "ordered_peer_sum", "peer_post_collapse", "mhc_gate_packet",
                                          "native_page_coalesce", "expert_consumer_stitch",
                                          "qk_flat", "qk_flat_direct", "stream_exp",
                                          "explicit_steps", "scaled_w13",
                                          "control_fp8",
                                          "control_fp8_pair",
                                          "moe_peer_post",
                                          "mhc_deferred", "mhc_high_plane", "input_fp8", "output_fp8", "output_layout",
                                          "input_norm", "mhc_statistics",
                                          "n512_decode", "dense_kn", "dense_bits12", "scheduled_peer", "rope_coherent",
                                          "router_shared", "router_shared_fused", "router_ready_fp8",
                                          "hw_dense", "hw_dense_fused", "shared_scale", "cooperative_silu",
                                          "router_shared_bf16", "router_batched_f32")
                        or args.group != 5):
                    raise ValueError("This real16 producer/consumer gate starts at source20 and owns16layers")
                report["real_layers"] = args.real_layer_count
            if args.candidate in ("joined_sampled_tail", "deep_queue", "round_input_publication"):
                if args.group != 6:
                    raise ValueError("Actual joined real16 tail begins at layer24")
                stop = 40
                report["real_layers"] = 16
            if args.fixed_prefix_acceptance:
                if (args.candidate not in ("mhc_post_stats", "main_single_bank", "coherent_swa", "c1_dense_chain", "fp8_prologue",
                                          "runtime_selection", "mhc_producer", "tensor_ready_peer",
                                          "mhc_native_dependencies", "index_query_local", "norm_roundtrip",
                                          "recipe_constants", "silu_decode", "silu_decode_affine",
                                          "split_scale_planes", "w13_k_pipeline", "w2_k_pipeline", "w2_ready_scale",
                                          "w2_channels", "w2_reduce_n256", "ordered_peer_sum", "peer_post_collapse", "mhc_gate_packet",
                                          "native_page_coalesce", "expert_consumer_stitch",
                                          "qk_flat", "qk_flat_direct", "stream_exp",
                                          "explicit_steps", "scaled_w13",
                                          "control_fp8",
                                          "control_fp8_pair",
                                          "moe_peer_post",
                                          "mhc_deferred", "mhc_high_plane", "input_fp8", "output_fp8", "output_layout",
                                          "input_norm", "mhc_statistics",
                                          "n512_decode", "dense_kn", "dense_bits12", "scheduled_peer", "rope_coherent",
                                          "router_shared", "router_shared_fused", "router_ready_fp8",
                                          "hw_dense", "hw_dense_fused", "shared_scale", "cooperative_silu",
                                          "router_shared_bf16", "router_batched_f32")
                        or args.group != 0):
                    raise ValueError("Fixed-prefix Target acceptance requires a full Target candidate from layer0")
                start, stop = 0, 40
                report["real_layers"] = 40
            if args.candidate == "native_target_input" and args.group != 0:
                raise ValueError("Native Target input must include the first four decoder layers")
            if (not args.fixed_prefix_acceptance and args.candidate != "native_target_input"
                    and any(i in (1, 14) for i in range(start, stop))):
                raise ValueError("This fixture does not retain Engram inputs for that group")
            shard = PreparedV41Shard(args.prepared, 0, rank)
            config = json.loads((args.prepared / "config.json").read_text())
            text = config["text_config"]
            first_indexed = next((i for i in range(start, stop) if text["compress_ratios"][i]), None)
            if first_indexed is not None and first_indexed not in text["index_source_layer_ids"]:
                raise ValueError("The fixture keeps final pooled indices; start at an index owner to regenerate them")
            if args.candidate in ("cohesive_mla", "merged_mla") and any(
                text["compress_ratios"][i] != 1 for i in range(start, stop)
            ):
                raise ValueError("The initial cohesive MLA candidate is confined to ratio-one cache layers")
            eligible_layers = (
                sum(ratio == 1 for ratio in text["compress_ratios"][:40])
                if args.candidate in ("cohesive_mla", "merged_mla")
                else text["num_hidden_layers"]
            )
            if args.candidate in ("compact_stream_mme", "full_row", "fp4_table", "row_cache",
                                  "pv_rope", "hash_mla", "mla_coord_cache",
                                  "shared_kv_mme",
                                  "scale_cache", "pair_pv", "stacked_pv", "batch6_kv", "merge_cache", "exp_pv"):
                eligible_layers = sum(r in (1, 2) for r in text["compress_ratios"][:40])
            if args.candidate in ("shared_kv_mme", "compact_kv_mme"):
                eligible_layers = sum(r == 1 for r in text["compress_ratios"][:40])
            if args.candidate in ("main_mirror", "decoded_publish", "swa_cache"):
                eligible_layers = sum(r == text["compress_ratios"][start]
                                      for r in text["compress_ratios"][:40])
            multiplier = eligible_layers / (stop - start)
            if args.candidate in ("main_single_bank", "coherent_swa", "main_adjacent_pv", "swa_source_reuse",
                                  "layer_main_reuse", "layer_main_split"):
                def reuse_count(first, last):
                    seen, publish, reused = set(), 0, 0
                    for index in range(first, last):
                        ratio = text["compress_ratios"][index]
                        if ratio not in (1, 2):
                            continue
                        kv_owner = max(source for source in text["kv_source_layer_ids"] if source <= index)
                        index_owner = max(source for source in text["index_source_layer_ids"] if source <= index)
                        key = kv_owner, index_owner, ratio
                        if key in seen:
                            reused += 1
                        else:
                            publish += 1
                            seen.add(key)
                    return publish, reused

                groups = [reuse_count(first, min(first + 4, 40)) for first in range(0, 40, 4)]
                local_publish, local_reuse = reuse_count(start, stop)
                eligible_layers = sum(reused for _, reused in groups)
                if local_reuse == 0:
                    raise ValueError("This group has no same-owner main consumer")
                multiplier = eligible_layers / local_reuse
                if args.candidate == "main_single_bank":
                    # Both publishers and reusers change. Use the smaller ownership
                    # multiplier; full40 confirmation is needed after this gate.
                    multiplier = min(sum(pub for pub, _ in groups) / local_publish,
                                     eligible_layers / local_reuse)
                    eligible_layers = (stop - start) * multiplier
                report["main_reuse_scope"] = dict(global_publish=sum(pub for pub, _ in groups),
                                                  global_reuse=sum(reused for _, reused in groups),
                                                  local_publish=local_publish, local_reuse=local_reuse,
                                                  layout="ephemeral BF16 operands within one native layer group")
            if args.candidate in ("threshold_selection", "candidate_keys"):
                full = text["candidate_source_layer_id"]
                owners = text["index_source_layer_ids"]
                if args.candidate == "candidate_keys" or start > full:
                    eligible_layers = sum(i > full for i in owners)
                    geometry = "Reindex per-query candidate rows"
                elif start == full:
                    eligible_layers = 1
                    geometry = "Full ratio-one source and candidate publication"
                else:
                    eligible_layers = sum(i < full for i in owners)
                    geometry = "Full ratio-two source without candidate publication"
                local_owners = sum(start <= i < stop for i in owners)
                if local_owners != 1:
                    raise ValueError("Index-only component must contain one production source owner")
                multiplier = eligible_layers / local_owners
                report["projection_geometry"] = geometry
            if args.candidate == "compressor_sequence":
                owners=text["kv_source_layer_ids"]
                eligible_layers=sum(text["compress_ratios"][i]==2 for i in owners)
                local=sum(start<=i<stop and text["compress_ratios"][i]==2 for i in owners)
                if not local:
                    raise ValueError("Compressor batch needs a ratio-two KV owner")
                multiplier=eligible_layers/local
                report["compressor_scope"]={"global_owners":eligible_layers,"local_owners":local}
            report.update(projection_multiplier=multiplier, projected_layer_count=eligible_layers)
            prefixes = tuple(f"layers.{i}." for i in range(start, stop))
            specs = {name: spec for name, spec in shard.specs.items() if name.startswith(prefixes)
                     or (args.candidate == "native_target_input" or args.fixed_prefix_acceptance)
                     and name == "embed.weight"
                     or args.fixed_prefix_acceptance and name in ("head.weight", "norm.weight")
                     or args.candidate in ("joined_sampled_tail", "deep_queue", "round_input_publication") and
                     (name.startswith("mtp.") or name in ("head.weight", "embed.weight", "norm.weight"))}
            weights = _weight_tree(specs)
            load_weight_tree(shard, weights, "hpu", specs, expert_n256_layers=list(range(start, stop)))
            if args.candidate in ("main_mirror", "decoded_publish"):
                os.environ["VLLM_HPU_DSV41_DSPARK_MAIN_MIRROR"] = "1"
                os.environ[flag] = "1"
            shared = PagedCSA2SharedState(text, 0, 40, "hpu", 1048576, tensor_parallel_size=tp)
            os.environ[flag] = "0"
            if args.candidate == "native_target_input" or args.fixed_prefix_acceptance:
                from deepseek_v41_native_input_gate import real_engram_rows
                engram_cases = []
                for case in raw:
                    rows = real_engram_rows(args.prepared, case, rank, tp)
                    engram_cases.append(tuple(value.to("hpu") for value in rows))
                if not args.fixed_prefix_acceptance:
                    eligible_layers, multiplier = 1, 1.
                    report.update(projection_multiplier=1., projected_layer_count=1,
                                  scope="Embedding/TP reduction -> first four layers -> FFN consumer")
            if args.candidate == "decoded_publish":
                os.environ["VLLM_HPU_DSV41_DSPARK_MAIN_MIRROR"] = "0"
            shared.index_mirror_tokens, shared.index_mirror_valid = 1048576, True
            selection_aliases = {}
            for selection in shared.topk.values():
                selection_aliases.setdefault(id(selection.indices), []).append(selection)

            def source_state(data, name):
                return torch.load(
                    Path(data["decoder_state_root"]) / data["decoder_state_files"][name], weights_only=True
                )

            # Bind the full physical allocation, not a compact view that
            # changes the compiler's access/placement contract.
            for name in raw[0]["decoder_state_files"]:
                if not name.startswith("shared."):
                    continue
                owner_name, _, leaf = name.removeprefix("shared.").rpartition(".")
                owner = shared.get_submodule(owner_name)
                owner.register_buffer(leaf, source_state(raw[0], name).to("hpu"), False)
            # named_buffers exports each shared allocation once. Preserve the
            # allocation ownership chosen by the production constructor.
            for selections in selection_aliases.values():
                for selection in selections[1:]:
                    selection.indices = selections[0].indices
            # A partial production group must bind only allocations present
            # in that group's captured program. Keep full physical caches;
            # remove metadata owners for unrelated layers, not tensor rows.
            sources = {
                str(max(source for source in text["kv_source_layer_ids"] if source <= i))
                for i in range(start, stop) if text["compress_ratios"][i]
            }
            selectors = {
                str(max(source for source in text["index_source_layer_ids"] if source <= i))
                for i in range(start, stop) if text["compress_ratios"][i]
            }
            shared.sources = torch.nn.ModuleDict(
                {key: value for key, value in shared.sources.items() if key in sources}
            )
            shared.topk = torch.nn.ModuleDict({key: value for key, value in shared.topk.items() if key in selectors})
            if text["candidate_source_layer_id"] not in range(start, stop) and all(
                text["compress_ratios"][i] != 1 for i in range(start, stop)
            ):
                # This early ratio-two group neither produces nor consumes
                # layer20's candidate pool. The native tensor program must
                # not bind that unrelated mutable allocation.
                shared.candidate_pool = None
            # TP4 exports its pooled selection under the first registered
            # owner. The retained owner still points to that same allocation.
            selection_name = "shared.topk." + next(iter(shared.topk)) + ".indices"
            reduce, gather = stage_collectives(rank, True, tp)
            lookup = mxfp4_bf16_lut(torch.device("hpu"))
            programs, plans = [], []
            arm_order = (1, 0) if args.candidate == "startup_frontend" and args.frontend_first else (0, 1)
            for arm in arm_order:
                if args.candidate in ("joined_sampled_tail", "deep_queue", "round_input_publication") and arm:
                    break
                os.environ[flag] = str(arm)
                if args.candidate == "mhc_native_dependencies":
                    # Minimal Target-only capability probe: both native graphs
                    # instantiate before timings. No draft graph is captured.
                    os.environ[native_dependency_flag] = str(arm)
                for stacked in args.stack_candidates:
                    os.environ["VLLM_HPU_DSV41_DSPARK_" + stacked.upper()] = str(arm)
                if args.candidate in ("shared_fp8", "shared_prequant"):
                    os.environ["VLLM_HPU_DSV41_SHARED_GATE_UP"] = str(arm if args.candidate == "shared_fp8" else 1)
                if args.candidate == "shared_prequant":
                    os.environ["VLLM_HPU_DSV41_DSPARK_SHARED_FP8"] = "1"
                program = torch.nn.Module()
                program.weights, program.shared = weight_views(weights), shared
                # Baseline precision belongs to the serving profile, regardless
                # of which independent operator this component measures.
                baseline_shared = (args.candidate not in ("dense_restore", "shared_fp8", "shared_prequant")
                                   and os.getenv("VLLM_HPU_DSV41_DSPARK_SHARED_FP8") == "1")
                if (baseline_shared or args.candidate in ("qkv_publish", "shared_prequant")
                        or (args.candidate in ("c1_dense_chain", "dense_restore", "shared_fp8") and arm)):
                    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar

                    sidecar = DenseFP8Sidecar(args.dense_sidecar, shard)
                    names = (("shared_w1", "shared_w3", "shared_w2")
                             if baseline_shared or args.candidate in ("shared_fp8", "shared_prequant")
                             else ("wq_a", "wkv", "wq_b", "wo_b"))
                    if args.candidate in ("wo_handoff", "output_fp8") and arm:
                        names = (*names, "wo_b")
                    dense_config = {name: list(range(start, stop)) for name in names}
                    selected = {
                        (f"layers.{index}.ffn.shared_experts.{name.removeprefix('shared_')}.weight"
                         if name.startswith("shared_") else f"layers.{index}.attn.{name}.weight")
                        for index in range(start, stop) for name in dense_config}
                    projection_specs = {name: spec for name, spec in specs.items() if name in selected}
                    # The real expert/router/wo_a allocations are already
                    # resident. Do not reload or duplicate their immutable
                    # weight working set when switching projection precision.
                    load_weight_tree(
                        shard, program.weights, "hpu", projection_specs,
                        dense_sidecar=sidecar, dense_config=dense_config
                    )
                if args.candidate == "wo_handoff" and arm:
                    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar

                    wo_sidecar = WoaFP8Sidecar(args.prepared / "sidecars/wo_a_fp8", shard)
                    wo_names = {f"layers.{index}.attn.wo_a.weight" for index in range(start, stop)}
                    load_weight_tree(shard, program.weights, "hpu",
                                     {name: spec for name, spec in specs.items() if name in wo_names},
                                     woa_sidecar=wo_sidecar, woa_layers=list(range(start, stop)))
                if args.candidate in ("q_prologue", "c1_dense_chain", "fp8_prologue", "input_fp8") and arm:
                    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
                    q_sidecar = DenseFP8Sidecar(args.dense_sidecar, shard)
                    q_projections = (("wq_a", "wkv", "wq_b", "wo_b")
                                     if args.candidate == "c1_dense_chain" else
                                     ("wq_a", "wkv", "wq_b") if args.candidate == "fp8_prologue" else
                                     ("wq_a", "wkv") if args.candidate == "input_fp8" else ("wq_b",))
                    q_names = {f"layers.{index}.attn.{name}.weight"
                               for index in range(start, stop) for name in q_projections}
                    load_weight_tree(shard, program.weights, "hpu",
                                     {name: spec for name, spec in specs.items() if name in q_names},
                                     dense_sidecar=q_sidecar,
                                     dense_config={name: list(range(start, stop)) for name in q_projections})
                program.config, program.shard = config, shard
                program.pp_rank, program.tp_rank, program.tensor_parallel_size = 0, rank, tp
                program.length, program.search_length = 1048576, raw[0]["search_length"]
                program.start, program.stop, program.dspark = start, stop, True
                program.is_last_stage = (args.candidate in
                                         ("joined_sampled_tail", "deep_queue", "round_input_publication")
                                         or args.fixed_prefix_acceptance)
                if (args.candidate in ("joined_sampled_tail", "deep_queue", "round_input_publication")
                        or args.fixed_prefix_acceptance):
                    program.bf16_head = True
                    program.weights.head.weight = program.weights.head.weight.bfloat16()
                program.fp8_decode, program.expert_n256 = True, True
                program.reduce, program.all_gather = reduce, gather
                program.generation, program.precision_fingerprint = 1, (
                    "request_c6", arm, args.candidate, tuple(args.stack_candidates))
                program.layers = torch.nn.ModuleList()
                if args.candidate == "router_bf16" and arm:
                    for index in range(start, stop):
                        gate = program.weights.layers.get_submodule(str(index)).ffn.gate
                        gate.weight = gate.weight.bfloat16()
                if args.candidate == "qkv_publish":
                    os.environ["VLLM_HPU_DSV41_ATTN_DENSE_FP8"] = "1"
                    os.environ["VLLM_HPU_DSV41_Q_SCALE_ROPE"] = "1"
                    os.environ["VLLM_HPU_DSV41_ATTN_FUSED_NORM"] = "1"
                for index in range(start, stop):
                    layer = PreparedDecoderLayer(
                        program.weights.layers.get_submodule(str(index)),
                        text,
                        index,
                        shared,
                        shard.manifest["normal_scales"][f"layers.{index}.ffn.experts"][rank],
                        lookup,
                        reduce,
                        gather,
                        "hpu",
                        tp,
                        collect_target_state=True,
                    )
                    for name in raw[0]["decoder_state_files"]:
                        if name.startswith(f"layers.{index}.attention."):
                            layer.attention.register_buffer(
                                name.rsplit(".", 1)[-1], source_state(raw[0], name).to("hpu"), False
                            )
                    if args.candidate in ("main_mirror", "decoded_publish"):
                        layer.attention.dspark_main_mirror = bool(arm)
                    layer.attention.set_search_length(program.search_length)
                    layer.prepare_mhc_control_weights()
                    layer.moe.prepare_shared_gate_up_weight()
                    layer.moe.prepare_split_scale_planes()
                    layer.moe.prepare_router_shared_weight()
                    layer.moe.prepare_router_shared_bf16_weight(shard)
                    layer.moe.prepare_router_batched_weight()
                    layer.moe.prepare_router_ready_weight()
                    if baseline_shared:
                        shared_weights = layer.moe.weights.shared_experts
                        shared_fp8 = all(getattr(getattr(shared_weights, name), "dense_fp8", False)
                                         for name in ("w1", "w3", "w2"))
                        report.setdefault("shared_weight_contract", []).append(
                            dict(arm=arm, layer=index, checkpoint_sidecar_fp8=shared_fp8)
                        )
                        if not shared_fp8:
                            raise RuntimeError("Native component omitted the serving shared-expert FP8 weights")
                    layer.attention.prepare_qkv_input_weight()
                    layer.attention.prepare_compressor_input_weight()
                    if args.candidate == "wo_handoff" and arm:
                        layer.attention.woa_fp8 = True
                        layer.attention.woa_output_roundtrip = True
                    layer.attention.prepare_output_weight()
                    layer.attention.prepare_dense_kn_weights()
                    layer.attention.prepare_dense_bits12_weights()
                    layer.attention.prepare_local_index_query_weights(release_local=False)
                    if args.candidate in ("hw_dense", "hw_dense_fused") and arm:
                        from vllm_gaudi.ops.deepseek_v41_hw_dense import prepare
                        prepare(layer.attention, layer.weights.attn_norm.weight)
                    if (
                        args.candidate in ("feature_silu", "w2_three_routes", "silu_unroll",
                                           "physical_silu", "physical_role_silu")
                        and not layer.moe.c6_token_wide_sat
                    ):
                        raise RuntimeError("The production weights did not enable the C6 token-wide SAT consumer")
                    program.layers.append(layer)
                if args.candidate == "recipe_constants" and arm:
                    from vllm_gaudi.ops.deepseek_v41_recipe_constants import prepare_layer_constants

                    report["recipe_constant_contracts"] = [prepare_layer_constants(layer) for layer in program.layers]
                programs.append(program)
                if args.peer_input_oracle:
                    from deepseek_v41_peer_input_oracle import install, install_reduction

                    for layer in program.layers:
                        install(layer.attention, raw[0]["positions"].numel())
                        install_reduction(layer.moe, raw[0]["positions"].numel())
                if args.attention_oracle:
                    from deepseek_v41_attention_oracle import install_probe

                    owner = program.layers[1 if args.candidate == "main_adjacent_pv" else 0].attention
                    install_probe(owner, raw[0]["positions"].numel())
                if args.selection_oracle and arm:
                    from deepseek_v41_selection_oracle import install

                    owner = program.layers[0].attention
                    columns = (shared.candidate_pool.shape[-1] * 8
                               if owner.layer > owner.candidate_source
                               else raw[0]["search_length"] // owner.ratio)
                    install(owner, 6, columns)
                if args.c1_projection_oracle and (arm or args.candidate == "compressor_sequence"):
                    from deepseek_v41_projection_oracle import (
                        install_probe, install_shared_probe, install_publish_probe)

                    if args.candidate == "compressor_sequence":
                        from deepseek_v41_compressor_oracle import install

                        install(program.layers[0].attention, raw[0]["positions"].numel())
                    elif args.candidate == "qkv_publish":
                        install_publish_probe(program.layers[0].attention, raw[0]["positions"].numel())
                    elif args.candidate == "q_prologue":
                        from deepseek_v41_query_prologue_oracle import install

                        install(program.layers[0].attention, raw[0]["positions"].numel())
                    elif args.candidate == "router_bf16":
                        from deepseek_v41_router_oracle import install

                        install(program.layers[0].moe, raw[0]["positions"].numel())
                    elif args.candidate in ("shared_gate_up", "shared_fp8", "shared_prequant"):
                        install_shared_probe(program.layers[0].moe, raw[0]["positions"].numel())
                    elif args.candidate in ("mhc_weight_reuse", "mhc_control_tiles"):
                        pass  # Isolated control/RRMS oracle reads the actual group-entry residual.
                    else:
                        install_probe(program.layers[0].attention, raw[0]["positions"].numel())
                if args.candidate == "qkv_publish":
                    flags = [dict(layer=layer.layer, publish=layer.attention.dspark_qkv_publish,
                                  fused_norm=layer.attention.fused_norm, q_scale=layer.attention.q_scale_rope,
                                  native_rope=layer.attention.native_rope,
                                  fp8=getattr(layer.attention.weights.wq_b, "dense_fp8", False))
                             for layer in program.layers]
                    report.setdefault("prepared_dispatch", []).append(dict(arm=arm, layers=flags))
                    if not all(all(value for name, value in row.items() if name not in ("layer", "publish"))
                               and bool(row["publish"]) == bool(arm) for row in flags):
                        raise RuntimeError("Q/KV candidate prerequisites not selected before native capture")
                source = raw[0]["groups"][max(0, args.group - 1)]
                values = tuple(
                    v.to("hpu") for v in (source["residual"], source["pre"], raw[0]["positions"], raw[0]["ids"])
                )
                plan_engram = (engram_cases[0]
                               if args.candidate == "native_target_input" or args.fixed_prefix_acceptance else ())
                if args.candidate == "startup_frontend":
                    os.environ["VLLM_HPU_DSV41_FRONTEND_CACHE_DIR"] = str(args.frontend_cache_dir) if arm else ""
                prepare_started = time.perf_counter()
                plan = StageVariant(program, *values, plan_engram,
                                    native_input=args.candidate == "native_target_input" and bool(arm),
                                    fused_text_io=args.fixed_prefix_acceptance)
                if start > text["candidate_source_layer_id"]:
                    # This isolated Reindex group reads an earlier owner's
                    # decoded cache and mirror, but does not capture that
                    # owner's packed-cache/history writes. Keep full resident
                    # allocations; bind only this partial group's live state.
                    unused = {
                        id(value) for name, value in program.named_buffers()
                        if name.startswith("shared.sources.")
                        and int(name.split(".")[2]) not in range(start, stop)
                        and name.rsplit(".", 1)[-1] not in ("decoded_main", "index_mirror")
                    }
                    plan.states = tuple(value for value in plan.states if id(value) not in unused)
                    report["partial_group_unbound_state"] = [
                        name for name, value in program.named_buffers() if id(value) in unused
                    ]
                plans.append(plan)
                if args.candidate in ("joined_sampled_tail", "deep_queue", "round_input_publication"):
                    from deepseek_v41_joint_tail_gate import qualify

                    qualify(args, program, plan, raw, source_state, selection_name, sources, report, save)
                    return
                # Cold compilation must happen inside this owner's native
                # context: the shared compiler rejects contextless captures.
                for _ in range(2):
                    with collect_prepared_group_replays(
                        owner=plan,
                        adapter=plan.adapter,
                        snapshot=plan.snapshot,
                        diagnostic_host_replay=True,
                        hidden_states=plan.fixed[0],
                        pre_mix=plan.fixed[1],
                        positions=plan.fixed[2],
                        input_ids=plan.fixed[3],
                        attention_inputs=plan.engram,
                        pp_wire=None,
                        metadata=plan.metadata,
                        state_generation=(program.generation, program.precision_fingerprint),
                        state_tensors=plan.states,
                    ):
                        plan(*values, plan_engram, native_input=plan.native_input,
                             fused_text_io=args.fixed_prefix_acceptance)
                    torch.hpu.synchronize()
                for _ in range(2):
                    plan(*values, plan_engram, native_input=plan.native_input,
                             fused_text_io=args.fixed_prefix_acceptance)
                    torch.hpu.synchronize()
                if args.candidate == "startup_frontend":
                    entries = plan.compiled.chunks
                    cache_stats = [dict(entry.stats) for entry in entries if hasattr(entry, "stats")]
                    report.setdefault("startup_preparation", []).append(dict(
                        arm=arm, rows=args.frontend_rows,
                        wall_seconds=time.perf_counter() - prepare_started,
                        cache=cache_stats,
                        backend_seconds=sum(getattr(artifact._artifacts.compiled_fn, "backend_seconds", 0.)
                                            for entry in entries if hasattr(entry, "variants")
                                            for artifact in entry.variants)))
                    if arm and (len(cache_stats) != len(entries) or args.frontend_restore_only and
                                any(row["captures"] or not row["restores"] for row in cache_stats)):
                        raise RuntimeError("Full native group did not reuse its guarded frontend without tracing")
                    save()
                if plan not in _native_entries:
                    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
                    from vllm_gaudi import envs

                    report["capture_diagnostic"] = dict(
                        stats=prepared_group_stats(),
                        graph_replay=envs.VLLM_HPU_DSV41_GRAPH_REPLAY,
                        modules=[
                            dict(
                                owners=list(module.plan_owners),
                                prepares=module.prepares,
                                operators=[str(node.target) for node in module.original.graph.nodes],
                            )
                            for module in tuple(_modules)
                        ],
                    )
                    raise RuntimeError("C6 component did not capture production joint native replay")
                # Check the captured producer/consumer recipes, rather than
                # crediting an environment flag which may be bypassed by an
                # earlier dispatch condition. No additional profiling run.
                operators = Counter()
                recipes = []
                mhc_partitions = 0
                mhc_producers = 0
                output_bmm_signatures = []
                control_weight_shapes = []
                for module in tuple(_modules):
                    if not any(owner is not None and owner[0] == id(plan) for owner in module.plan_owners):
                        continue
                    mhc_partitions += sum(node.op == "call_module" and "_mhc_" in str(node.target)
                                          for node in module.original.graph.nodes)
                    mhc_producers += sum(node.op == "call_module" and "_mhc_producer_" in str(node.target)
                                         for node in module.original.graph.nodes)
                    for node in module.original.graph.nodes:
                        if node.op != "call_module":
                            continue
                        recipe = module.original.get_submodule(node.target)
                        graph = recipe.fx_module.graph
                        targets = [str(item.target) for item in graph.nodes if item.op == "call_function"]
                        if args.candidate == "mhc_high_plane":
                            for item in graph.nodes:
                                if (item.op == "call_function" and
                                        "custom_deepseek_v41_control_mme_f32_gaudi2" in str(item.target)):
                                    meta = getattr(item.args[1], "meta", {})
                                    tensor = meta.get("val", meta.get("example_value"))
                                    control_weight_shapes.append(None if not isinstance(tensor, torch.Tensor) else
                                                                 list(tensor.shape))
                        if args.candidate == "output_layout":
                            for item in graph.nodes:
                                if item.op == "call_function" and str(item.target) == "aten.bmm.default":
                                    operands = []
                                    for arg in item.args[:2]:
                                        meta = getattr(arg, "meta", {})
                                        tensor = meta.get("val", meta.get("example_value"))
                                        operands.append(None if not isinstance(tensor, torch.Tensor) else
                                                        dict(shape=list(tensor.shape), stride=list(tensor.stride())))
                                    output_bmm_signatures.append(dict(recipe_id=recipe._recipe_id, operands=operands))
                        operators.update(targets)
                        recipes.append(dict(recipe_id=recipe._recipe_id, operators=targets))
                expected = (
                    (
                        "custom_deepseek_v41_logical_mla_merged_gaudi2"
                        if args.candidate == "merged_mla"
                        else "custom_deepseek_v41_logical_mla_cohesive_gaudi2"
                    )
                    if args.candidate in ("cohesive_mla", "merged_mla")
                    else (
                        "custom_deepseek_v41_expert_n256_moe_token_wide_three_route_w2_fp8_gaudi2"
                        if args.candidate == "w2_three_routes"
                        else "custom_deepseek_v41_expert_n256_moe_token_wide_feature_silu_fp8_gaudi2"
                    )
                )
                if args.candidate == "mhc_post_stats":
                    expected="custom_deepseek_v41_dspark_mhc_post_norm_statistics_gaudi2"
                elif args.candidate == "main_single_bank":
                    expected="custom_deepseek_v41_main_single_bank_reuse_mla_gaudi2"
                elif args.candidate == "coherent_swa":
                    expected="custom_deepseek_v41_coherent_swa_mla_gaudi2"
                elif args.candidate == "main_adjacent_pv":
                    expected="custom_deepseek_v41_main_adjacent_reuse_mla_gaudi2"
                elif args.candidate == "swa_source_reuse":
                    expected="custom_deepseek_v41_swa_source_reuse_mla_gaudi2"
                elif args.candidate == "silu_full_rows":
                    expected="custom_deepseek_v41_expert_n256_moe_token_wide_silu_full_rows_fp8_gaudi2"
                elif args.candidate == "compressor_sequence":
                    expected="custom_deepseek_v41_compressor_sequence_ordered_bf16_gaudi2"
                elif args.candidate == "pair_silu":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_pair_silu_fp8_gaudi2"
                elif args.candidate == "swa_cache":
                    expected = "custom_deepseek_v41_swa_batch_cache_ordered_gaudi2"
                elif args.candidate == "split_feature_silu":
                    expected = "custom_deepseek_v41_expert_n256_moe_split_feature_silu_fp8_gaudi2"
                elif args.candidate == "stream_exp":
                    expected = "custom_deepseek_v41_main_stream_exp_"
                elif args.candidate == "qk_flat_direct":
                    expected = "custom_deepseek_v41_main_qk_flat_direct_"
                elif args.candidate == "qk_flat":
                    expected = "custom_deepseek_v41_main_qk_flat_"
                elif args.candidate == "layer_main_split":
                    expected = "custom_deepseek_v41_main_split_reuse_mla_gaudi2"
                elif args.candidate == "layer_main_reuse":
                    expected = "custom_deepseek_v41_main_batch_reuse_mla_gaudi2"
                elif args.candidate == "compact_stream_mme":
                    expected = "custom_deepseek_v41_logical_mla_compact_stream_mme_gaudi2"
                elif args.candidate == "unpaired_feature_silu":
                    expected = "custom_deepseek_v41_expert_n256_moe_unpaired_feature_silu_fp8_gaudi2"
                elif args.candidate == "silu_scalar_cache":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_silu_scalar_cache_fp8_gaudi2"
                elif args.candidate == "q_bf16_rope":
                    expected = "custom_deepseek_v41_q_bf16_projection_rope_gaudi2"
                elif args.candidate == "query_norm":
                    expected = "custom_deepseek_v41_attention_norm_bf16_gaudi2"
                elif args.candidate == "peer_post_collapse":
                    expected = "custom_deepseek_v41_peer_mhc_post_collapse_gaudi2"
                elif args.candidate == "ordered_peer_sum":
                    expected = "custom_deepseek_v41_ordered_peer_sum_gaudi2"
                elif args.candidate == "router_ready_fp8":
                    expected = ("custom_deepseek_v41_router_pair_fp8_gaudi2"
                                if os.getenv("VLLM_HPU_DSV41_DSPARK_ROUTER_READY_PAIR", "0") == "1"
                                else "custom_deepseek_v41_router_ready_fp8_gaudi2")
                elif args.candidate == "router_shared_fused":
                    expected = "custom_deepseek_v41_router_shared_scaled_gaudi2"
                elif args.candidate == "router_shared":
                    expected = "joint Router/shared FP8 projection"
                elif args.candidate in ("hw_dense", "hw_dense_fused"):
                    expected = ("custom_deepseek_v41_hw_dense_roundtrip_fp8_gaudi2"
                                if args.candidate == "hw_dense_fused" else
                                "custom_deepseek_v41_hw_dense_fp8_gaudi2")
                elif args.candidate == "rope_coherent":
                    expected = "coherent_bf16_gaudi2"
                elif args.candidate == "scheduled_peer":
                    expected = "custom_deepseek_v41_dspark_peer_post_norm_quant_gaudi2"
                elif args.candidate == "dense_kn":
                    expected = "aten.mm.default"
                elif args.candidate == "dense_bits12":
                    expected = "custom_deepseek_v41_dense_bits12_projection_bf16_gaudi2"
                elif args.candidate == "n512_decode":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_n512_decode_fp8_gaudi2"
                elif args.candidate in ("control_fp8", "control_fp8_pair"):
                    expected = ("custom_deepseek_v41_control_fp8_pair_gaudi2" if args.candidate == "control_fp8_pair"
                                else "custom_deepseek_v41_control_fp8_rrms_gaudi2")
                elif args.candidate == "moe_peer_post":
                    expected = "custom_deepseek_v41_dspark_moe_peer_post_gaudi2"
                elif args.candidate == "scaled_w13":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_scaled_w13_fp8_gaudi2"
                elif args.candidate == "explicit_steps":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_explicit_steps_fp8_gaudi2"
                elif args.candidate == "mhc_gate_packet":
                    expected="custom_deepseek_v41_mhc_gates_post_gaudi2"
                elif args.candidate == "expert_consumer_stitch":
                    expected=("custom_deepseek_v41_expert_n256_moe_token_wide_physical_silu_fp8_gaudi2"
                              if os.getenv("VLLM_HPU_DSV41_DSPARK_EXPERT_CONSUMER_STITCH_PAIR") == "1" else
                              "custom_deepseek_v41_expert_n256_moe_token_wide_slicer_stitch_fp8_gaudi2")
                elif args.candidate == "w2_channels":
                    expected="custom_deepseek_v41_expert_n256_moe_w2_channels_fp8_gaudi2"
                elif args.candidate == "w2_reduce_n256":
                    expected="custom_deepseek_v41_expert_n256_moe_w2_reduce_n256_fp8_gaudi2"
                elif args.candidate == "router_batched_f32":
                    expected = "custom_deepseek_v41_router_batched_f32_gaudi2"
                elif args.candidate == "router_shared_bf16":
                    expected="custom_deepseek_v41_router_shared_scaled_gaudi2"
                elif args.candidate == "cooperative_silu":
                    expected="custom_deepseek_v41_expert_n256_moe_cooperative_silu_fp8_gaudi2"
                elif args.candidate == "shared_scale":
                    expected="custom_deepseek_v41_expert_n256_moe_w2_shared_scale_n256_fp8_gaudi2"
                    selected=sum(count for name,count in operators.items() if expected in name)
                elif args.candidate == "w2_ready_scale":
                    expected="custom_deepseek_v41_expert_n256_moe_w2_ready_scale_fp8_gaudi2"
                elif args.candidate == "w2_k_pipeline":
                    expected="custom_deepseek_v41_expert_n256_moe_w2_k_pipeline_fp8_gaudi2"
                elif args.candidate == "w13_k_pipeline":
                    expected=("custom_deepseek_v41_expert_n256_moe_w13_k_pipeline_k512_fp8_gaudi2"
                              if os.getenv("VLLM_HPU_DSV41_DSPARK_EXPERT_K_TILE", "128") == "512" else
                              "custom_deepseek_v41_expert_n256_moe_w13_k_pipeline_four_fp8_gaudi2"
                              if os.getenv("VLLM_HPU_DSV41_DSPARK_W13_K_PIPELINE_STAGES", "2") == "4" else
                              "custom_deepseek_v41_expert_n256_moe_w13_k_pipeline_fp8_gaudi2")
                elif args.candidate == "split_scale_planes":
                    expected="custom_deepseek_v41_expert_n256_moe_split_scale_planes_fp8_gaudi2"
                elif args.candidate == "silu_decode_affine":
                    expected="custom_deepseek_v41_expert_n256_moe_token_wide_silu_decode_affine_fp8_gaudi2"
                elif args.candidate == "silu_decode":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_silu_decode_fp8_gaudi2"
                elif args.candidate == "norm_roundtrip":
                    expected = "custom_deepseek_v41_norm_roundtrip_bf16_gaudi2"
                elif args.candidate == "input_norm":
                    expected = ("custom_deepseek_v41_input_norm_bf16_gaudi2" if hasattr(torch.ops.custom_op,
                        "custom_deepseek_v41_input_norm_bf16_gaudi2") else
                        "custom_deepseek_v41_attention_norm_bf16_gaudi2")
                elif args.candidate == "full_row":
                    expected = "custom_deepseek_v41_logical_full_row_gaudi2"
                elif args.candidate == "fp4_table":
                    expected = "custom_deepseek_v41_logical_fp4_table_gaudi2"
                elif args.candidate == "row_cache":
                    expected = "custom_deepseek_v41_logical_row_cache_gaudi2"
                elif args.candidate == "shared_finalize":
                    expected = "custom_deepseek_v41_moe_shared_finalize_gaudi2"
                elif args.candidate == "wide_codec":
                    expected = "custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2"
                elif args.candidate == "native_target_input":
                    expected = "embedding"
                elif args.candidate == "runtime_selection":
                    expected = "custom_deepseek_v41_index_threshold_gaudi2"
                elif args.candidate == "fp8_prologue":
                    expected = ("custom_deepseek_v41_fp8_qkv_prologue_split_gaudi2" if args.independent_prologue else
                                "custom_deepseek_v41_fp8_qkv_prologue_gaudi2")
                elif args.candidate in ("c1_dense_chain", "input_fp8"):
                    expected = "custom_deepseek_v41_attention_norm_quant_gaudi2"
                elif args.candidate in ("output_fp8", "dense_restore"):
                    expected = "custom_deepseek_v41_dense_fp8_gaudi2"
                elif args.candidate == "pv_rope":
                    expected = "custom_deepseek_v41_logical_mla_pv_rope_gaudi2"
                elif args.candidate == "shared_prequant":
                    expected = "custom_deepseek_v41_dense_quant_gaudi2"
                elif args.candidate == "shared_fp8":
                    expected = "custom_deepseek_v41_shared_silu_quant_gaudi2"
                elif args.candidate == "candidate_keys":
                    expected = "custom_deepseek_v41_candidate_mirror_keys_gaudi2"
                elif args.candidate == "affine_route":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_affine_route_fp8_gaudi2"
                elif args.candidate == "transpose_sat":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_transpose_mme_fp8_gaudi2"
                elif args.candidate == "threshold_selection":
                    expected = "custom_deepseek_v41_index_threshold_gaudi2"
                elif args.candidate == "silu_affine":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_silu_affine_fp8_gaudi2"
                elif args.candidate == "scale_cache":
                    expected = "custom_deepseek_v41_logical_scale_cache_gaudi2"
                elif args.candidate == "unpaired_w13":
                    expected = "custom_deepseek_v41_expert_n256_moe_unpaired_w13_fp8_gaudi2"
                elif args.candidate == "k64_partition":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_k64_partition_fp8_gaudi2"
                elif args.candidate == "batch6_kv":
                    expected = "custom_deepseek_v41_logical_scale_batch6_gaudi2"
                elif args.candidate == "merge_cache":
                    expected = "custom_deepseek_v41_logical_merge_cache_gaudi2"
                elif args.candidate == "stacked_pv":
                    expected = "custom_deepseek_v41_logical_scale_stacked_pv_gaudi2"
                elif args.candidate == "pair_pv":
                    expected = "custom_deepseek_v41_logical_scale_pair_pv_gaudi2"
                elif args.candidate == "decoded_publish":
                    expected = "custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2"
                elif args.candidate == "main_mirror":
                    expected = "custom_deepseek_v41_logical_main_mirror_gaudi2"
                elif args.candidate == "w13_unroll":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_unroll_steps_fp8_gaudi2"
                elif args.candidate == "channel_silu":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_channel_silu_fp8_gaudi2"
                elif args.candidate == "group_pipeline":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_group_pipe_fp8_gaudi2"
                elif args.candidate == "compact_kv_mme":
                    expected = "custom_deepseek_v41_logical_mla_compact_mme_gaudi2"
                elif args.candidate == "exp_pv":
                    expected = "custom_deepseek_v41_logical_scale_exp_pv_gaudi2"
                elif args.candidate == "wo_handoff":
                    expected = "custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2"
                elif args.candidate == "resident_constants":
                    expected = "custom_deepseek_v41_logical_scale_cache_gaudi2"
                elif args.candidate == "shared_kv_mme":
                    expected = "custom_deepseek_v41_logical_mla_shared_mme_gaudi2"
                elif args.candidate == "mla_coord_cache":
                    expected = "custom_deepseek_v41_logical_mla_coord_cached_gaudi2"
                elif args.candidate == "hash_mla":
                    expected = "custom_deepseek_v41_logical_mla_hash_gaudi2"
                elif args.candidate == "mhc_deferred":
                    expected = "custom_deepseek_v41_mhc_mme_post_collapse_gaudi2"
                elif args.candidate == "mhc_statistics":
                    expected = "custom_deepseek_v41_mhc_statistics_epilogue_gaudi2"
                elif args.candidate == "mhc_mme_epilogue":
                    expected = "custom_deepseek_v41_mhc_mme_epilogue_gaudi2"
                elif args.candidate in ("mhc_weight_reuse", "mhc_control_tiles"):
                    expected = ("custom_deepseek_v41_mhc_control_tiles_gaudi2"
                                if args.candidate == "mhc_control_tiles" else
                                "custom_deepseek_v41_mhc_control_reuse_gaudi2")
                elif args.candidate == "q_prologue":
                    expected = "custom_deepseek_v41_q_"
                elif args.candidate == "qkv_publish":
                    expected = "custom_deepseek_v41_dspark_qkv_projection_publish_gaudi2"
                elif args.candidate == "kv_publish":
                    expected = "custom_deepseek_v41_dspark_kv_norm_publish_ordered_gaudi2"
                elif args.candidate == "physical_role_silu":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_physical_role_silu_fp8_gaudi2"
                elif args.candidate == "physical_silu":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_physical_silu_fp8_gaudi2"
                elif args.candidate == "silu_unroll":
                    expected = "custom_deepseek_v41_expert_n256_moe_token_wide_silu_unroll_fp8_gaudi2"
                elif args.candidate == "router_bf16":
                    expected = "custom_deepseek_v41_bf16_linear_f32_gaudi2"
                elif args.candidate == "peer_post_norm":
                    expected = "custom_deepseek_v41_dspark_peer_post_norm_quant_gaudi2"
                if args.candidate == "native_page_coalesce":
                    from vllm_gaudi.ops.tp2_prepared_plan import _native_entries

                    graph = _native_entries[plan][0]
                    info = list(graph.joint_info())
                    row = dict(arm=arm, joint_info=info, segments=graph.segment_count(),
                               collectives=graph.collective_count(), commands=graph.captured_command_count())
                    rows = report.setdefault("native_publication", [])
                    if arm:
                        if any(info[i] != rows[0]["joint_info"][i] for i in (1, 2, 3, 4, 5, 8)):
                            raise RuntimeError("Coalescing changed native packets or completion dependencies")
                        if dict(operators) != report["captured_arms"][0]["operators"]:
                            raise RuntimeError("Coalescing changed the Tensor program")
                    selected = int(arm and info[7] < rows[0]["joint_info"][7])
                    rows.append(row)
                    expected = "lower native SCAL submissions for identical captured packets"
                elif args.candidate == "recipe_constants":
                    expected = "compiled immutable constant sections"
                    if arm:
                        from deepseek_v41_recipe_constant_sections import inspect

                        paths = sorted((root / f"recipes/rank{rank}").glob("*.recipe"))
                        records = [inspect(path) for path in paths]
                        sections = sum(row["constant_section_bytes"] > 1048576 for row in records)
                        const_bytes = sum(row["constant_section_bytes"] for row in records)
                        report["compiled_constant_sections"] = records
                        report["constant_section_gate"] = dict(recipes=sections, bytes=const_bytes,
                                                               covered_layers=stop-start)
                        contracts = report["recipe_constant_contracts"]
                        if (sections < stop-start or const_bytes < (stop-start)*20*1048576
                                or len(contracts) != stop-start
                                or not all(c["tensors"] and all(v["marked"] for v in c["tensors"])
                                           for c in contracts)):
                            save()
                            raise RuntimeError("Immutable weight marks did not reach compiled constant sections")
                        # A layer can span several recipes; this is layer
                        # coverage, not a claim of one physical op per layer.
                        selected = len(contracts)
                    else:
                        selected = 0
                elif args.candidate == "mhc_high_plane":
                    expected = "actual prepared mHC MME weight shape"
                    selected = sum(shape == [24 if arm else 48, 20480] for shape in control_weight_shapes)
                    report.setdefault("mhc_weight_shapes", []).append(dict(arm=arm, shapes=control_weight_shapes))
                elif args.candidate == "output_layout":
                    expected = "actual wo_a RHS BMM stride"
                    expected_stride = [1, program.layers[0].attention.heads * 512 //
                                       program.layers[0].attention.groups] if arm else [1024, 1]
                    selected = sum(
                        bool(row["operands"][1]) and row["operands"][1]["shape"][-2:] ==
                        [program.layers[0].attention.heads * 512 // program.layers[0].attention.groups, 1024]
                        and row["operands"][1]["stride"][-2:] == expected_stride
                        for row in output_bmm_signatures)
                    report.setdefault("output_layout_signatures", []).append(
                        dict(arm=arm, expected_stride=expected_stride, signatures=output_bmm_signatures))
                elif args.candidate == "index_query_local":
                    expected = "actual query/gains peer gathers removed by two full-head projection GEMMs"
                    # Exchange nodes live between recipes, so the compute-only
                    # operator Counter intentionally cannot count them.
                    current = _native_entries[plan][0].collective_count()
                    parent = report["captured_arms"][0]["native_collectives"] if arm else current
                    selected = parent - current
                    local_owners = sum(layer.attention.owns_index and layer.attention.search_length //
                                       layer.attention.ratio > 512 for layer in program.layers)
                    global_owners = sum(program.search_length // text["compress_ratios"][i] > 512
                                        for i in text["index_source_layer_ids"] if i < 40)
                    report["index_query_scope"] = dict(local_owners=local_owners, global_owners=global_owners,
                                                       original_peer_gathers=parent, current_peer_gathers=current)
                    report["projection_multiplier"] = global_owners / local_owners
                    report["projected_layer_count"] = global_owners
                elif args.candidate in ("mhc_overlap", "mhc_producer", "tensor_ready_peer", "mhc_native_dependencies"):
                    merged = args.candidate in ("mhc_producer", "tensor_ready_peer")
                    selected = int((mhc_producers if merged else mhc_partitions) > 0)
                    expected = ("compiler mHC split with native explicit producer/consumer dependencies"
                                if args.candidate == "mhc_native_dependencies" else
                                "independent control MME branch joined to payload producer")
                    if arm and not selected:
                        report["overlap_noop"] = dict(partitions=mhc_partitions, operators=dict(operators))
                        save()
                        raise RuntimeError("Shared mHC transform found no movable production branch; no timing credit")
                elif args.candidate == "router_shared":
                    current = operators.get("aten.linear", 0)
                    parent = report["captured_arms"][0]["operators"].get("aten.linear", 0) if arm else current
                    selected = parent - current
                elif args.candidate == "shared_prequant":
                    current = sum(count for name, count in operators.items() if expected in name)
                    parent_count = (sum(count for name, count in report["captured_arms"][0]["operators"].items()
                                        if expected in name) if arm else current)
                    selected = parent_count - current
                elif args.candidate == "shared_gate_up":
                    for layer in program.layers:
                        if bool(layer.moe.shared_gate_up) != bool(arm):
                            raise RuntimeError("Shared gate/up dispatch differs from the selected arm")
                        if arm and (
                            layer.moe.shared_gate_up_weight is None
                            or layer.moe.weights.shared_experts.w1.weight.device.type != "meta"
                        ):
                            raise RuntimeError("The shared gate/up weight was not actually prepared")
                    selected = arm * (stop - start)
                    expected = "prepared shared gate/up constant"
                else:
                    ordered = expected.removesuffix("_gaudi2") + "_ordered_gaudi2"
                    selected = sum(
                        count for target, count in operators.items() if expected in target or ordered in target
                    )
                required_calls = (2 if args.candidate in (
                    "dense_restore", "mhc_deferred", "mhc_high_plane", "control_fp8", "control_fp8_pair",
                    "mhc_mme_epilogue",
                    "mhc_statistics", "mhc_weight_reuse",
                    "mhc_control_tiles") else 1) * (stop - start)
                if args.candidate == "router_ready_fp8":
                    eligible = [layer.layer for layer in program.layers
                                if layer.moe.n256_fp8 and layer.moe.n256_fused]
                    required_calls = len(eligible)
                    report["router_ready_eligible_layers"] = eligible
                if args.candidate in ("qk_flat", "qk_flat_direct", "stream_exp"):
                    local = sum(text["compress_ratios"][i] in (1, 2) for i in range(start, stop))
                    global_calls = sum(r in (1, 2) for r in text["compress_ratios"][:40])
                    required_calls = local
                    report["projection_multiplier"] = global_calls / local
                    report["projected_layer_count"] = global_calls
                    report["qk_flat_scope"] = dict(local_layers=local, global_layers=global_calls,
                                                  preserves_pv=True)
                if args.candidate == "peer_post_collapse":
                    required_calls = 2 * (stop-start)
                    report["projection_multiplier"] = 40 / (stop-start)
                    report["projected_layer_count"] = 40
                    report["peer_post_scope"] = dict(local_calls=required_calls, full_calls=80,
                                                    includes_group_tails=True)
                if args.candidate == "mhc_gate_packet":
                    local = stop-start+sum(i+1<stop and i%4!=3 and i+1 not in text["engram_layer_ids"]
                                          for i in range(start,stop))
                    global_calls = 40+sum(i<39 and i%4!=3 and i+1 not in text["engram_layer_ids"]
                                          for i in range(40))
                    required_calls = local
                    report["projection_multiplier"] = global_calls/local
                    report["projected_layer_count"] = global_calls
                    report["gate_packet_scope"] = dict(local_post_calls=local, full_post_calls=global_calls,
                                                       real_layers=stop-start)
                if args.candidate == "moe_peer_post":
                    local = [i for i in range(start, stop) if i % 4 != 3 and i + 1 not in text["engram_layer_ids"]]
                    global_layers = [i for i in range(40) if i % 4 != 3 and i + 1 not in text["engram_layer_ids"]]
                    required_calls = len(local)
                    report["projection_multiplier"] = len(global_layers) / len(local)
                    report["projected_layer_count"] = len(global_layers)
                    report["moe_peer_post_scope"] = dict(local_layers=local, global_layers=global_layers,
                                                         preserves_existing_collapse_boundary=True)
                if args.candidate == "ordered_peer_sum":
                    # Attention and MoE each consume one ordered peer sum.
                    required_calls = 2 * (stop - start)
                if args.candidate in ("dense_kn", "dense_bits12"):
                    required_calls = 3 * (stop - start)
                if args.candidate in ("hw_dense", "hw_dense_fused", "norm_roundtrip"):
                    required_calls = 2 * (stop - start)
                if args.candidate == "rope_coherent":
                    parent_operators = (report["captured_arms"][0]["operators"] if arm else operators)
                    required_calls = sum(count for name, count in parent_operators.items() if
                        "custom_deepseek_v41_rope_bf16_gaudi2" in name or
                        "custom_deepseek_v41_rope_inverse_bf16_gaudi2" in name)
                if args.candidate in ("mhc_overlap", "mhc_producer", "tensor_ready_peer",
                                      "mhc_native_dependencies", "native_target_input", "native_page_coalesce"):
                    required_calls = 1
                if args.candidate == "index_query_local":
                    required_calls = 2 * report["index_query_scope"]["local_owners"]
                if args.candidate in ("main_single_bank", "coherent_swa", "main_adjacent_pv", "swa_source_reuse"):
                    required_calls=report["main_reuse_scope"]["local_reuse"]
                if args.candidate == "compressor_sequence":
                    required_calls=report["compressor_scope"]["local_owners"]
                if args.candidate == "main_single_bank":
                    publisher = "custom_deepseek_v41_main_single_bank_publish_mla_gaudi2"
                    published = sum(count for name, count in operators.items() if publisher in name)
                    if published != arm * report["main_reuse_scope"]["local_publish"]:
                        raise AssertionError("Single-bank publisher count disagrees with ownership keys")
                if args.candidate in ("layer_main_reuse", "layer_main_split"):
                    required_calls = report["main_reuse_scope"]["local_reuse"]
                    publisher = ("custom_deepseek_v41_main_split_publish_mla_gaudi2"
                                 if args.candidate == "layer_main_split"
                                 else "custom_deepseek_v41_main_batch_publish_mla_gaudi2")
                    published = sum(count for name, count in operators.items() if publisher in name)
                    if published != arm * report["main_reuse_scope"]["local_publish"]:
                        raise AssertionError("Selected main publisher count disagrees with actual ownership keys")
                if args.candidate == "decoded_publish":
                    required_calls = sum(layer.attention.owns_kv and layer.attention.ratio in (1, 2)
                                         for layer in program.layers)
                    consumed = sum(count for target, count in operators.items()
                                   if "custom_deepseek_v41_logical_main_mirror_gaudi2" in target)
                    if consumed != arm * (stop - start):
                        raise AssertionError("Decoded publication did not reach every real MLA consumer")
                if args.candidate == "threshold_selection":
                    required_calls = 2 if start == program.layers[0].attention.candidate_source else 1
                elif args.candidate == "candidate_keys":
                    required_calls = sum(layer.attention.owns_index and layer.layer > layer.attention.candidate_source
                                         for layer in program.layers)
                if args.candidate == "wide_codec":
                    scalar = "custom_deepseek_v41_quant_roundtrip_bf16_gaudi2"
                    parent_ops = report["captured_arms"][0]["operators"] if arm else dict(operators)
                    required_calls = sum(count for name, count in parent_ops.items() if scalar in name)
                    if required_calls == 0:
                        raise AssertionError("Production codec parent has no scalar quantization consumers")
                if args.candidate == "runtime_selection":
                    active = [layer for layer in program.layers
                              if layer.attention.owns_index and layer.attention.ratio == 1
                              and layer.layer > layer.attention.candidate_source
                              and program.search_length // layer.attention.ratio > 512]
                    retained_source_calls = sum(
                        (2 if program.search_length // layer.attention.ratio > 16384 else 1)
                        for layer in program.layers
                        if layer.attention.owns_index and layer.layer == layer.attention.candidate_source
                        and layer.attention.decode_threshold_selection
                        and program.search_length // layer.attention.ratio > 512)
                    required_arm_calls = retained_source_calls + (len(active) if arm else 0)
                    if not all(layer.attention.c6_index_reduce and layer.attention.c6_index_wide_scores
                               for layer in active):
                        raise ValueError("C1 selector reuse requires the unchanged causal BF16 score producer")
                    global_owners = [index for index in text["index_source_layer_ids"]
                                     if text["compress_ratios"][index] == 1
                                     and index > text["candidate_source_layer_id"]
                                     and program.search_length // text["compress_ratios"][index] > 512]
                    report["projection_multiplier"] = len(global_owners) / len(active)
                    report["projected_layer_count"] = len(global_owners)
                    report["selection_scope"] = dict(local_owners=[layer.layer for layer in active],
                                                      global_owners=global_owners,
                                                      preserves_existing_score_producer=True)
                elif args.candidate == "startup_frontend":
                    selected = int(bool(arm) and bool(report["startup_preparation"][-1]["cache"]))
                    required_arm_calls = arm
                    if arm and not args.frontend_first:
                        parent = report["captured_arms"][0]
                        report["frontend_operator_delta"] = {
                            name: operators[name] - parent["operators"].get(name, 0)
                            for name in set(operators) | set(parent["operators"])
                            if operators[name] != parent["operators"].get(name, 0)}
                        if _native_entries[plan][0].collective_count() != parent["native_collectives"]:
                            raise RuntimeError("Frontend restoration changed the native communication contract")
                else:
                    required_arm_calls = (required_calls if arm or args.candidate in
                                          ("resident_constants", "output_layout", "mhc_high_plane") else 0)
                if selected != required_arm_calls:
                    report["capture_operator_mismatch"] = dict(arm=arm, selected=selected,
                                                               required=required_arm_calls, operators=dict(operators))
                    save()
                    raise RuntimeError(f"Captured arm {arm} has {selected} candidate operators; "
                                       f"expected {required_arm_calls}")
                stacked_calls = {}
                for stacked in args.stack_candidates:
                    if stacked == "index_query_local":
                        current = _native_entries[plan][0].collective_count()
                        parent = report["captured_arms"][0]["native_collectives"] if arm else current
                        count = parent - current
                        owners = sum(layer.attention.uses_local_index_queries(raw[0]["positions"].numel())
                                     and layer.attention.owns_index for layer in program.layers)
                        if count != (2 * owners if arm else 0):
                            raise RuntimeError("Local index queries must remove exactly two exchanges per owner")
                        stacked_calls[stacked] = count
                        continue
                    marker = {"norm_roundtrip": "custom_deepseek_v41_norm_roundtrip_bf16_gaudi2",
                              "w13_k_pipeline": ("custom_deepseek_v41_expert_n256_moe_w13_k_pipeline_k512_fp8_gaudi2"
                              if os.getenv("VLLM_HPU_DSV41_DSPARK_EXPERT_K_TILE", "128") == "512" else
                              "custom_deepseek_v41_expert_n256_moe_w13_k_pipeline_four_fp8_gaudi2"
                                  if os.getenv("VLLM_HPU_DSV41_DSPARK_W13_K_PIPELINE_STAGES", "2") == "4" else
                                  "custom_deepseek_v41_expert_n256_moe_w13_k_pipeline_fp8_gaudi2"),
                              "mhc_gate_packet": "custom_deepseek_v41_mhc_gates_post_gaudi2",
                              "w2_reduce_n256":
                                  "custom_deepseek_v41_expert_n256_moe_w2_reduce_n256_fp8_gaudi2"}[stacked]
                    count = sum(amount for name, amount in operators.items() if marker in name)
                    if bool(count) != bool(arm):
                        raise RuntimeError(f"Stacked {stacked} was not selected exclusively in candidate arm")
                    stacked_calls[stacked] = count
                report.setdefault("stacked_operator_coverage", []).append(dict(arm=arm, calls=stacked_calls))
                if args.candidate in ("mhc_producer", "tensor_ready_peer") and arm:
                    parent_recipes = len(report["captured_arms"][0]["recipes"])
                    if len(recipes) > parent_recipes:
                        raise RuntimeError("mHC producer merge added recipe boundaries; no timing vote")
                    report["mhc_producer_scope"] = dict(merged_calls=mhc_producers, parent_recipes=parent_recipes,
                                                         candidate_recipes=len(recipes))
                if args.candidate == "tensor_ready_peer" and arm:
                    graph = _native_entries[plan][0]
                    signal_info = graph.tensor_ready_info()
                    if len(signal_info) != 5 or sum(value > 0 for value in signal_info[4]) != mhc_producers:
                        raise RuntimeError("Actual tensor-ready signal coverage differs from producer coverage")
                    report["tensor_ready_scope"] = dict(signal_metadata=signal_info,
                                                       actual_early_peer_points=mhc_producers,
                                                       final_retirement_unchanged=True)
                report.setdefault("captured_arms", []).append(
                    dict(arm=arm, candidate_calls=selected, operators=dict(operators), recipes=recipes,
                         native_collectives=_native_entries[plan][0].collective_count(),
                         native_dependency_flag=os.environ.get(native_dependency_flag, "0"))
                )
                save()
            if args.candidate == "startup_frontend" and args.frontend_first:
                plans.reverse()
                programs.reverse()
                report["captured_arms"].sort(key=lambda item: item["arm"])
                parent, changed = report["captured_arms"]
                if changed["native_collectives"] != parent["native_collectives"]:
                    raise RuntimeError("Frontend restoration changed the native communication contract")
                report["frontend_operator_delta"] = {
                    name: changed["operators"].get(name, 0) - parent["operators"].get(name, 0)
                    for name in set(changed["operators"]) | set(parent["operators"])
                    if changed["operators"].get(name, 0) != parent["operators"].get(name, 0)}
            cases = []
            states = []
            for data in raw:
                source = data["groups"][max(0, args.group - 1)]
                cases.append(
                    tuple(v.to("hpu") for v in (source["residual"], source["pre"], data["positions"], data["ids"]))
                )
                active = {}
                for name in data["decoder_state_files"]:
                    if name.startswith("shared.sources.") and name.split(".")[2] not in sources:
                        continue
                    if not (
                        name.startswith("shared.") or any(name.startswith(f"layers.{i}.") for i in range(start, stop))
                    ):
                        continue
                    # The partial ratio-2 group deliberately has no ratio-1
                    # candidate pool. It is not a mutable state of this plan.
                    if name == "shared.candidate_pool" and shared.candidate_pool is None:
                        continue
                    destination = selection_name if name.startswith("shared.topk.") else name
                    active[destination] = source_state(data, name).to("hpu")
                if args.candidate in ("main_mirror", "decoded_publish"):
                    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4

                    table = active["shared.block_table"].cpu()
                    for key, cache in shared.sources.items():
                        capacity = cache.main_mirror.shape[0]
                        mirror = torch.zeros((capacity, 512), dtype=torch.bfloat16)
                        count = min(capacity, int(data["positions"][0]) // cache.ratio)
                        logical = torch.arange(count, dtype=torch.int64)
                        width = 128 // cache.ratio
                        physical = table[logical // width].long() * width + logical.remainder(width)
                        packed = active[f"shared.sources.{key}.main"].cpu()[physical]
                        mirror[:count] = unpack_fp4(packed, group=16).bfloat16()
                        active[f"shared.sources.{key}.main_mirror"] = mirror.to("hpu")
                states.append(active)

            def reset(arm, case):
                program = programs[arm]
                for name, value in states[case].items():
                    parts = name.split(".")
                    if parts[0] == "layers":
                        parts[1] = str(int(parts[1]) - start)
                    owner_name, _, leaf = ".".join(parts).rpartition(".")
                    getattr(program.get_submodule(owner_name), leaf).copy_(value)
                if args.candidate == "swa_cache" and arm:
                    from vllm_gaudi.ops.deepseek_v41_math import unpack_swa

                    for layer in program.layers:
                        layer.attention.swa_decoded.swa.copy_(unpack_swa(layer.attention.swa).bfloat16())
                torch.hpu.synchronize()

            input_producers = []
            if args.fixed_prefix_acceptance:
                from deepseek_v41_fixed_prefix_target_gate import qualify

                qualify(programs, plans, raw, engram_cases, reset, root, rank, report, save,
                        timing=not args.quality_only)
                return

            if args.candidate == "native_target_input":
                from vllm_gaudi.models.deepseek_v41_program import PreparedInput
                from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
                for program in programs:
                    input_producers.append(torch.compile(
                        PreparedInput(program.weights.embed, rank, reduce), backend=make_backend(),
                        fullgraph=True, dynamic=False))
                # Warm the actual serving producer before any device timer.
                for producer in input_producers:
                    for values in cases:
                        producer(values[3])
                torch.hpu.synchronize()

            def invoke(arm, case):
                plan = plans[arm]
                if args.candidate != "native_target_input":
                    return plan(*cases[case], ())
                _, _, positions, ids = cases[case]
                if arm:
                    residual, pre = cases[case][:2]
                else:
                    residual, pre = input_producers[arm](ids)
                return plan(residual, pre, positions, ids, engram_cases[case], native_input=bool(arm))

            for case in range(len(cases)):
                results, caches = [], []
                mirror_exact = True
                operands = []
                selection_overlap = []
                for arm, plan in enumerate(plans):
                    reset(arm, case)
                    results.append(tuple(v.cpu() if v is not None else None for v in invoke(arm, case)))
                    caches.append(tuple(layer.attention.swa.cpu() for layer in programs[arm].layers))
                    if args.candidate == "swa_cache" and arm:
                        from vllm_gaudi.ops.deepseek_v41_math import unpack_swa

                        mirror_exact = all(
                            torch.equal(layer.attention.swa_decoded.swa.cpu(), unpack_swa(cache).bfloat16())
                            for layer, cache in zip(programs[arm].layers, caches[-1], strict=True)
                        )
                    if args.candidate == "threshold_selection":
                        selected = programs[arm].layers[-1].attention.selection.indices[:6].cpu()
                        selection_overlap.append(dict(arm=arm, pairs=[
                            dict(rows=[i, i + 1],
                                 same_slot_fraction=float((selected[i] == selected[i + 1]).float().mean()),
                                 set_overlap_fraction=len(
                                     set(selected[i].tolist()) & set(selected[i + 1].tolist())) / 512)
                            for i in (0, 2, 4)]))
                    if args.attention_oracle:
                        from deepseek_v41_attention_oracle import actual_operands

                        owner = programs[arm].layers[1 if args.candidate == "main_adjacent_pv" else 0].attention
                        operands.append(actual_operands(owner, cases[case][2]))
                exact = all(
                    (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                    for a, b in zip(*results, strict=True)
                )
                exact = exact and all(torch.equal(a, b) for a, b in zip(*caches, strict=True))
                exact = exact and mirror_exact
                agreement = [None] * tp
                dist.all_gather_object(agreement, exact, group=get_tp_group().cpu_group)
                errors = [
                    None
                    if a is None or b is None
                    else dict(
                        dtype=str(a.dtype),
                        shape=list(a.shape),
                        max_abs=float((a.float() - b.float()).abs().max()),
                        rms=float((a.float() - b.float()).square().mean().sqrt()),
                    )
                    for a, b in zip(*results, strict=True)
                ]
                report["checks"].append(dict(case=case, outputs_and_swa_exact=exact, errors=errors,
                                             swa_mirror_exact=mirror_exact))
                if selection_overlap:
                    report["checks"][-1]["selection_overlap"] = selection_overlap
                if args.peer_input_oracle:
                    from deepseek_v41_peer_input_oracle import operands, reduction_operands

                    packet = [[dict(layer=layer.layer, attention=operands(layer.attention),
                                    moe=reduction_operands(layer.moe)) for layer in program.layers]
                              for program in programs]
                    torch.save(packet, root / f"peer-inputs-rank{rank}-case{case}.pt")
                    report["checks"][-1]["peer_input_diagnostic_only"] = True
                    save()
                    continue
                if args.attention_oracle:
                    from deepseek_v41_attention_oracle import audit

                    torch.save(operands, root / f"attention-operands-rank{rank}-case{case}.pt")
                    report.setdefault("attention_oracle", []).append(audit(*operands))
                    save()
                    continue
                if args.selection_oracle:
                    from deepseek_v41_selection_oracle import audit

                    report.setdefault("selection_oracle", []).append(audit(programs[1].layers[0].attention))
                    save()
                    continue
                if args.candidate in ("main_mirror", "decoded_publish"):
                    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4

                    mirror_checks = []
                    positions = cases[case][2]
                    for key, cache in shared.sources.items():
                        # New rows plus actually selected rows exercise both
                        # canonical writer and consumer addresses.
                        selected = programs[1].layers[0].attention.selection.indices[:positions.numel()]
                        rows = torch.cat((selected.flatten(), (positions // cache.ratio).to(torch.int32)))
                        rows = rows.clamp(0, cache.main_mirror.shape[0] - 1).long()
                        physical = shared.physical_rows(rows.to(torch.int32), cache.ratio).long()
                        packed = cache.main.index_select(0, physical).cpu()
                        actual = cache.main_mirror.index_select(0, rows).cpu()
                        matched = torch.equal(actual, unpack_fp4(packed, group=16).bfloat16())
                        mirror_checks.append(dict(source=key, canonical_selected_and_new_rows_exact=matched))
                    report["checks"][-1]["main_mirror"] = mirror_checks
                    if not all(row["canonical_selected_and_new_rows_exact"] for row in mirror_checks):
                        raise AssertionError("Main mirror differs from authoritative packed cache")
                if args.c1_projection_oracle:
                    from deepseek_v41_projection_oracle import audit, audit_shared, audit_publish

                    if args.candidate == "compressor_sequence":
                        from deepseek_v41_compressor_oracle import audit as audit_compressor

                        result = audit_compressor(programs[0].layers[0].attention, programs[1].layers[0].attention,
                                                  cases[case][2], root / f"compressor-rank{rank}-case{case}.pt")
                    elif args.candidate == "qkv_publish":
                        result = audit_publish(programs[1].layers[0].attention, cases[case][2])
                    elif args.candidate == "q_prologue":
                        from deepseek_v41_query_prologue_oracle import audit as audit_query

                        result = audit_query(programs[1].layers[0].attention, cases[case][2])
                    elif args.candidate == "router_bf16":
                        from deepseek_v41_router_oracle import audit as audit_router

                        result = audit_router(programs[1].layers[0].moe)
                    elif args.candidate == "mhc_mme_epilogue":
                        from deepseek_v41_mhc_epilogue_oracle import audit as audit_mhc

                        result = audit_mhc(programs[1].layers[0], cases[case][0])
                    elif args.candidate in ("mhc_weight_reuse", "mhc_control_tiles"):
                        from deepseek_v41_mhc_control_oracle import audit as audit_control

                        result = audit_control(programs[1].layers[0], cases[case][0])
                    elif args.candidate in ("shared_gate_up", "shared_fp8", "shared_prequant"):
                        result = audit_shared(programs[1].layers[0].moe)
                    else:
                        result = audit(programs[1].layers[0].attention, cases[case][2])
                    report.setdefault("projection_oracle", []).append(result)
                    save()
                    continue
                if args.constant_oracle:
                    from deepseek_v41_recipe_constants_oracle import audit as audit_constants
                    if args.candidate != "recipe_constants":
                        raise ValueError("Constant oracle belongs to the immutable recipe capability")
                    result = audit_constants(programs[0].layers[0], programs[1].layers[0],
                                             cases[case][0], cases[case][1], cases[case][2],
                                             root / f"constant-oracle-rank{rank}-case{case}.pt")
                    report.setdefault("constant_oracle", []).append(result)
                    report["checks"].append(dict(case=case, outputs_and_swa_exact=all(agreement),
                                                 errors=errors, swa_mirror_exact=mirror_exact,
                                                 precision_oracle_qualified=False))
                    save()
                    continue
                finite = all(v is None or bool(torch.isfinite(v).all()) for arm_outputs in results for v in arm_outputs)
                finite = finite and all(bool(torch.isfinite(v).all()) for arm_caches in caches for v in arm_caches)
                inline_qualified = False
                if args.candidate in ("hw_dense", "hw_dense_fused"):
                    from deepseek_v41_hw_dense_oracle import audit as audit_static_dense
                    result = audit_static_dense(programs[1].layers[0], cases[case][0], cases[case][1])
                    report.setdefault("hw_dense_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Static dense MME differs from its independently decoded operands")
                    inline_qualified = True
                    report["precision_oracle"] = "Static exponent-bias arithmetic checked; whole-Target alpha pending"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate == "router_shared_fused":
                    from deepseek_v41_router_shared_epilogue_oracle import audit as audit_joint_epilogue
                    result = audit_joint_epilogue(programs[1].layers[0], cases[case][0], cases[case][1])
                    report.setdefault("router_shared_epilogue_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Fused Router/shared epilogue changed the reference joint arithmetic")
                    inline_qualified = True
                    report["precision_oracle"] = "Joint FP8 epilogues exact; whole-model alpha remains pending"
                if args.inline_control_oracle:
                    from deepseek_v41_mhc_control_oracle import audit as audit_control

                    result = audit_control(programs[1].layers[0], cases[case][0])
                    report.setdefault("projection_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Tiled control differs from accepted C1 beyond the existing tolerance")
                    inline_qualified = True
                    report["precision_oracle"] = "Same-process accepted C1 control/RRMS, three real request cases"
                if args.candidate == "router_batched_f32":
                    from deepseek_v41_router_batched_f32_oracle import audit as audit_batched_f32
                    oracle=audit_batched_f32(programs[1].layers[0],cases[case][0],cases[case][1])
                    report.setdefault("router_batched_c1_oracle",[]).append(oracle)
                    save()
                    if not finite or not oracle["passed"]:
                        raise AssertionError("Batched FP32 Router exceeds accepted C1 FFN tolerance")
                    inline_qualified=True
                    report["precision_oracle"]=("Actual FFN inputs, FP32 accumulation and accepted "
                                                 "C1 routed/shared consumer")
                    report["teacher_forced_acceptance_qualified"]=False
                if args.candidate == "router_shared_bf16":
                    from vllm_gaudi.ops.deepseek_v41_math import unpack_swa
                    differences=[]
                    for parent, selected in zip(*results,strict=True):
                        if parent is None or selected is None:
                            if parent is not selected:
                                raise AssertionError("Joint BF16 changed state ownership")
                            continue
                        if not parent.dtype.is_floating_point:
                            if not torch.equal(parent,selected):
                                raise AssertionError("Joint BF16 changed discrete position state")
                            continue
                        delta=selected.float()-parent.float()
                        relative=float(delta.norm()/parent.float().norm().clamp_min(1e-30))
                        differences.append(dict(shape=list(parent.shape),relative_l2=relative,
                                                max_abs=float(delta.abs().max())))
                    codec_differences=[]
                    for parent, selected in zip(*caches,strict=True):
                        changed=(parent!=selected).any(-1)
                        if not changed.any():
                            codec_differences.append(dict(changed_rows=0,relative_l2=0.))
                            continue
                        x,y=unpack_swa(parent[changed]),unpack_swa(selected[changed])
                        delta=y.float()-x.float()
                        relative=float(delta.norm()/x.float().norm().clamp_min(1e-30))
                        codec_differences.append(dict(changed_rows=int(changed.sum()),relative_l2=relative,
                                                      max_abs=float(delta.abs().max())))
                    report.setdefault("router_shared_bf16_numeric",[]).append(
                        dict(outputs=differences,decoded_active_swa=codec_differences))
                    save()
                    from deepseek_v41_router_shared_bf16_oracle import audit as audit_joint_bf16
                    oracle=audit_joint_bf16(programs[1].layers[0],cases[case][0],cases[case][1])
                    report.setdefault("router_shared_bf16_c1_oracle",[]).append(oracle)
                    save()
                    if not finite or not oracle["passed"]:
                        raise AssertionError("Joint BF16 does not meet accepted C1 FFN/Router tolerance")
                    inline_qualified=True
                    report["precision_oracle"]=("Same C1 effective shared FP8 operands decoded exactly to BF16; "
                                                 "original Router BF16 preserved")
                    report["teacher_forced_acceptance_qualified"]=False
                if args.candidate in ("dense_kn", "dense_bits12"):
                    # Only the immutable BF16 operand orientation changes.
                    # Treat MME reduction-order differences explicitly; an
                    # actual timing win still needs whole-model acceptance.
                    differences = []
                    for a, b in zip((*results[0], *caches[0]), (*results[1], *caches[1]), strict=True):
                        if a is None or b is None:
                            continue
                        if a.dtype.is_floating_point:
                            relative = float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))
                            differences.append(dict(shape=list(a.shape), relative_l2=relative,
                                                    max_abs=float((a.float() - b.float()).abs().max())))
                        elif not torch.equal(a, b):
                            raise AssertionError("Dense operand layout changed discrete state")
                    if not finite or any(row["relative_l2"] > 0.002 for row in differences):
                        raise AssertionError("Dense operand orientation exceeded the established MME tolerance")
                    numeric_key = "dense_kn_tolerance" if args.candidate == "dense_kn" else "dense_projection_tolerance"
                    report.setdefault(numeric_key, []).append(differences)
                    inline_qualified = True
                    report["precision_oracle"] = "Unchanged BF16 operand bytes; MME scheduling tolerance L2 <=0.002"
                    report["teacher_forced_acceptance_qualified"] = False
                qualified = bool(args.precision_proof) or inline_qualified
                if args.candidate == "mhc_post_stats":
                    from deepseek_v41_mhc_post_stats_oracle import audit as audit_post_stats
                    result = audit_post_stats(programs[1].layers[0], cases[case][0], cases[case][1], cases[case][2])
                    report.setdefault("mhc_post_stats_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("C1 post statistics exceeded the existing FFN/control consumer envelope")
                    qualified = True
                    report["precision_oracle"] = "Actual attention/gates through C1 FFN dual quant and shared consumer"
                if args.candidate == "main_single_bank":
                    from deepseek_v41_layer_main_split_oracle import audit as audit_single_bank
                    result = audit_single_bank(programs[1].layers[0], cases[case][0], cases[case][1],
                                               cases[case][2], single_bank=True)
                    report.setdefault("main_single_bank_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Single BF16 KV bank exceeded the accepted C1 attention envelope")
                    qualified = True
                    report["precision_oracle"] = "Actual C1 codecs/current KV and independent FP64 attention"
                if args.candidate == "coherent_swa":
                    from deepseek_v41_layer_main_split_oracle import audit as audit_coherent_swa
                    result = audit_coherent_swa(programs[1].layers[0], cases[case][0], cases[case][1],
                                                cases[case][2], coherent=True)
                    report.setdefault("coherent_swa_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Shared SWA MME exceeded the C1 attention precision envelope")
                    qualified = True
                    report["precision_oracle"] = "Actual C1 codecs/current KV and independent FP64 attention"
                if args.candidate == "stream_exp":
                    from deepseek_v41_stream_exp_oracle import audit as audit_stream_exp
                    result = audit_stream_exp(programs[1].layers[0], cases[case][0], cases[case][1], cases[case][2])
                    report.setdefault("stream_exp_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Streaming C1 exponent PV exceeded the existing attention envelope")
                    qualified = True
                    report["precision_oracle"] = ("Actual C1 codecs/FP64 attention; "
                                                 "same BF16 and inverse RoPE boundaries")
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate in ("qk_flat", "qk_flat_direct") and not exact:
                    from deepseek_v41_layer_main_split_oracle import audit as audit_flat_qk
                    result = audit_flat_qk(programs[1].layers[0], cases[case][0], cases[case][1],
                                           cases[case][2], flat_qk=True,
                                           flat_qk_direct=args.candidate=="qk_flat_direct")
                    report.setdefault("qk_flat_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Flat QK exceeded the accepted C1 attention precision envelope")
                    qualified = True
                    report["precision_oracle"] = "Actual C1 codecs/current KV; QK diagonal output audited in FP64"
                if args.candidate == "layer_main_split":
                    from deepseek_v41_layer_main_split_oracle import audit as audit_main_split

                    result = audit_main_split(programs[1].layers[0], cases[case][0], cases[case][1],
                                             cases[case][2])
                    report.setdefault("layer_main_split_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Split main PV exceeded the existing attention precision envelope")
                    qualified = True
                    report["precision_oracle"] = "Real producers/current KV; split publish/reuse audited in FP64"
                if args.candidate == "compact_stream_mme":
                    from deepseek_v41_compact_stream_oracle import audit as audit_compact_stream

                    result = audit_compact_stream(programs[1].layers[0], cases[case][0], cases[case][1],
                                                  cases[case][2])
                    report.setdefault("compact_stream_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Compact stream MME exceeded the existing attention precision envelope")
                    qualified = True
                    report["precision_oracle"] = "Real producers/current KV and independent FP64 attention"
                if args.candidate == "q_bf16_rope":
                    from deepseek_v41_q_bf16_rope_oracle import audit as audit_q_bf16_rope

                    result = audit_q_bf16_rope(programs[1].layers[0], cases[case][0], cases[case][1],
                                              cases[case][2])
                    report.setdefault("q_bf16_rope_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("BF16 Q/RoPE fusion exceeded the existing projection tolerance")
                    qualified = True
                    report["precision_oracle"] = "Three real producers and unchanged BF16 projection/RoPE reference"
                if args.candidate == "fp8_prologue":
                    from deepseek_v41_fp8_prologue_oracle import audit as audit_prologue
                    result = audit_prologue(programs[1].layers[0], cases[case][0], cases[case][1], cases[case][2])
                    report.setdefault("fp8_prologue_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("FP8 fused prologue exceeded shared C1 consumer tolerance")
                    qualified = True
                    report["precision_oracle"] = "Shared C1 FP8 projection/Q/KV operators; BF16 rounding retained"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate in ("c1_dense_chain", "input_fp8"):
                    from deepseek_v41_c1_dense_chain_oracle import audit as audit_dense_chain

                    result = audit_dense_chain(programs[1].layers[0], cases[case][0], cases[case][1],
                                               cases[case][2], rank, case, input_only=args.candidate == "input_fp8")
                    report.setdefault("projection_oracle", []).append(result)
                    if os.getenv("VLLM_HPU_DSV41_DSPARK_INPUT_QUANT_REMAT") == "1":
                        from deepseek_v41_input_quant_remat_oracle import audit as audit_input_quant_remat

                        remat = audit_input_quant_remat(programs[1].layers[0], cases[case][0], cases[case][1])
                        report.setdefault("input_quant_remat_oracle", []).append(remat)
                        if not remat["passed"]:
                            raise AssertionError("Rematerialized producer changed the qualified C1 operand")
                    if not result["passed"]:
                        raise AssertionError("C6 dense chain exceeded accepted C1 operator tolerance")
                    qualified = True
                    report["precision_oracle"] = "C1 dense operators; teacher-forced whole-model gate pending"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate == "query_norm":
                    from deepseek_v41_query_norm_oracle import audit as audit_query_norm

                    result = audit_query_norm(programs[1].layers[0], cases[case][0], cases[case][1])
                    report.setdefault("query_norm_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("C6 query norm exceeded accepted C1 / existing numerical tolerance")
                    qualified = True
                    report["precision_oracle"] = "Three real C6 input producers and accepted C1 query norm"
                if args.candidate == "exp_pv":
                    from deepseek_v41_exp_pv_oracle import audit as audit_exp_pv

                    result = audit_exp_pv(programs[1].layers[0].attention, rank, case)
                    report.setdefault("attention_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("BF16 exponent PV exceeded the existing official attention tolerance")
                    qualified = True
                    report["precision_oracle"] = "Actual hardware BF16-exponent PV, three real queries and current KV"
                if args.candidate == "wo_handoff":
                    from deepseek_v41_wo_handoff_oracle import audit as audit_wo

                    result = audit_wo(programs[1].layers[0].attention, rank, case)
                    report.setdefault("projection_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Batched output handoff exceeded the accepted C1 projection tolerance")
                    qualified = True
                    report["precision_oracle"] = "Same real MLA output and accepted C1 FP8 output projection"
                if args.candidate == "mhc_statistics":
                    from deepseek_v41_mhc_epilogue_oracle import audit as audit_mhc

                    result = audit_mhc(programs[1].layers[0], cases[case][0], statistics=True)
                    report.setdefault("mhc_statistics_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("mHC tiled statistics exceeded the accepted control/gate tolerance")
                    qualified = True
                    report["precision_oracle"] = "Actual BF16 residual/MME projection and accepted C1 RRMS/Sinkhorn"
                if args.candidate in ("scaled_w13", "w13_k_pipeline", "w2_k_pipeline", "w2_ready_scale",
                                      "w2_reduce_n256"):
                    from deepseek_v41_scaled_w13_oracle import audit as audit_scaled_w13

                    result=audit_scaled_w13(programs[1].layers[0],cases[case][0],cases[case][1])
                    report.setdefault("scaled_w13_oracle",[]).append(result)
                    differences=[]
                    for parent, selected in zip((*results[0],*caches[0]),(*results[1],*caches[1]),strict=True):
                        if parent is None or selected is None:
                            continue
                        if parent.dtype.is_floating_point:
                            difference=selected.float()-parent.float()
                            relative=float(difference.norm()/parent.float().norm().clamp_min(1e-30))
                            differences.append(dict(shape=list(parent.shape),relative_l2=relative,
                                                    max_abs=float(difference.abs().max())))
                        else:
                            differences.append(dict(shape=list(parent.shape),dtype=str(parent.dtype),
                                                    differing_elements=int((parent!=selected).sum())))
                    report.setdefault("scaled_w13_internal_differences",[]).append(differences)
                    if not finite or not result["passed"]:
                        raise AssertionError("Scaled W13 exceeds accepted C1 routed-expert tolerance")
                    qualified=True
                    report["precision_oracle"]="Accepted C1 one-row routed expert; teacher mean alpha pending"
                    report["teacher_forced_acceptance_qualified"]=False
                if args.candidate in ("control_fp8", "control_fp8_pair"):
                    from deepseek_v41_control_fp8_oracle import audit as audit_control_fp8

                    result = audit_control_fp8(programs[1].layers[0], cases[case][0])
                    report.setdefault("control_fp8_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("FP8 control exceeds C1 downstream tolerance")
                    qualified = True
                    report["precision_oracle"] = ("Actual controller through C1 gates/post consumer; "
                                                 "FP32 projection error retained")
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate == "mhc_high_plane":
                    from deepseek_v41_mhc_high_plane_oracle import audit as audit_high_plane

                    result = audit_high_plane(programs[1].layers[0], cases[case][0])
                    report.setdefault("mhc_high_plane_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("mHC high-plane controller exceeds production tolerance")
                    qualified = True
                    report["precision_oracle"] = "Actual residual/controller; accepted hi/lo gates and post consumer"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate == "mhc_deferred":
                    from deepseek_v41_mhc_deferred_oracle import audit as audit_deferred

                    result = audit_deferred(programs[1].layers[0], cases[case][0])
                    report.setdefault("mhc_deferred_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Deferred mHC differs from accepted C1 gate/post consumer")
                    qualified = True
                    report["precision_oracle"] = "Actual residual/controller and shared C1 post/collapse"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate in ("output_fp8", "output_layout"):
                    from deepseek_v41_output_fp8_oracle import audit as audit_output

                    result = audit_output(programs[1].layers[0].attention, rank, case)
                    report.setdefault("output_fp8_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("C6 output projection exceeds shared C1 production tolerance")
                    qualified = True
                    report["precision_oracle"] = "Actual MLA producer and common C1 FP8 output consumer"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.candidate == "norm_roundtrip":
                    from deepseek_v41_norm_roundtrip_oracle import audit as audit_norm_roundtrip

                    result = audit_norm_roundtrip(programs[1].layers[0], cases[case][0], cases[case][1], cases[case][2])
                    report.setdefault("norm_roundtrip_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Fused BF16 norm/quant differs from accepted C1 boundaries")
                    qualified = True
                    report["precision_oracle"] = "Actual mHC/QKV producer, exact C1 norm and both BF16 consumers"
                    report["teacher_forced_acceptance_qualified"] = False
                if args.inline_input_norm_oracle:
                    from deepseek_v41_input_norm_oracle import audit as audit_input_norm

                    result = audit_input_norm(programs[1].layers[0], cases[case][0], cases[case][1])
                    report.setdefault("input_norm_oracle", []).append(result)
                    if not result["passed"]:
                        raise AssertionError("Attention input norm or QKV consumer exceeded C1 tolerance")
                    qualified = True
                    report["precision_oracle"] = "Actual BF16 collapse, accepted C1 input norm and QKV consumers"
                report["checks"][-1]["precision_oracle_qualified"] = qualified and finite
                if not all(agreement) and not (qualified and finite):
                    raise AssertionError("C6 native candidate changed group output or KV state")
            if (args.constant_oracle or args.attention_oracle or args.c1_projection_oracle
                    or args.selection_oracle or args.peer_input_oracle):
                report.update(status="diagnostic_completed", micro_qualified=False, performance_measured=False)
                return
            for iteration in range(3):
                medians = []
                for arm, plan in enumerate(plans):
                    device_ms, wall_ms = [], []
                    for sample in range(args.samples):
                        case = sample % len(cases)
                        reset(arm, case)
                        dist.barrier(group=get_tp_group().cpu_group)
                        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        clock = time.perf_counter()
                        begin.record()
                        invoke(arm, case)
                        end.record()
                        end.synchronize()
                        device_ms.append(begin.elapsed_time(end))
                        wall_ms.append((time.perf_counter() - clock) * 1000)
                    medians.append(statistics.median(device_ms))
                    report["rounds"].append(
                        dict(
                            iteration=iteration,
                            arm=arm,
                            device_ms=device_ms,
                            wall_ms=wall_ms,
                            median_device_ms=medians[-1],
                        )
                    )
                report.setdefault("paired_saved_ms", []).append(medians[0] - medians[1])
            saved = statistics.median(report["paired_saved_ms"])
            projection = report["projection_multiplier"] * saved
            report.update(
                status="component_passed",
                saved_ms_per_four_layers=saved,
                projected_ms_per_round=projection,
                micro_qualified=all(v > 0 for v in report["paired_saved_ms"]),
            )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        shutdown_prepared_group_plans()
        if args.candidate == "mhc_native_dependencies":
            if previous_native_dependency is None:
                os.environ.pop(native_dependency_flag, None)
            else:
                os.environ[native_dependency_flag] = previous_native_dependency
        if previous is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = previous


if __name__ == "__main__":
    main()
