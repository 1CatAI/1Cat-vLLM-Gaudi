# SPDX-License-Identifier: Apache-2.0
"""Actual C6 -> official verification -> C5/Markov native control gate.

Only the existing sampled boundary is timed. Discovery uses real request
fixtures and actual MTP cache state, followed by hot native replay. An
uncertified packet is a capability failure here, never an accepted proposal.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--original-parent-head-norm", action="store_true",
                        help="Time untouched generic-norm parent; numerical oracle retains checkpoint BF16 boundary")
    parser.add_argument("--round-input-ab", action="store_true")
    parser.add_argument("--repair-journal", action="store_true")
    parser.add_argument(
        "--native-full",
        action="store_true",
        help="Compare exact full-vocabulary native repair against the existing segmented reference",
    )
    parser.add_argument("--journal-coordinate-ab", action="store_true")
    parser.add_argument("--nucleus-mass-ab", action="store_true")
    parser.add_argument("--global-bounded-ab", action="store_true")
    parser.add_argument("--stock-hccl-bounded-ab", action="store_true")
    parser.add_argument("--local-bounded-ab", action="store_true")
    parser.add_argument("--native-full-main", action="store_true")
    parser.add_argument("--production-stochastic", action="store_true",
                        help="Exercise the production temperature>0 full-repair specialization")
    parser.add_argument("--draft-body-ab", action="store_true")
    parser.add_argument("--stochastic-only-ab", action="store_true")
    parser.add_argument("--framework-head-ab", action="store_true")
    parser.add_argument("--stock-hccl-full-ab", action="store_true",
                        help="Private C6-capacity bridge; exact full HCCL protocol vs production compiled regions")
    parser.add_argument(
        "--bf16-head-ab", action="store_true", help="Common BF16 LM head/FP32 output versus a native FP32 head"
    )
    parser.add_argument(
        "--native-parent",
        action="store_true",
        help="Compare partition selection with the same bounded native protocol using ordinary topk",
    )
    parser.add_argument(
        "--selection-flag",
        choices=("PARTITION_RADIX_SAMPLING", "STREAM_FILTER_SAMPLING"),
        default="PARTITION_RADIX_SAMPLING",
    )
    parser.add_argument("--probability-diagnostic", action="store_true")
    parser.add_argument(
        "--producer-bf16-max-ulp",
        type=int,
        default=0,
        help="Explicit internal BF16 rounding gate; token/RNG/cache gates stay exact",
    )
    parser.add_argument(
        "--fallback-contract-only",
        action="store_true",
        help="Force an uncovered real request, overwrite future write rows, and check exact rollback",
    )
    parser.add_argument("--queued-device-window", action="store_true",
                        help="Queue behind the same real native producer; exclude artificial drained-start host gaps")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--weighted-draft-ab", action="store_true",
                        help="Compare only weighted draft selection; bounded Target and journal stay identical")
    parser.add_argument("--static-weighted-ab", action="store_true")
    parser.add_argument("--sparse-weighted-ab", action="store_true",
                        help="Exact block-bound skip vs existing weighted native probability protocol")
    parser.add_argument("--draft-kv-decode-ab", action="store_true",
                        help="Same native official protocol and sparse consumer; one packed KV decoder")
    parser.add_argument("--draft-packed-mla-ab", action="store_true",
                        help="Same native official protocol; common packed MME attention in C5")
    parser.add_argument("--mtp-k128-ab", action="store_true",
                        help="Same BF16 expert arithmetic; common K128 prepared decoder in C5")
    parser.add_argument("--draft-mhc-ab", action="store_true",
                        help="Same protocol; shared MME/gates/post-collapse for draft rows")
    parser.add_argument("--draft-shared-fp8-ab", action="store_true",
                        help="Same native official protocol; C1 FP8 shared experts in the draft")
    parser.add_argument("--draft-query-fp8-ab", action="store_true",
                        help="Same native official protocol; C1 FP8 Q/RoPE in the draft")
    parser.add_argument("--mtp-fp8-ab", action="store_true", help="Same official protocol; N128 draft FP8 body only")
    parser.add_argument("--vocab-cdf-proof", type=Path)
    parser.add_argument("--vocab-cdf-ab", action="store_true", help="Exact conditional q, alternative proposal CDF")
    parser.add_argument("--record-readback-ab", action="store_true", help="Exact full record; one native D2H")
    parser.add_argument("--protocol-writeback-ab", action="store_true",
                        help="Same sampler; publication inside native tail vs serving eager updates")
    parser.add_argument("--vocab-softmax-ab", action="store_true",
                        help="Full official q; two-node full vocabulary softmax vs production softmax")
    parser.add_argument("--vocab-head-fp8-ab", action="store_true",
                        help="Same actual native protocol; prepared FP8 vocabulary matrix with FP32 output")
    parser.add_argument("--vocab-head-teacher-proof", type=Path)
    parser.add_argument("--journal-batch-direct-ab", action="store_true",
                        help="Direct typed journal outputs without packed slicing")
    parser.add_argument("--journal-batch-ab", action="store_true",
                        help="Exact production protocol with batched journal snapshots")
    parser.add_argument("--batch-staging-ab", action="store_true", help="Same native inputs; one D2D transfer manifest")
    parser.add_argument("--mtp-cache-ab", action="store_true", help="Exact load-time draft BF16 banks")
    parser.add_argument("--mtp-sat-ab", action="store_true", help="Same protocol; shared N256 fused draft body")
    parser.add_argument("--mtp-teacher-proof", type=Path)
    parser.add_argument("--bounded-draft-ab", action="store_true",
                        help="Production weighted-q parent versus certified K64 q; exact repair in both")
    args = parser.parse_args()
    if sum((args.mtp_fp8_ab, args.mtp_sat_ab, args.mtp_cache_ab, args.mtp_k128_ab, args.draft_mhc_ab,
            args.draft_shared_fp8_ab, args.draft_query_fp8_ab,
            args.draft_packed_mla_ab, args.draft_kv_decode_ab)) > 1:
        parser.error("Select one draft precision candidate")
    mtp_precision_ab = (args.mtp_fp8_ab or args.mtp_sat_ab or args.mtp_cache_ab or args.mtp_k128_ab or args.draft_mhc_ab
                        or args.draft_shared_fp8_ab or args.draft_query_fp8_ab
                        or args.draft_packed_mla_ab or args.draft_kv_decode_ab)
    same_protocol_ab = (mtp_precision_ab or args.batch_staging_ab or args.vocab_cdf_ab
                        or args.record_readback_ab or args.journal_batch_ab or args.journal_batch_direct_ab
                        or args.protocol_writeback_ab or args.vocab_softmax_ab or args.vocab_head_fp8_ab)
    head_teacher_qualified = False
    if args.vocab_head_fp8_ab:
        if args.vocab_head_teacher_proof is None:
            parser.error('Vocabulary precision needs its fixed-prefix Target/draft head p/q proof')
        proof = json.loads(args.vocab_head_teacher_proof.read_text())
        if (proof.get('candidate') != 'vocab_head_fp8' or not proof.get('passed')
                or len(proof.get('cases', [])) != 3 or proof.get('mean_alpha_delta', -1.) < -1e-7):
            parser.error('Vocabulary precision fixed-prefix acceptance did not qualify')
        if not proof.get('head_source_sha256'):
            parser.error('Vocabulary precision proof lacks source identity')
        workspace = Path(__file__).resolve().parents[1]
        for name, expected in proof['head_source_sha256'].items():
            path = (workspace / name).resolve()
            if not path.is_relative_to(workspace) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                parser.error('Vocabulary precision implementation differs from its teacher proof')
        native = Path(os.environ['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
        for name, expected in proof['native_binaries'].items():
            if hashlib.sha256((native / name).read_bytes()).hexdigest() != expected:
                parser.error('Vocabulary precision proof uses different native operators')
        head_teacher_qualified = True
    mtp_precision_attribute = ("draft_query_fp8" if args.draft_query_fp8_ab else
                               "draft_shared_fp8" if args.draft_shared_fp8_ab else
                               "draft_mhc" if args.draft_mhc_ab else "mtp_k128" if args.mtp_k128_ab else
                               "draft_kv_decode" if args.draft_kv_decode_ab else
                               "draft_packed_mla" if args.draft_packed_mla_ab else
                               "mtp_cache" if args.mtp_cache_ab else "mtp_sat" if args.mtp_sat_ab else "mtp_fp8")

    def precision_owner(layer):
        if args.draft_mhc_ab:
            return layer
        return layer.attention if (args.draft_packed_mla_ab or args.draft_kv_decode_ab
                                   or args.draft_query_fp8_ab) else layer.moe

    if args.bounded_draft_ab and (mtp_precision_ab or args.weighted_draft_ab
                                 or args.sparse_weighted_ab or args.static_weighted_ab):
        parser.error("Bounded draft selection is an independent sampler change")
    if ((args.draft_packed_mla_ab or args.draft_kv_decode_ab or args.mtp_k128_ab or args.draft_mhc_ab
         or args.draft_shared_fp8_ab or args.draft_query_fp8_ab)
            and args.mtp_teacher_proof is None):
        parser.error("Draft attention change requires its same-prefix conditional acceptance proof")
    if args.mtp_sat_ab and args.mtp_teacher_proof is None:
        parser.error("Draft SAT requires its current native fixed-prefix acceptance proof")
    vocab_cdf_qualified = False
    if args.vocab_cdf_ab:
        if args.vocab_cdf_proof is None:
            parser.error("Proposal draw needs its actual fixed-prefix q/CDF proof")
        cdf_proof = json.loads(args.vocab_cdf_proof.read_text())
        vocab_cdf_qualified = (cdf_proof.get("status") == "passed"
                              and cdf_proof.get("teacher_forced")
                              and cdf_proof.get("probability_construction_unchanged")
                              and cdf_proof.get("mean_alpha_delta", -1) >= -1e-7
                              and len(cdf_proof.get("cases", [])) == 3
                              and all(c.get("teacher_forced_q_exact") for c in cdf_proof["cases"]))
        if not vocab_cdf_qualified:
            parser.error("Conditional proposal probability/draw gate did not pass")
    mtp_teacher_qualified = False
    if args.mtp_teacher_proof is not None:
        proof = json.loads(args.mtp_teacher_proof.read_text())
        if (not mtp_precision_ab or proof.get("candidate") != mtp_precision_attribute or not proof.get("passed")
                or not proof.get("teacher_forced") or not 3 <= len(proof.get("cases", [])) <= 5
                or proof.get("temperature") != 1.0 or proof.get("top_p") != 0.95
                or proof.get("mtp_anchor_alignment") != "C5 positions begin at current Target anchor position"
                or proof.get("mean_alpha_delta", -1.0) < -1e-7):
            parser.error("MTP numerical change requires its three-to-five fixed-prefix probability-overlap proof")
        if args.draft_packed_mla_ab or args.draft_kv_decode_ab:
            workspace = Path(__file__).resolve().parents[1]
            if not proof.get("draft_attention_source_sha256"):
                parser.error("Draft Attention proof lacks source identity")
            for name, expected in proof["draft_attention_source_sha256"].items():
                path = (workspace / name).resolve()
                if not path.is_relative_to(workspace) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    parser.error("Draft Attention implementation differs from teacher proof")
            native = Path(os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
            for name, expected in proof["native_binaries"].items():
                if hashlib.sha256((native / name).read_bytes()).hexdigest() != expected:
                    parser.error("Draft Attention proof uses different native operators")
        if (args.mtp_sat_ab or args.mtp_k128_ab or args.draft_mhc_ab
                or args.draft_shared_fp8_ab or args.draft_query_fp8_ab):
            native = Path(os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
            if not proof.get("native_binaries"):
                parser.error("Draft expert proof lacks native binary identity")
            for name, expected in proof["native_binaries"].items():
                if hashlib.sha256((native / name).read_bytes()).hexdigest() != expected:
                    parser.error("Draft expert teacher proof belongs to a different native installation")
        if args.draft_mhc_ab:
            workspace = Path(__file__).resolve().parents[1]
            if not proof.get('draft_mhc_source_sha256'):
                parser.error('Draft mHC proof lacks source identity')
            for name, expected in proof['draft_mhc_source_sha256'].items():
                path = (workspace / name).resolve()
                if not path.is_relative_to(workspace) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    parser.error('Draft mHC implementation differs from its teacher proof')
        if args.draft_shared_fp8_ab:
            workspace = Path(__file__).resolve().parents[1]
            if not proof.get('draft_shared_source_sha256'):
                parser.error('Draft shared proof lacks source identity')
            for name, expected in proof['draft_shared_source_sha256'].items():
                path = (workspace / name).resolve()
                if (not path.is_relative_to(workspace)
                        or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                    parser.error('Draft shared implementation differs from its teacher proof')
        if args.draft_query_fp8_ab:
            workspace = Path(__file__).resolve().parents[1]
            if not proof.get('draft_query_source_sha256'):
                parser.error('Draft Q proof lacks source identity')
            for name, expected in proof['draft_query_source_sha256'].items():
                path = (workspace / name).resolve()
                if (not path.is_relative_to(workspace)
                        or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                    parser.error('Draft Q implementation differs from its teacher proof')
        mtp_teacher_qualified = True
    if args.stock_hccl_bounded_ab and not args.global_bounded_ab:
        parser.error("Stock-HCCL bounded comparison reuses the global bounded repair contract")
    if args.stock_hccl_full_ab and not (args.native_full and args.native_full_main):
        parser.error("Stock HCCL gate requires the exact full native main protocol")
    if args.stock_hccl_full_ab and any(os.environ.get("VLLM_HPU_DSV41_DSPARK_" + suffix) != "1"
                                      for suffix in ("LARGE_HCCL", "FULL_HCCL_MAIN")):
        parser.error("Stock HCCL gate requires the independently built capacity adapter")
    if args.journal_coordinate_ab and (not args.repair_journal or not args.native_parent or args.native_full):
        parser.error("Journal coordinate gate needs the bounded native parent and real journal")
    if args.native_full_main and (not args.native_full or args.repair_journal
                                  or (args.native_parent and not (args.round_input_ab
                                      or args.bf16_head_ab and args.stock_hccl_full_ab))):
        parser.error("Exact full main compares native and production segmented full protocols without journal")
    if args.round_input_ab and not (args.native_full and args.native_parent and args.native_full_main):
        parser.error("Round input comparison requires exact full native parent and candidate")
    if args.native_full and args.fallback_contract_only:
        parser.error("Use the full repair journal gate before forced bounded fallback testing")
    if (args.bf16_head_ab or args.nucleus_mass_ab) and not args.global_bounded_ab and (
        not args.native_parent or not args.native_full or args.repair_journal
    ):
        parser.error("BF16 head gate requires full native parent/candidate without repair journal")
    if (args.global_bounded_ab or args.local_bounded_ab) and (
        not args.native_parent or not args.repair_journal or args.native_full
    ):
        parser.error("Global bounded gate includes the real journal and compares exact native full parent")
    if (args.native_parent and args.native_full
            and not (args.bf16_head_ab or args.nucleus_mass_ab or args.round_input_ab)):
        parser.error("Native parent comparison is for bounded partition sampling")
    if args.producer_bf16_max_ulp < 0 or (args.producer_bf16_max_ulp and not args.probability_diagnostic):
        parser.error("A producer rounding gate requires the same-logit probability diagnostic")
    rank, tp = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
        "{rank}", str(rank)
    )
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment

    prepare_environment(args.prepared, tensor_parallel_size=tp, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed import get_tp_group
    import torch.distributed as dist
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import PreparedDraft, PreparedInput, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs
    from vllm_gaudi.ops.deepseek_v41_round_repair import SampledRoundRepairFrame
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import speculative_sampling_draws
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import full_sampling_distribution_sharded
    from vllm_gaudi.ops.deepseek_v41_verify import DRAFT_START, OUTPUT_START, STATUS
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(
        rank=rank,
        status="loading",
        capability_passed=False,
        micro_qualified=False,
        fixtures=[],
        checks=[],
        rounds=[],
        qualified_gain_ms=0,
    )
    report["producer_bf16_max_ulp"] = args.producer_bf16_max_ulp
    report["native_full_repair"] = args.native_full and not args.native_full_main
    report["native_full_main"] = args.native_full_main

    def save():
        (root / f"sampled-control-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    if (args.weighted_draft_ab or same_protocol_ab or args.bounded_draft_ab
            or args.sparse_weighted_ab or args.static_weighted_ab):
        # Record only an unsupported view's origin on compile failure. This
        # neither changes replay eligibility nor exports a graph/profiler run.
        from vllm_gaudi.ops import tp2_prepared_plan

        original_eligible = tp2_prepared_plan._eligible

        def audit_eligible(graph, *arguments, **keywords):
            try:
                return original_eligible(graph, *arguments, **keywords)
            except RuntimeError:
                nodes = list(graph.graph.nodes)
                views = []
                for index, node in enumerate(nodes):
                    if str(node.target) != "aten.transpose.int":
                        continue
                    value = node.meta.get("val")
                    views.append(dict(name=node.name, arguments=str(node.args),
                                      shape=None if value is None else list(value.shape),
                                      stack_trace=node.meta.get("stack_trace"),
                                      source_fn_stack=str(node.meta.get("source_fn_stack")),
                                      neighbors=[str(n.target) for n in nodes[max(0, index - 2):index + 3]]))
                (root / f"unsupported-view-origin-rank{rank}.json").write_text(json.dumps(views, indent=2) + "\n")
                raise

        tp2_prepared_plan._eligible = audit_eligible
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))
    plan = parent_plan = None
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            bind_worker_cpu(rank)
            torch.hpu.set_device(rank)
            load_native_operators()
            init_distributed_environment(
                world_size=tp, rank=rank, local_rank=rank, distributed_init_method="env://", backend="hccl"
            )
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            if os.environ.get("GRAPH_VISUALIZATION") == "1":
                from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

                configure_post_graph_directory(root / "graphs" / f"rank{rank}")
            reduce, gather = stage_collectives(rank, True, tp)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            specs = {
                name: spec
                for name, spec in shard.specs.items()
                if name.startswith("mtp.") or name in ("head.weight", "embed.weight", "norm.weight")
            }
            weights = _weight_tree(specs)
            load_weight_tree(shard, weights, "hpu", specs)
            production_bf16_head = (args.production_stochastic
                                    and os.environ.get('VLLM_HPU_DSV41_BF16_LM_HEAD') == '1')
            if (production_bf16_head or args.weighted_draft_ab or same_protocol_ab or args.bounded_draft_ab
                    or args.sparse_weighted_ab or args.static_weighted_ab):
                # Both sampler arms retain the accepted serving298 head.
                # A false fixture flag dispatches F.linear instead and puts
                # its weight transpose outside an eligible native recipe.
                weights.head.weight = weights.head.weight.bfloat16()
            stage = SimpleNamespace(
                weights=weights,
                config={"text_config": text},
                shard=shard,
                tp_rank=rank,
                tensor_parallel_size=tp,
                reduce=reduce,
                all_gather=gather,
                shared=PagedCSA2SharedState(text, 0, 40, "hpu", 1048576, tensor_parallel_size=tp),
                bf16_head=(production_bf16_head or args.weighted_draft_ab or same_protocol_ab or args.bounded_draft_ab
                           or args.sparse_weighted_ab or args.static_weighted_ab),
            )
            draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
            if args.probability_diagnostic:
                draft.register_buffer(
                    "sampling_normalized_debug",
                    torch.empty((5, text["hidden_size"]), dtype=torch.bfloat16, device="hpu"),
                )
                for name in ("sampling_base_logits_debug", "sampling_logits_debug"):
                    draft.register_buffer(
                        name, torch.empty((5, draft.output_head.weight.shape[0]), dtype=torch.float32, device="hpu")
                    )

            reference_draft = draft
            if args.bf16_head_ab or args.framework_head_ab:
                import copy

                copies = {}

                def clone_metadata(module):
                    if id(module) in copies:
                        return copies[id(module)]
                    copied = copy.copy(module)
                    copies[id(module)] = copied
                    copied._buffers = module._buffers.copy()
                    copied._parameters = module._parameters.copy()
                    copied._modules = {
                        name: clone_metadata(child) if child is not None else None
                        for name, child in module._modules.items()
                    }
                    return copied

                # Shared real C5 matrices/state, private vocab-head metadata.
                # BF16 checkpoint weights are exact; both paths output FP32.
                draft = clone_metadata(reference_draft)
                draft.output_head.weight = reference_draft.output_head.weight.bfloat16()
                draft.bf16_head = True
                report["candidate_dispatch"] = "common BF16_LM_HEAD with FP32 accumulator/output"

            def full_from_biased(logits, controls):
                return torch.cat(
                    tuple(
                        full_sampling_distribution_sharded(
                            logits[i : i + 1], controls[i : i + 1], tp_rank=rank, all_gather=gather, force_legacy=True
                        )[1]
                        for i in range(5)
                    )
                )

            full_probability = torch.compile(full_from_biased, backend="hpu_backend", fullgraph=True, dynamic=False)
            embedding = PreparedInput(weights.embed, rank, reduce)

            def next_input(record):
                anchor = record[OUTPUT_START:DRAFT_START].gather(0, record[2:3].clamp(1, 6) - 1)
                return embedding(torch.cat((anchor, record[DRAFT_START:STATUS])).long())

            consume = torch.compile(next_input, backend="hpu_backend", fullgraph=True, dynamic=False)
            from functools import partial

            reference_known = (args.production_stochastic or args.framework_head_ab or args.stock_hccl_full_ab
                               or args.stock_hccl_bounded_ab)
            verify = torch.compile(
                partial(reference_draft.verify_sampled_prefix_full, known_stochastic=reference_known),
                backend="hpu_backend", fullgraph=True, dynamic=False
            )
            propose = torch.compile(
                partial(reference_draft.draft_sampled_from_prefix, known_stochastic=reference_known),
                backend="hpu_backend", fullgraph=True, dynamic=False
            )
            draws = torch.compile(speculative_sampling_draws, backend="hpu_backend", fullgraph=True, dynamic=False)

            def reference(inputs):
                hidden, ids, control, aux, positions, q, params, seed, counter, offsets = inputs
                dc, acceptance, correction, tc = draws(params, seed, counter, offsets)
                prefix = verify(hidden, ids, q, control, aux, positions, tc, acceptance, correction)
                record, wire, confidence, probability, covered = propose(control, positions, *prefix[:6], dc)
                return record, wire, probability, confidence, counter.clone(), covered

            files = sorted((args.fixtures / f"rank{rank}").glob("c6-*.pt"))
            if not 3 <= len(files) <= 5 or args.samples < 3:
                raise ValueError("Need 3-5 actual request cases and >=3 samples")
            cases, raw = [], []
            for path in files:
                data = torch.load(path, map_location="cpu", weights_only=True)
                if not data.get("request_context_qualified") or data["rank"] != rank:
                    raise ValueError("Only actual rank-matched 16K request fixtures qualify")
                values = [
                    data[k]
                    for k in (
                        "hidden",
                        "ids",
                        "control",
                        "auxiliary",
                        "positions",
                        "proposal",
                        "sampling_parameters",
                        "sampling_seed",
                        "sampling_counter",
                        "sampling_offsets",
                    )
                ]
                values[1], values[2] = values[1][1:].contiguous(), values[2][:7].contiguous()
                inputs = tuple(v.to("hpu") for v in values)
                caches = tuple(v.to("hpu") for v in data["draft_swa"])
                cases.append((inputs, caches))
                raw.append(data)
                report["fixtures"].append(dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

            bank = frame = None
            target_cases = []
            if args.repair_journal:
                if any(
                    not data.get("history_context_qualified") or not data.get("decoder_state_files") for data in raw
                ):
                    raise ValueError("Native serving journal requires actual decoder state and request lookback")
                bank = torch.nn.Module()
                bank.length, bank.search_length = 1048576, raw[0]["search_length"]
                bank.layers = torch.nn.ModuleList()
                bank.shared = torch.nn.Module()
                bank.shared.sources, bank.shared.topk = torch.nn.ModuleDict(), torch.nn.ModuleDict()
                bank.shared.decoded_kv_state = False
                bank.shared.index_mirror_tokens, bank.shared.index_mirror_valid = 1048576, True
                for layer_index in range(text["num_hidden_layers"]):
                    layer = torch.nn.Module()
                    layer.layer = layer_index
                    layer.attention = torch.nn.Module()
                    bank.layers.append(layer)
                for name in raw[0]["decoder_state_files"]:
                    parts = name.split(".")
                    owner = bank
                    for component in parts[:-1]:
                        if not hasattr(owner, component):
                            owner.add_module(component, torch.nn.Module())
                        owner = getattr(owner, component)
                    path = Path(raw[0]["decoder_state_root"]) / raw[0]["decoder_state_files"][name]
                    owner.register_buffer(parts[-1], torch.load(path, weights_only=True).to("hpu"))
                for source, cache in bank.shared.sources.items():
                    cache.ratio = text["compress_ratios"][int(source)]
                if not hasattr(bank.shared, "candidate_pool"):
                    bank.shared.candidate_pool = None
                bank.draft = draft
                for data in raw:
                    values = {}
                    for name, relative in data["decoder_state_files"].items():
                        values[name] = torch.load(Path(data["decoder_state_root"]) / relative, weights_only=True)
                    target_cases.append(values)
                cursor = DeviceRoundInputs(
                    torch.empty_like(raw[0]["ids"], device="hpu"), torch.empty_like(raw[0]["positions"], device="hpu")
                )
                engram = SimpleNamespace(histories=torch.empty((7, 3), dtype=torch.int32, device="hpu"))
                frame = SampledRoundRepairFrame(bank, cursor, engram, cases[0][0], 0)
                report["journal_saved_bytes"] = frame.journal.bytes
                report["journal_target_states"] = frame.journal.names
            if args.draft_body_ab or args.stochastic_only_ab or args.framework_head_ab:
                from deepseek_v41_draft_body_gate import qualify

                qualify(args, draft, cases, reference, verify, draws, consume, full_probability, report, save)
                return
            sampling_publication = None
            if args.protocol_writeback_ab:
                from vllm_gaudi.ops.deepseek_v41_sampling_publication import SamplingStatePublication

                publication_state = SimpleNamespace(proposal=torch.zeros_like(cases[0][0][5]),
                                                    counter=torch.zeros_like(cases[0][0][8]),
                                                    proposal_valid=torch.zeros(1, dtype=torch.bool, device="hpu"))
                sampling_publication = SamplingStatePublication(publication_state)

            def publish_sampling(values):
                record, _, q, _, counter = values
                publication_state.proposal.copy_(q)
                publication_state.counter.copy_(counter)
                publication_state.proposal_valid.copy_(record[STATUS:STATUS + 1] == 0)

            plan = NativeDraftProtocol(
                draft,
                generation=1,
                sampled=True,
                repair_frame=frame,
                full=args.native_full or args.global_bounded_ab,
                full_main=args.nucleus_mass_ab or args.native_full_main or args.global_bounded_ab,
                known_stochastic=reference_known,
                state_publication=sampling_publication,
            )
            if args.stock_hccl_full_ab:
                report["timed_parent"] = "production full HCCL, prefix CDF specialization in both arms"
                report["candidate_dispatch"] = "DSpark-private large stock HCCL, exact full native protocol"

            round_state = None
            if args.round_input_ab:
                from vllm_gaudi.ops.deepseek_v41_round_inputs import advance_round
                from vllm_gaudi.ops.deepseek_v41_sampled_round_replay import NativeSampledRoundProtocol

                if any(not data.get("history_context_qualified") for data in raw):
                    raise ValueError("Round input gate requires actual compressed-prefix histories")
                plan = NativeSampledRoundProtocol(draft, embedding, generation=1,
                                                  histories=raw[0]["engram_histories"].to("hpu"))
                round_state = plan

                def publish_and_consume(values, control):
                    record, _, q, _, counter = values
                    round_state.round_proposal.copy_(q)
                    round_state.round_counter.copy_(counter)
                    round_state.round_valid.copy_((record[STATUS:STATUS + 1] == 0).to(torch.int32))
                    advance_round(record, control, round_state.round_histories, round_state.round_ids,
                                  round_state.round_positions, round_state.round_control, round_state.round_history)
                    return embedding(round_state.round_ids.long())

                round_parent_consumer = torch.compile(publish_and_consume, backend="hpu_backend",
                                                       fullgraph=True, dynamic=False)

            # Store the request counter once; restoration stays device-side.
            counters = tuple(case[0][8].clone() for case in cases)
            cursor_cases = (
                tuple(
                    tuple(
                        data[name].to("hpu")
                        for name in ("ids", "positions", "control", "cursor_history", "engram_histories")
                    )
                    for data in raw
                )
                if bank is not None
                else ()
            )

            def reset(index, *, target_state=True):
                inputs, caches = cases[index]
                for layer, cache in zip(draft.layers, caches, strict=True):
                    layer.attention.swa.copy_(cache)
                inputs[8].copy_(counters[index])
                if sampling_publication is not None:
                    publication_state.proposal.copy_(inputs[5])
                    publication_state.counter.copy_(counters[index])
                    publication_state.proposal_valid.zero_()
                if round_state is not None:
                    round_state.round_histories.copy_(raw[index]["engram_histories"].to("hpu"))
                if bank is not None:
                    if target_state:
                        for name, value in target_cases[index].items():
                            module, _, leaf = name.rpartition(".")
                            getattr(bank.get_submodule(module), leaf).copy_(value)
                    for destination, source in zip(
                        (cursor.ids, cursor.positions, cursor.control, cursor.history, engram.histories),
                        cursor_cases[index],
                        strict=True,
                    ):
                        destination.copy_(source)
                    if args.native_full:
                        frame.capture_inputs(*inputs)
                        frame.journal(cursor.positions)

            reset(0)
            plan.prepare(*cases[0][0])
            plan.require_ready()
            if args.native_parent:
                selection_flag = (
                    "VLLM_HPU_DSV41_DSPARK_VOCAB_HEAD_FP8" if args.vocab_head_fp8_ab else
                    "VLLM_HPU_DSV41_DSPARK_VOCAB_SOFTMAX" if args.vocab_softmax_ab else
                    "VLLM_HPU_DSV41_DSPARK_PROTOCOL_WRITEBACK" if args.protocol_writeback_ab else
                    "VLLM_HPU_DSV41_DSPARK_JOURNAL_BATCH_DIRECT" if args.journal_batch_direct_ab else
                    "VLLM_HPU_DSV41_DSPARK_JOURNAL_BATCH" if args.journal_batch_ab else
                    "VLLM_HPU_DSV41_DSPARK_RECORD_READBACK" if args.record_readback_ab else
                    "VLLM_HPU_DSV41_DSPARK_DRAFT_VOCAB_CDF" if args.vocab_cdf_ab else
                    "VLLM_HPU_DSV41_DSPARK_BATCH_INPUT_STAGING"
                    if args.batch_staging_ab else
                    "VLLM_HPU_DSV41_DSPARK_WEIGHTED_STATIC_MASK"
                    if args.static_weighted_ab else
                    "VLLM_HPU_DSV41_DSPARK_WEIGHTED_SPARSE_BINS"
                    if args.sparse_weighted_ab else
                    ("VLLM_HPU_DSV41_DSPARK_DRAFT_QUERY_FP8" if args.draft_query_fp8_ab else
                     "VLLM_HPU_DSV41_DSPARK_DRAFT_SHARED_FP8" if args.draft_shared_fp8_ab else
                     "VLLM_HPU_DSV41_DSPARK_DRAFT_MHC" if args.draft_mhc_ab else
                     "VLLM_HPU_DSV41_DSPARK_MTP_K128" if args.mtp_k128_ab else
                     "VLLM_HPU_DSV41_DSPARK_DRAFT_KV_DECODE" if args.draft_kv_decode_ab else
                     "VLLM_HPU_DSV41_DSPARK_DRAFT_PACKED_MLA" if args.draft_packed_mla_ab else
                     "VLLM_HPU_DSV41_DSPARK_MTP_CACHE" if args.mtp_cache_ab else
                     "VLLM_HPU_DSV41_DSPARK_MTP_SAT" if args.mtp_sat_ab else "VLLM_HPU_DSV41_DSPARK_MTP_FP8")
                    if mtp_precision_ab else
                    "VLLM_HPU_DSV41_DSPARK_WEIGHTED_DRAFT_NUCLEUS"
                    if args.weighted_draft_ab or args.bounded_draft_ab else
                    "VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING"
                    if args.global_bounded_ab
                    else "VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"
                    if args.nucleus_mass_ab
                    else "VLLM_HPU_DSV41_DSPARK_" + args.selection_flag
                )
                if (
                    not (args.bf16_head_ab or args.journal_coordinate_ab
                         or args.local_bounded_ab or args.round_input_ab)
                    and os.environ.get(selection_flag) != ("0" if args.bounded_draft_ab else "1")
                ):
                    raise ValueError("Native parent gate requires the candidate selection flag")
                if args.global_bounded_ab or args.nucleus_mass_ab or not (
                    args.bf16_head_ab or args.journal_coordinate_ab or args.local_bounded_ab or args.round_input_ab
                ):
                    os.environ[selection_flag] = "1" if args.bounded_draft_ab else "0"
                parent_head_boundary = os.environ.get("VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY", "0")
                parent_exact_draft = os.environ.get("VLLM_HPU_DSV41_DSPARK_EXACT_DRAFT_SAMPLING", "0")
                if args.weighted_draft_ab:
                    os.environ["VLLM_HPU_DSV41_DSPARK_EXACT_DRAFT_SAMPLING"] = "1"
                if args.original_parent_head_norm:
                    os.environ["VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY"] = "0"
                try:
                    if mtp_precision_ab:
                        for layer in draft.layers:
                            setattr(precision_owner(layer), mtp_precision_attribute, False)
                    if args.journal_batch_direct_ab:
                        frame.journal.batch_direct = False
                    if args.journal_batch_ab:
                        frame.journal.batch_copy = False
                    if args.journal_coordinate_ab:
                        frame.journal.cache_coordinates = False
                    reset(0)
                    parent_plan = NativeDraftProtocol(
                        reference_draft if args.bf16_head_ab else draft,
                        generation=1,
                        sampled=True,
                        repair_frame=(frame if (args.weighted_draft_ab or same_protocol_ab or args.bounded_draft_ab
                                                or args.sparse_weighted_ab or args.static_weighted_ab) else
                                      None if (args.global_bounded_ab or args.local_bounded_ab) else frame),
                        full=(plan.full if same_protocol_ab or args.bounded_draft_ab else
                              args.native_full or args.global_bounded_ab or args.local_bounded_ab),
                        full_main=(
                            plan.full_main if same_protocol_ab or args.bounded_draft_ab else args.nucleus_mass_ab
                            or args.native_full_main
                            or args.global_bounded_ab
                            or args.local_bounded_ab
                        ),
                        known_stochastic=reference_known,
                    )
                    parent_plan.prepare(*cases[0][0])
                    parent_plan.require_ready()
                finally:
                    if mtp_precision_ab:
                        for layer in draft.layers:
                            setattr(precision_owner(layer), mtp_precision_attribute, True)
                    if args.weighted_draft_ab:
                        os.environ["VLLM_HPU_DSV41_DSPARK_EXACT_DRAFT_SAMPLING"] = parent_exact_draft
                    if args.original_parent_head_norm:
                        os.environ["VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY"] = parent_head_boundary
                        report["timed_parent_head_norm"] = "untouched generic RMSNorm, original220 dtype boundary"
                        report["numerical_reference_head_norm"] = "explicit native BF16 checkpoint value boundary"
                    if args.journal_batch_direct_ab:
                        frame.journal.batch_direct = True
                    if args.journal_batch_ab:
                        frame.journal.batch_copy = True
                    if args.journal_coordinate_ab:
                        frame.journal.cache_coordinates = True
                    if args.global_bounded_ab or args.nucleus_mass_ab or not (
                        args.bf16_head_ab or args.journal_coordinate_ab or args.local_bounded_ab or args.round_input_ab
                    ):
                        os.environ[selection_flag] = "0" if args.bounded_draft_ab else "1"
                report["timed_parent"] = (
                    "same full-q weighted nucleus, native protocol and next embedding; BF16 vocabulary head"
                    if args.vocab_head_fp8_ab else
                    "same full vocabulary q, weighted nucleus, native protocol and next embedding; production softmax"
                    if args.vocab_softmax_ab else
                    "same weighted nucleus/BF16 head/Target/journal/native continuation, without block skipping"
                    if (args.sparse_weighted_ab or args.static_weighted_ab) else
                    "same BF16 head, bounded Target, journal and native protocol; original BF16 N128 draft experts"
                    if mtp_precision_ab else
                    "same BF16 head, bounded Target, journal and native protocol; weighted radix draft nucleus"
                    if args.bounded_draft_ab else
                    "same BF16 head, bounded Target, journal and native protocol; exact full draft sort"
                    if args.weighted_draft_ab else
                    "same exact full native protocol, state publication/advance/next embedding compiled separately"
                    if args.round_input_ab else
                    "exact full native sampler without journal vs local K256 packets plus production journal"
                    if args.local_bounded_ab
                    else "exact full native sampler without journal vs global K256 plus production journal"
                    if args.global_bounded_ab
                    else "same native bounded protocol; uncached journal coordinates"
                    if args.journal_coordinate_ab
                    else "same native full official protocol; full vocabulary sorting"
                    if args.nucleus_mass_ab
                    else "same native full official protocol; FP32 head"
                    if args.bf16_head_ab
                    else "same native bounded protocol and journal; ordinary topk selection"
                )
            fallback_cases = set()
            if args.correctness_only and frame is not None and getattr(frame.journal, "native_copy", False):
                # Reproduce the serving cold sequence, not only a warm
                # bounded invocation followed by a forced repair call.
                reset(0)
                plan.prepare(*cases[0][0])
                certificate = getattr(frame, "coverage_flags", None)
                retained_certificate = certificate.cpu().clone() if certificate is not None else None
                cold_repair = NativeDraftProtocol(
                    draft, generation=1, sampled=True, full=True, repair_frame=frame,
                    known_stochastic=reference_known)
                cold_repair.prepare_repair(*cases[0][0], journal_positions=cursor.positions)
                cold_repair.require_ready()
                if certificate is not None:
                    ownership = dict(
                        bounded_plan_binds_certificate=any(v is certificate for v in plan.states),
                        exact_repair_excludes_unused_certificate=all(v is not certificate for v in cold_repair.states),
                        exact_repair_preserves_certificate=torch.equal(certificate.cpu(), retained_certificate),
                    )
                    report["coverage_certificate_state_ownership"] = ownership
                    if not all(ownership.values()):
                        raise AssertionError("Bounded coverage certificate ownership differs from serving")
                cold_repair.close()
                report["serving_cold_repair_sequence_passed"] = True
            for index, (inputs, _) in enumerate(cases):
                reset(index)
                if args.nucleus_mass_ab:
                    os.environ["VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"] = "0"
                try:
                    expected = tuple(v.cpu() for v in reference(inputs))
                finally:
                    if args.nucleus_mass_ab:
                        os.environ["VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"] = "1"
                if args.probability_diagnostic:
                    reference_debug = tuple(
                        getattr(draft, name).cpu().clone()
                        for name in ("sampling_normalized_debug", "sampling_base_logits_debug", "sampling_logits_debug")
                    )
                expected_input = tuple(v.cpu() for v in consume(expected[0].to("hpu")))
                reference_caches = tuple(layer.attention.swa.cpu().clone() for layer in draft.layers)
                reset(index)
                if args.native_full and frame is not None:
                    saved_target = tuple(
                        getattr(frame.journal, f"target_{i}").cpu().clone() for i in range(len(frame.journal.specs))
                    )
                    for i in range(len(frame.journal.specs)):
                        getattr(frame.journal, f"target_{i}").index_fill_(0, getattr(frame.journal, f"indices_{i}"), 7)
                    for layer in draft.layers:
                        layer.attention.swa.zero_()
                    cursor.ids.zero_()
                    cursor.positions.zero_()
                    cursor.history.zero_()
                    engram.histories.zero_()
                actual = tuple(v.cpu() for v in plan(*inputs))
                if frame is not None and getattr(frame.journal, "native_copy", False):
                    # Audit against immutable actual-request state, before
                    # any repair can make a bad saved row self-consistent.
                    logical = inputs[4].cpu()[0].long() + actual[0][1].long() + torch.arange(6)
                    pages_cpu = frame.journal.pages.cpu().long()
                    checks = []
                    for j, ((mode, parameter), name) in enumerate(zip(
                        frame.journal.specs, frame.journal.names, strict=True
                    )):
                        source = target_cases[index][name]
                        if mode == "ring":
                            rows = logical.remainder(parameter)
                        elif mode == "paged":
                            page = pages_cpu.index_select(0, logical.div(128, rounding_mode="floor"))
                            rows = page * (128 // parameter) + logical.remainder(128).div(
                                parameter, rounding_mode="floor")
                        elif mode == "logical":
                            rows = logical.div(parameter, rounding_mode="floor")
                        else:
                            rows = torch.arange(6)
                        rows = torch.where((rows >= 0) & (rows < source.shape[0]), rows, 0)
                        actual_rows = getattr(frame.journal, f"indices_{j}").cpu().long()
                        saved = getattr(frame.journal, f"saved_{j}").cpu().contiguous().view(torch.uint8)
                        expected_rows = source.index_select(0, rows).contiguous().view(torch.uint8)
                        checks.append(torch.equal(actual_rows, rows) and torch.equal(saved, expected_rows))
                    report.setdefault("native_journal_checks", []).append(
                        dict(case=index, allocations=len(checks), coordinates_and_saved_bytes_exact=all(checks)))
                    if not all(checks):
                        bad = [name for name, ok in zip(frame.journal.names, checks, strict=True) if not ok]
                        raise AssertionError(f"Native rollback coordinate/byte-copy mismatch: {bad}")
                if args.native_full and frame is not None:
                    target_restored = all(
                        torch.equal(getattr(frame.journal, f"target_{i}").cpu(), saved)
                        for i, saved in enumerate(saved_target)
                    )
                    if not target_restored:
                        raise AssertionError("Native full repair failed to restore speculative Target writes")
                    report.setdefault("native_repair_restoration", []).append(
                        dict(
                            case=index,
                            all_target_allocations_exact=target_restored,
                            cursor_and_history_exact=all(
                                torch.equal(value.cpu(), expected.cpu())
                                for value, expected in zip(
                                    (cursor.ids, cursor.positions, cursor.control, cursor.history, engram.histories),
                                    cursor_cases[index],
                                    strict=True,
                                )
                            ),
                        )
                    )
                if args.probability_diagnostic:
                    native_debug = tuple(
                        getattr(draft, name).cpu().clone()
                        for name in ("sampling_normalized_debug", "sampling_base_logits_debug", "sampling_logits_debug")
                    )
                    dc, _, _, _ = speculative_sampling_draws(inputs[6], inputs[7], counters[index].clone(), inputs[9])
                    recomputed = full_probability(native_debug[2].to("hpu"), dc).cpu()
                    diagnostic = dict(
                        reference=reference_debug,
                        native=native_debug,
                        reference_q=expected[2],
                        native_q=actual[2],
                        native_full_q=recomputed,
                        controls=dc.cpu(),
                    )
                    torch.save(diagnostic, root / f"producer-case{index}-rank{rank}.pt")
                    report.setdefault("producer_diagnostics", []).append(
                        dict(
                            case=index,
                            normalized_max_abs=float((native_debug[0] - reference_debug[0]).abs().max()),
                            base_logits_max_abs=float((native_debug[1] - reference_debug[1]).abs().max()),
                            biased_logits_max_abs=float((native_debug[2] - reference_debug[2]).abs().max()),
                            native_bounded_full_q_max_abs=float((recomputed - actual[2]).abs().max()),
                        )
                    )
                    # Distribution correctness is tested against the logits
                    # that actually produced q. Comparing q alone across two
                    # different C5 numerical paths conflates producer rounding
                    # with a speculative rejection probability bug.
                    bf16_a, bf16_b = native_debug[0].view(torch.int16), reference_debug[0].view(torch.int16)
                    producer_ulp = int((bf16_a.int() - bf16_b.int()).abs().max())
                    producer_finite = bool(
                        torch.isfinite(native_debug[0]).all() and torch.isfinite(reference_debug[0]).all()
                    )
                    report["producer_diagnostics"][-1]["normalized_max_ulp"] = producer_ulp
                if int(actual[0][STATUS]) != 0:
                    if frame is None:
                        raise AssertionError("Actual p/q nucleus uncovered: exact serving repair remains required")
                    fallback_cases.add(index)
                    frame.journal.restore()
                    frame.restore_protocol()
                    actual = tuple(v.cpu() for v in reference(frame.payload))[:5]
                if any(not torch.equal(a, b) for a, b in zip(expected[:2], actual[:2], strict=True)):
                    raise AssertionError("Native verification or Markov token record changed")
                q_close = bool(torch.isclose(actual[2], expected[2], rtol=2e-5, atol=2e-7).all())
                producer_rounding = False
                if not q_close and args.producer_bf16_max_ulp and index not in fallback_cases:
                    producer_rounding = (
                        producer_finite
                        and producer_ulp <= args.producer_bf16_max_ulp
                        and bool(torch.isclose(actual[2], recomputed, rtol=2e-5, atol=2e-7).all())
                    )
                agreement = [None] * tp
                dist.all_gather_object(agreement, q_close or producer_rounding, group=get_tp_group().cpu_group)
                if not all(agreement):
                    # Distinguish a native binding/dependency bug from the
                    # whole-region bounded computation before changing math
                    # or accepting a looser probability tolerance.
                    reset(index)
                    ordinary = torch.compile(
                        draft.verify_and_propose_sampled_full
                        if args.native_full
                        else draft.verify_and_propose_sampled_bound,
                        backend="hpu_backend",
                        fullgraph=True,
                        dynamic=False,
                    )
                    ordinary_values = tuple(v.cpu() for v in ordinary(*inputs))
                    ordinary_cache = tuple(layer.attention.swa.cpu().clone() for layer in draft.layers)
                    torch.save(
                        dict(
                            reference=expected,
                            native=actual,
                            ordinary=ordinary_values,
                            reference_cache=reference_caches,
                            ordinary_cache=ordinary_cache,
                            fixed_inputs=tuple(v.cpu() for v in plan.fixed),
                            fixture=str(files[index]),
                        ),
                        root / f"numerical-case{index}-rank{rank}.pt",
                    )
                    report["checks"].append(
                        dict(
                            case=index,
                            native_reference_q_max_abs=float((actual[2] - expected[2]).abs().max()),
                            ordinary_reference_q_max_abs=float((ordinary_values[2] - expected[2]).abs().max()),
                            ordinary_native_q_max_abs=float((ordinary_values[2] - actual[2]).abs().max()),
                        )
                    )
                    dist.barrier(group=get_tp_group().cpu_group)
                    raise AssertionError("Official q mismatch; whole-region/native diagnostics preserved on all ranks")
                confidence_error = float((actual[3] - expected[3]).abs().max())
                if not args.producer_bf16_max_ulp:
                    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
                if not torch.equal(actual[4], expected[4]):
                    raise AssertionError("Native replay did not advance the request RNG exactly once")
                actual_input = tuple(v.cpu() for v in consume(actual[0].to("hpu")))
                if any(not torch.equal(a, b) for a, b in zip(expected_input, actual_input, strict=True)):
                    raise AssertionError("The next actual Target input producer changed")
                if args.round_input_ab:
                    if not all(torch.equal(a, b) for a, b in zip(actual[5:7], actual_input, strict=True)):
                        raise AssertionError("In-plan next embedding differs from the actual consumer")
                    internal = tuple(value.cpu().clone() for value in (
                        plan.round_ids, plan.round_positions, plan.round_control, plan.round_history,
                        plan.round_proposal, plan.round_counter, plan.round_valid))
                    round_parent_consumer(tuple(value.to("hpu") for value in actual[:5]), inputs[2])
                    reference_state = tuple(value.cpu() for value in (
                        plan.round_ids, plan.round_positions, plan.round_control, plan.round_history,
                        plan.round_proposal, plan.round_counter, plan.round_valid))
                    if not all(torch.equal(a, b) for a, b in zip(internal, reference_state, strict=True)):
                        raise AssertionError("Native publication/positions/accepted history changed")
                    report.setdefault("round_input_checks", []).append(
                        dict(case=index, states_and_embedding_exact=True))
                for layer, cache in zip(draft.layers, reference_caches, strict=True):
                    torch.testing.assert_close(layer.attention.swa.cpu(), cache, rtol=0, atol=0)
                if parent_plan is not None:
                    parent_reference = actual
                    if args.original_parent_head_norm:
                        # The untouched timer arm retains its old numerical
                        # boundary. Validate its replay against that same
                        # original reference; the candidate has already passed
                        # the explicit checkpoint-boundary oracle above.
                        reset(index)
                        boundary = os.environ.get("VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY", "0")
                        os.environ["VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY"] = "0"
                        try:
                            parent_reference = tuple(value.cpu() for value in reference(inputs))
                        finally:
                            os.environ["VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY"] = boundary
                    if mtp_precision_ab:
                        # Each arithmetic arm must match its own producer.
                        # The parent is BF16; comparing it to the FP8 oracle
                        # wrongly rejects a legal producer-distribution change.
                        reset(index)
                        for layer in draft.layers:
                            setattr(precision_owner(layer), mtp_precision_attribute, False)
                        try:
                            parent_reference = tuple(value.cpu() for value in reference(inputs))
                        finally:
                            for layer in draft.layers:
                                setattr(precision_owner(layer), mtp_precision_attribute, True)
                    if args.vocab_cdf_ab or args.vocab_head_fp8_ab:
                        reset(index)
                        os.environ[selection_flag] = "0"
                        try:
                            parent_reference = tuple(value.cpu() for value in reference(inputs))
                        finally:
                            os.environ[selection_flag] = "1"
                    reset(index)
                    native_parent = tuple(value.cpu() for value in parent_plan(*inputs))
                    if ((same_protocol_ab or args.bounded_draft_ab
                         or args.sparse_weighted_ab or args.static_weighted_ab)
                            and int(native_parent[0][STATUS]) != 0):
                        if frame is None:
                            raise AssertionError("Uncovered weighted parent requires its exact repair journal")
                        report.setdefault("parent_exact_repairs", []).append(
                            dict(case=index, raw_status=int(native_parent[0][STATUS])))
                        # Compare the transaction actually returned by serving:
                        # an uncovered parent record is repaired with the same
                        # saved draw. Tokens/probabilities/RNG remain strict.
                        frame.journal.restore()
                        frame.restore_protocol()
                        if mtp_precision_ab:
                            for layer in draft.layers:
                                setattr(precision_owner(layer), mtp_precision_attribute, False)
                        if args.vocab_cdf_ab or args.vocab_head_fp8_ab:
                            os.environ[selection_flag] = "0"
                        try:
                            native_parent = tuple(value.cpu() for value in reference(frame.payload))[:5]
                        finally:
                            if args.vocab_cdf_ab or args.vocab_head_fp8_ab:
                                os.environ[selection_flag] = "1"
                            if mtp_precision_ab:
                                for layer in draft.layers:
                                    setattr(precision_owner(layer), mtp_precision_attribute, True)
                    exact_record = all(torch.equal(native_parent[i], actual[i]) for i in (0, 1, 4))
                    parent_record = all(torch.equal(native_parent[i], parent_reference[i]) for i in (0, 1, 4))
                    probability_close = bool(torch.isclose(
                        native_parent[2], parent_reference[2], rtol=2e-5, atol=2e-7).all())
                    confidence_close = bool(torch.isclose(
                        native_parent[3], parent_reference[3], rtol=2e-5, atol=2e-7).all())
                    rng_same = torch.equal(native_parent[4], actual[4])
                    producer_change_allowed = (
                        ((mtp_precision_ab and mtp_teacher_qualified) or vocab_cdf_qualified
                         or head_teacher_qualified) and rng_same)
                    report.setdefault("native_parent_checks", []).append(
                        dict(
                            case=index,
                            record_rng_exact=exact_record,
                            probability_max_abs=float((native_parent[2] - actual[2]).abs().max()),
                            confidence_max_abs=float((native_parent[3] - actual[3]).abs().max()),
                            same_parent_reference_q_max_abs=float((native_parent[2] - parent_reference[2]).abs().max()),
                            parent_reference_record_exact=parent_record,
                            each_arm_matches_its_own_producer=parent_record and probability_close and confidence_close,
                            teacher_qualified_producer_change=mtp_precision_ab and producer_change_allowed,
                            distribution_preserving_proposal_draw_change=vocab_cdf_qualified,
                            teacher_qualified_vocabulary_precision=head_teacher_qualified,
                        )
                    )
                    if not ((exact_record or producer_change_allowed)
                            and parent_record and probability_close and confidence_close):
                        torch.save(
                            dict(parent=native_parent, parent_reference=parent_reference, candidate=actual),
                            root / f"native-parent-case{index}-rank{rank}.pt",
                        )
                        raise AssertionError("Selection changed record/RNG or exceeded the probability tolerance")
                report["checks"].append(
                    dict(
                        case=index,
                        record_exact=True,
                        rng_exact=True,
                        caches_exact=True,
                        full_repair=index in fallback_cases,
                        producer_rounding=producer_rounding,
                        confidence_max_abs=confidence_error,
                        probability_max_abs=float((actual[2] - expected[2]).abs().max()),
                    )
                )
            if args.protocol_writeback_ab:
                for index, (inputs, _) in enumerate(cases):
                    reset(index)
                    published = plan(*inputs)
                    torch.hpu.synchronize()
                    state_exact = (torch.equal(publication_state.proposal, published[2])
                                   and torch.equal(publication_state.counter, published[4])
                                   and torch.equal(publication_state.proposal_valid,
                                                   published[0][STATUS:STATUS + 1] == 0))
                    report.setdefault("sampling_publication_checks", []).append(
                        dict(case=index, proposal_counter_valid_exact=state_exact))
                    if not state_exact:
                        raise AssertionError("Captured sampling publication differs from serving state writes")
            report["capability_passed"] = True
            if args.fallback_contract_only:
                if frame is None:
                    raise ValueError("The hardware rollback contract requires the production repair journal")
                inputs = cases[0][0]
                parameters = inputs[6].clone()
                # Unfiltered full-vocabulary sampling is deliberately never
                # certified by K64. It must publish a repair marker.
                inputs[6][:, 1].fill_(1.0)
                reset(0)
                if args.nucleus_mass_ab:
                    os.environ["VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"] = "0"
                try:
                    expected = tuple(v.cpu() for v in reference(inputs))
                finally:
                    if args.nucleus_mass_ab:
                        os.environ["VLLM_HPU_DSV41_DSPARK_NUCLEUS_MASS"] = "1"
                expected_cache = tuple(layer.attention.swa.cpu().clone() for layer in draft.layers)
                reset(0)
                snapshots = tuple(
                    getattr(frame.journal, f"target_{i}").cpu().clone() for i in range(len(frame.journal.specs))
                )
                provisional = plan(*inputs)
                status = int(provisional[0][STATUS].cpu())
                if status != 2:
                    raise AssertionError("Uncovered official p/q was incorrectly certified")
                # Reproduce the unpublished following Target's write set.
                # All tested positions are produced by the captured device
                # journal, including compressed duplicates and page mappings.
                for i in range(len(frame.journal.specs)):
                    target = getattr(frame.journal, f"target_{i}")
                    target.index_fill_(0, getattr(frame.journal, f"indices_{i}"), 7)
                for layer in draft.layers:
                    layer.attention.swa.zero_()
                cursor.ids.zero_()
                cursor.positions.zero_()
                cursor.history.zero_()
                engram.histories.zero_()
                frame.journal.restore()
                frame.restore_protocol()
                target_exact = all(
                    torch.equal(getattr(frame.journal, f"target_{i}").cpu(), value) for i, value in enumerate(snapshots)
                )
                repaired = tuple(v.cpu() for v in reference(frame.payload))[:5]
                exact = target_exact and all(torch.equal(a, b) for a, b in zip(expected[:5], repaired, strict=True))
                exact = exact and all(
                    torch.equal(layer.attention.swa.cpu(), value)
                    for layer, value in zip(draft.layers, expected_cache, strict=True)
                )
                agreement = [None] * tp
                dist.all_gather_object(agreement, exact, group=get_tp_group().cpu_group)
                inputs[6].copy_(parameters)
                report["hardware_fallback_contract"] = dict(
                    status=status,
                    all_target_rows_exact=target_exact,
                    record_q_rng_draft_cache_exact=exact,
                    all_ranks_passed=all(agreement),
                )
                if not all(agreement):
                    raise AssertionError("Late full-distribution repair changed output/state/RNG")
                report.update(status="fallback_contract_passed", micro_qualified=False)
                return
            if args.correctness_only:
                report.update(status="correctness_passed", micro_qualified=False, performance_tested=False)
                return
            # The control chain reads Target state only to save the journal;
            # it never executes Target or mutates these buffers. Full decoder
            # H2D restoration is needed for the correctness cases, not for
            # every timed C5/verification call. Keep changing MTP/cursor/RNG
            # inputs resident, and drain their reset before timing on all ranks.
            report["timer_reset"] = (
                "same native sampled producer before both arms; resident reset in stream order; "
                "device markers exclude primer and its host dispatch; all actual repair cases retained"
                if args.queued_device_window else
                "resident MTP/cursor/RNG; unchanged Target journal source; drained before start")

            parent_fallback_cases = {row["case"] for row in report.get("parent_exact_repairs", [])}
            report["timed_parent_fallback_cases"] = sorted(parent_fallback_cases)

            readback_pending = []
            if args.record_readback_ab:
                from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
                from vllm_gaudi.ops.deepseek_v41_verify import _copy_commit_words
                readback_bridge, _ = resolve_device_runtime(tp)
                if getattr(readback_bridge, "dspark_record_readback_version", None) != 1:
                    raise ValueError("Private record readback ABI is missing")
                record_words = tuple(torch.empty((1, 4), dtype=torch.int64, device="hpu") for _ in range(4))
                copy_record_words = torch.compile(_copy_commit_words, backend="hpu_backend",
                                                   fullgraph=True, dynamic=False)

                def stage_readback(arm, record):
                    if arm == "native":
                        tickets = (readback_bridge.copy_dspark_record_to_host(record),)
                    else:
                        copy_record_words(record, record_words)
                        tickets = tuple(readback_bridge.copy_integer_record_to_host(word) for word in record_words)
                    readback_pending.extend(tickets)
                    return tickets

                for arm in ("parent", "native"):
                    for index in range(len(cases)):
                        reset(index, target_state=False)
                        producer = (parent_plan if arm == "parent" else plan)(*cases[index][0])
                        copies = stage_readback(arm, producer[0])
                        consume(producer[0])
                        actual_words = []
                        for host, completion in copies:
                            completion.synchronize()
                            actual_words.extend(host.tolist()[0])
                        if actual_words != producer[0].cpu().tolist():
                            raise AssertionError("Full speculative record readback changed")
                readback_pending.clear()
                report["record_readback_checks"] = dict(actual_cases=len(cases), all_words_exact=True,
                                                        consumer_before_host_read=True,
                                                        copies_per_round=dict(parent=4, native=1))

            def execute(arm, inputs, index):
                if arm == "native":
                    result = plan(*inputs)
                    repair = index in fallback_cases
                else:
                    result = None
                    repair = index in parent_fallback_cases
                    if repair:
                        result = parent_plan(*inputs)
                if arm == "native" or repair:
                    if repair:
                        # Both arms return the same repaired transaction. Its
                        # journal restore and exact sampling stay inside both
                        # timing windows; never compare an unrepaired parent.
                        frame.journal.restore()
                        frame.restore_protocol()
                        if arm == "parent" and mtp_precision_ab:
                            for layer in draft.layers:
                                setattr(precision_owner(layer), mtp_precision_attribute, False)
                        if arm == "parent" and args.vocab_cdf_ab:
                            os.environ[selection_flag] = "0"
                        try:
                            result = reference(frame.payload)
                        finally:
                            if arm == "parent" and args.vocab_cdf_ab:
                                os.environ[selection_flag] = "1"
                            if arm == "parent" and mtp_precision_ab:
                                for layer in draft.layers:
                                    setattr(precision_owner(layer), mtp_precision_attribute, True)
                    if args.protocol_writeback_ab and (arm == "parent" or repair):
                        publish_sampling(result)
                    return result
                if args.native_full and frame is not None:
                    # The production repair reference restores the same
                    # speculative Target/draft/cursor writes before sampling.
                    # Excluding this work would compare different boundaries.
                    frame.journal.restore()
                    frame.restore_protocol()
                result = parent_plan(*inputs) if parent_plan is not None else reference(inputs)
                if args.protocol_writeback_ab:
                    publish_sampling(result)
                return result

            for arm in ("parent", "native"):
                for index in range(len(cases)):
                    reset(index, target_state=False)
                    result = execute(arm, cases[index][0], index)
                    if args.round_input_ab:
                        if arm == "parent":
                            round_parent_consumer(result, cases[index][0][2])
                    else:
                        consume(result[0])
                    torch.hpu.synchronize()
            dist.barrier(group=get_tp_group().cpu_group)
            for iteration in range(3):
                medians = {}
                for arm in ("parent", "native"):
                    samples = []
                    wall_samples = []
                    from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
                    staged_plan = parent_plan if arm == "parent" else plan
                    staged_graph = _native_entries[staged_plan][0]
                    copy_start = staged_graph.input_update_copies()
                    batch_start = (staged_graph.input_staging_submissions()
                                   if hasattr(staged_graph, "input_staging_submissions") else None)
                    for sample in range(args.samples):
                        index = sample % len(cases)
                        reset(index, target_state=False)
                        torch.hpu.synchronize()
                        dist.barrier(group=get_tp_group().cpu_group)
                        start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        if args.queued_device_window:
                            if parent_plan is None:
                                raise ValueError(
                                    "A queued window requires the same native parent producer on both arms")
                            producer = parent_plan(*cases[index][0])
                            consume(producer[0])
                            # Restore resident MTP/cursor/RNG state in stream
                            # order, preserving all actual cases and repairs.
                            # No synchronize/barrier separates this producer
                            # from the timing marker and its consumer.
                            reset(index, target_state=False)
                        wall_start = time.perf_counter()
                        start.record()
                        result = execute(arm, cases[index][0], index)
                        if args.record_readback_ab:
                            stage_readback(arm, result[0])
                        if args.round_input_ab:
                            if arm == "parent":
                                round_parent_consumer(result, cases[index][0][2])
                        else:
                            consume(result[0])
                        stop.record()
                        stop.synchronize()
                        if args.record_readback_ab:
                            for _, completion in readback_pending:
                                completion.synchronize()
                            readback_pending.clear()
                        samples.append(start.elapsed_time(stop))
                        wall_samples.append((time.perf_counter() - wall_start) * 1000)
                    medians[arm] = statistics.median(samples)
                    report["rounds"].append(
                        dict(
                            iteration=iteration,
                            arm=arm,
                            device_ms=samples,
                            synchronized_wall_ms=wall_samples,
                            median_device_ms=medians[arm],
                            staging_graph_calls=args.samples * (2 if arm == "parent"
                                and args.queued_device_window else 1),
                            staging_logical_copies=staged_graph.input_update_copies() - copy_start,
                            staging_api_submissions=(None if batch_start is None else
                                staged_graph.input_staging_submissions() - batch_start),
                        )
                    )
                report.setdefault("paired_saved_ms", []).append(medians["parent"] - medians["native"])
            saved = statistics.median(report["paired_saved_ms"])
            report.update(
                status="passed",
                native_stats=prepared_group_stats(),
                saved_ms_per_round=saved,
                fallback_cases=sorted(fallback_cases),
                micro_qualified=all(x > 0 for x in report["paired_saved_ms"]),
            )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        if plan is not None:
            plan.close()
        if parent_plan is not None:
            parent_plan.close()
        shutdown_prepared_group_plans()


if __name__ == "__main__":
    main()
