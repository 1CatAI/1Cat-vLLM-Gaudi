# SPDX-License-Identifier: Apache-2.0
"""Real four-layer Full/Reuse chain: serial C1 versus request-owned batching."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import FunctionType, SimpleNamespace

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[int(os.environ["LOCAL_RANK"])]

if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[int(os.environ.get("LOCAL_RANK", "0"))])

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=8)
    p.add_argument("--reference-timing", action="store_true")
    p.add_argument("--correctness-only",
                   action="store_true",
                   help="Compare changing batch outputs/state with the C1 oracle without timing either path")
    p.add_argument("--native-replay",
                   action="store_true",
                   help="After the C1 oracle gate qualify and time the same candidate through native joint replay")
    p.add_argument("--diagnose-mhc", action="store_true")
    p.add_argument("--diagnose-layer", action="store_true")
    p.add_argument("--layers", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--layer-start", type=int, choices=(2, 20, 24), default=2)
    p.add_argument("--compare-main-fusions",
                   action="store_true",
                   help="Qualify main mHC and FFN preparation in the native request-batch chain")
    p.add_argument("--fusion-scope",
                   choices=("both", "mhc", "ffn", "weight_reuse", "compressor_pair", "compressor_gather",
                            "expert_prefetch", "expert_reuse", "w13_horizontal", "control_prefetch", "w13_control",
                            "full_index", "route_pack", "index_tiled_keys", "packed_mla", "packed_mla_sram",
                            "index_mla_sram"),
                   default="both",
                   help="Component bisection within the main fusion candidate")
    p.add_argument("--diagnose-fusion-replay",
                   action="store_true",
                   help="On a failed fusion check compare unchanged recipes against native bindings; no timing")
    p.add_argument("--compare-expert-finalize",
                   action="store_true",
                   help="Only fuse the terminal batched expert scaling and ordered reduction")
    p.add_argument("--compare-expert-transpose",
                   action="store_true",
                   help="Only reverse the MME operand orientation; retain the horizontal N256 parent")
    p.add_argument("--compare-vector-codecs",
                   action="store_true",
                   help="Compare PR38 group-parallel codecs in the unchanged real layer graph")
    p.add_argument("--compare-packed-vector",
                   action="store_true",
                   help="Only widen the fused SRAM KV decoder; keep QK/PV and production inputs")
    p.add_argument("--compare-reindex",
                   action="store_true",
                   help="Compare original batch scorer with bounded MME in a real Reindex group")
    p.add_argument("--compare-concurrent-moe",
                   type=int,
                   choices=(1, 4, 8, 16),
                   default=0,
                   help="Compare retained and SRAM-bounded MoE in the same real batched TP chain")
    p.add_argument("--compare-bounded-reindex",
                   action="store_true",
                   help="Qualified real Reindex group through bounded native joint replay")
    p.add_argument("--context-profile",
                   choices=("legacy", "2k-decode"),
                   default="legacy",
                   help="Keep archived fixtures explicit; 2k-decode also executes Full index scoring")
    p.add_argument("--long-reindex",
                   action="store_true",
                   help="Exercise short/whole scorer switching with resident 16K candidate rows")
    p.add_argument("--candidate-pool-layout",
                   choices=("legacy-scattered", "full-producer"),
                   default="legacy-scattered",
                   help="Freeze the actual Full-emitter candidate contract explicitly")
    p.add_argument("--diagnose-boundaries", action="store_true")
    p.add_argument("--tp2", action="store_true")
    p.add_argument("--prefix-checkpoint-contract",
                   action="store_true",
                   help="Restore request state into new slots before the real TP consumer; no timing baseline")
    p.add_argument("--reuse-reference", action="store_true", help="One C1 reference graph, explicit state transfers")
    p.add_argument("--reference-directory", type=Path, help="Check reused C1 graph against archived outputs")
    p.add_argument("--archived-output-only",
                   action="store_true",
                   help="Numerical diagnostic using saved C1 outputs; no reference execution or timing")
    p.add_argument("--request-chunk",
                   type=int,
                   choices=(8, 16, 32, 64),
                   default=64,
                   help="Check ordinary production batch partitions against the unchanged C1 oracle")
    p.add_argument("--recipe-output-row",
                   type=int,
                   help="Observe existing recipe outputs for one archived C1 row and its production batch")
    p.add_argument("--fragmented-reference",
                   action="store_true",
                   help="Compare one complete batch against serialized B1 plus a padded remainder")
    args = p.parse_args()
    if args.native_replay and (not args.tp2 or args.layers != 4 or args.reference_timing or args.diagnose_layer
                               or args.diagnose_boundaries or args.diagnose_mhc or args.recipe_output_row is not None or
                               args.archived_output_only or args.fragmented_reference or args.prefix_checkpoint_contract
                               or args.compare_reindex or args.compare_bounded_reindex or args.compare_concurrent_moe
                               or args.compare_main_fusions or args.compare_expert_finalize
                               or args.compare_expert_transpose or args.compare_vector_codecs
                               or args.compare_packed_vector):
        p.error("Native C1 oracle requires an uninstrumented TP2 four-layer candidate without another comparison")
    args.prepared = args.prepared.resolve()
    args.output = args.output.resolve()
    if os.environ.get("GRAPH_VISUALIZATION") == "1" and os.environ.get("DSV41_RUN_EVIDENCE"):
        graph_cwd = Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"compiler-rank{os.environ.get('LOCAL_RANK', '0')}"
        graph_cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(graph_cwd)
    if args.prefix_checkpoint_contract and (not args.tp2 or args.layers != 4 or args.batch > 32):
        p.error("Prefix consumer checks require TP2, four layers and disjoint source/destination slots")
    if args.long_reindex and not args.compare_bounded_reindex:
        p.error("Long-pool tactic qualification requires bounded native Reindex")
    if args.compare_bounded_reindex and args.batch != 8:
        p.error("The retained bounded Reindex dispatch is B8; other buckets keep the parent scorer")
    if args.archived_output_only and (args.reference_directory is None or args.reference_timing or args.diagnose_layer
                                      or args.diagnose_boundaries or args.batch != 64):
        p.error("Archived B64 output check requires a saved uninstrumented oracle and no timing")
    if args.recipe_output_row is not None and (args.reference_directory is None or not args.reuse_reference
                                               or args.archived_output_only or args.reference_timing
                                               or args.diagnose_layer or args.diagnose_boundaries
                                               or not 0 <= args.recipe_output_row < args.batch):
        p.error("Recipe output isolation requires a reused C1 graph and archived uninstrumented outputs")
    if (args.compare_reindex or args.compare_bounded_reindex) and args.layer_start != 24:
        p.error("Reindex comparison uses layers24..27")
    if sum(
            bool(x) for x in (args.compare_reindex, args.compare_concurrent_moe, args.compare_bounded_reindex,
                              args.compare_main_fusions, args.compare_expert_finalize, args.compare_vector_codecs,
                              args.compare_expert_transpose, args.compare_packed_vector)) > 1:
        p.error("Component attribution requires one primary change")
    if args.reuse_reference and args.reference_timing:
        p.error("State-transfer reference is a correctness oracle, not a timing baseline")
    if args.correctness_only and (args.reference_timing or args.compare_reindex or args.compare_concurrent_moe
                                  or args.compare_bounded_reindex or args.compare_main_fusions
                                  or args.compare_expert_finalize or args.compare_vector_codecs
                                  or args.compare_expert_transpose or args.compare_packed_vector
                                  or args.fragmented_reference or args.archived_output_only or args.recipe_output_row
                                  is not None or args.diagnose_mhc or args.diagnose_layer or args.diagnose_boundaries):
        p.error("Correctness-only mode requires the uninstrumented batch-versus-C1 production outputs")
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41_program import (
        PreparedDecoderLayer,
        PreparedLayerGroup,
        _weight_tree,
        load_weight_tree,
    )
    from vllm_gaudi.models.deepseek_v41_batch_program import PreparedBatchLayerGroup
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_batch_state import BatchLayerState
    from vllm_gaudi.ops.deepseek_v41_batch_replay import BatchSnapshot
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar, precision_config
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar, layer_selection

    torch.set_num_threads(1)
    torch.manual_seed(1188)
    rank, backend = 0, "hpu_backend"
    if args.tp2:
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
        rank = int(os.environ["LOCAL_RANK"])
        torch.hpu.set_device(rank)
        context = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2)))
        context.__enter__()
        init_distributed_environment(world_size=2,
                                     rank=rank,
                                     distributed_init_method="env://",
                                     local_rank=rank,
                                     backend="hccl")
        initialize_model_parallel(tensor_model_parallel_size=2)
        initialize_tp2_fused_ar_norm_runtime()
        if envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP:
            from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
            backend = make_backend()
        args.output = args.output.with_name(f"rank{rank}-" + args.output.name)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    first, stop = args.layer_start, args.layer_start + args.layers
    shard = PreparedV41Shard(args.prepared, first // 20, rank)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    specs = {k: v for k, v in shard.specs.items() if k.startswith(tuple(f"layers.{i}." for i in range(first, stop)))}
    weights = _weight_tree(specs)
    dc = precision_config(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG)
    wc = layer_selection(envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG)
    load_weight_tree(shard,
                     weights,
                     "hpu",
                     specs,
                     woa_sidecar=WoaFP8Sidecar(envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, shard),
                     woa_layers=wc["layers"],
                     dense_sidecar=DenseFP8Sidecar(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, shard),
                     dense_config=dc)
    from deepseek_v41_micro_contexts import micro_contexts
    b, capacity = args.batch, 64
    context_positions, context_pages = micro_contexts(b, first, args.context_profile, long_reindex=args.long_reindex)
    if args.prefix_checkpoint_contract:
        context_positions = [1920] * b
        context_pages = max(context_pages, 16)
    source = max(i for i in config["kv_source_layer_ids"] if i <= first)
    ratio = config["compress_ratios"][source]
    page_count = context_pages
    page_rows = 128 // ratio
    shared = PagedCSA2SharedState(config, source, stop, "hpu", 1048576)
    cache = shared.sources[str(source)]
    cache.main = torch.zeros((capacity * page_count + 1) * page_rows, 288, dtype=torch.uint8, device="hpu")
    cache.index = torch.zeros((capacity * page_count + 1) * page_rows, 68, dtype=torch.uint8, device="hpu")
    reduce = lambda x, **kwargs: x
    gather = lambda x, dim: torch.cat((x, x), dim)
    if args.tp2:
        from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
        reduce, gather = stage_collectives(rank, True)
    lookup = mxfp4_bf16_lut("hpu")

    def make_layers():
        return torch.nn.ModuleList([
            PreparedDecoderLayer(weights.layers.get_submodule(str(i)), config, i, shared, True, lookup, reduce, gather,
                                 "hpu") for i in range(first, stop)
        ])

    layers = make_layers()
    for layer in layers:
        at = layer.attention
        at.woa_fp8 = layer.layer in wc["layers"]
        at.prepare_output_weight()
        at.prepare_qkv_input_weight()
        at.prepare_compressor_input_weight()
        at.set_search_length(1048576)
        at.batch_state = BatchLayerState(capacity, "hpu", compressor=at.owns_kv and at.ratio == 2)
        if args.compare_packed_vector:
            at.batch_packed_mla = at.batch_packed_mla_sram = at.batch_packed_mla_vector = True
        layer.moe.prepare_shared_gate_up_weight()
        if args.compare_expert_transpose:
            layer.moe.batch_expert_transpose_mme = True
        if args.compare_expert_finalize:
            layer.moe.batch_expert_direct_finalize = True
        if args.compare_main_fusions:
            layer.batch_main_fusions = True
            layer.batch_mhc_fusion = args.fusion_scope in ("both", "mhc")
            layer.batch_ffn_fusion = args.fusion_scope in ("both", "ffn")
            layer.batch_control_reuse = args.fusion_scope == "weight_reuse"
            layer.batch_control_prefetch = (args.fusion_scope in ("control_prefetch", "w13_control")
                                            or (args.fusion_scope in ("full_index", "route_pack", "index_tiled_keys",
                                                                      "packed_mla", "packed_mla_sram", "index_mla_sram")
                                                and b >= 32))
            at.batch_full_index_mme = args.fusion_scope in ("full_index", "route_pack", "index_tiled_keys",
                                                            "packed_mla", "packed_mla_sram", "index_mla_sram")
            at.batch_index_tiled_keys = args.fusion_scope in ("index_tiled_keys", "index_mla_sram")
            at.batch_packed_mla = args.fusion_scope in ("packed_mla", "packed_mla_sram", "index_mla_sram")
            at.batch_packed_mla_sram = args.fusion_scope in ("packed_mla_sram", "index_mla_sram")
            at.batch_compressor_pair = args.fusion_scope == "compressor_pair"
            at.batch_compressor_gather = args.fusion_scope == "compressor_gather"
            layer.moe.batch_prefetch_w2 = args.fusion_scope == "expert_prefetch"
            layer.moe.batch_route_pack = args.fusion_scope == "route_pack"
            layer.moe.batch_expert_reuse = args.fusion_scope == "expert_reuse"
            layer.moe.batch_w13_horizontal = args.fusion_scope in ("w13_horizontal", "w13_control", "full_index",
                                                                   "route_pack", "index_tiled_keys", "packed_mla",
                                                                   "packed_mla_sram", "index_mla_sram")
            layer.prepare_mhc_control_weights()
    # This is an internal four-layer fragment, not the PP1 decoder tail.
    stage = SimpleNamespace(layers=layers, shared=shared, config={"text_config": config}, dspark=False, pp_rank=0)
    grouped = PreparedBatchLayerGroup(stage, 0, args.layers)
    observer = None
    if args.recipe_output_row is not None:
        from deepseek_v41_recipe_outputs import RecipeOutputs
        observer = RecipeOutputs(b * 4 * 5120)
    candidate = torch.compile(grouped, backend=backend, fullgraph=True, dynamic=False)
    slots_cpu = torch.randperm(capacity)[:b].int()
    pos_cpu = torch.tensor(context_positions, dtype=torch.int32)
    page_cpu = torch.zeros(b, 8192, dtype=torch.int32)
    for i, slot in enumerate(slots_cpu):
        page_cpu[i, :page_count] = torch.arange(page_count) + 1 + int(slot) * page_count
    slots, positions, pages = [v.to("hpu") for v in (slots_cpu, pos_cpu, page_cpu)]
    ids = torch.ones(b, dtype=torch.int32, device="hpu")
    hidden = torch.randn(b, 4, 5120).bfloat16().to("hpu")
    pre = torch.rand(b, 4).to("hpu")
    if args.diagnose_mhc:
        from vllm_gaudi.ops.deepseek_v41_math import hc_pre, hc_post, rms_norm, row_mean_square
        w = layers[0].weights

        def mhc(x, previous, request_batch):
            flat = x.flatten(1).float()
            projection = torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2(flat, w.hc_attn_fn)
            rrms = torch.rsqrt(row_mean_square(flat, request_batch=request_batch) + layers[0].eps)
            value, next_pre, post, comb = hc_pre(x,
                                                 previous,
                                                 w.hc_attn_fn,
                                                 w.hc_attn_scale,
                                                 w.hc_attn_base,
                                                 layers[0].eps,
                                                 layers[0].hc_eps,
                                                 layers[0].iterations,
                                                 request_batch=request_batch)
            norm = rms_norm(value, w.attn_norm.weight, layers[0].eps, request_batch=request_batch)
            result = hc_post(norm, x, post, comb)
            return projection, rrms, value, next_pre, post, comb, norm, result

        fn = torch.compile(mhc, backend="hpu_backend", fullgraph=True, dynamic=False)
        with torch.inference_mode():
            serial = [tuple(v.cpu() for v in fn(hidden[i:i + 1], pre[i:i + 1], False)) for i in range(b)]
            actual = tuple(v.cpu() for v in fn(hidden, pre, True))
        expected = tuple(torch.cat([row[i] for row in serial]) for i in range(len(actual)))
        checks = []
        for name, x, y in zip(("projection", "rrms", "collapsed", "pre", "post", "comb", "norm", "post_result"),
                              expected, actual):
            checks.append(
                dict(name=name,
                     exact=torch.equal(x, y),
                     max_abs=float((x.float() - y.float()).abs().max()),
                     different=int((x != y).sum()),
                     elements=x.numel()))
        args.output.write_text(json.dumps(checks, indent=2))
        torch.save(dict(reference=expected, actual=actual), args.output.parent / f"rank{rank}-mhc.pt")
        print(json.dumps(checks), flush=True)
        return
    selections = tuple(torch.full((b, 512), -1, dtype=torch.int32, device="hpu") for _ in shared.topk)
    pool = torch.full((b, 2048), -1, dtype=torch.int32, device="hpu")
    if first == 24:
        host_pool = torch.full((b, 2048), -1, dtype=torch.int32)
        for row in range(b):
            count = (int(pos_cpu[row]) + 8) // 8
            blocks = (torch.arange(count).int()
                      if args.candidate_pool_layout == "full-producer" else torch.randperm(count).int())
            host_pool[row, :len(blocks)] = blocks
        # The current Full emitter publishes the complete causal block prefix
        # while it fits in 2048 slots. Scattered valid slots beyond that prefix
        # violate the bounded threshold/emitter contract in current main.
        if args.candidate_pool_layout == "legacy-scattered":
            host_pool = host_pool[:, torch.randperm(2048)].contiguous()
        pool.copy_(host_pool)
    ready = (torch.zeros(b, dtype=torch.int32, device="hpu"), )
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
    buffers = []
    for layer in layers:
        state = layer.attention.batch_state
        state.swa.copy_(pack_swa(torch.randn(capacity * 256, 512).bfloat16()).to("hpu"))
        if hasattr(state, "kv_history"):
            state.kv_history.copy_(torch.randn(capacity * 8, 512))
            state.score_history.copy_(torch.randn(capacity * 8, 512))
        buffers.extend(state.buffers())
    if first == 2:
        # Preserve the archived Full/Reuse fixture's random draw order.
        cache.main.copy_(pack_fp4(torch.randn(cache.main.shape[0], 512).bfloat16(), 16).to("hpu"))
        cache.index.copy_(pack_fp4(torch.randn(cache.index.shape[0], 128).bfloat16(), 32).to("hpu"))
    else:
        for start in range(0, cache.main.shape[0], 4096):
            count = min(4096, cache.main.shape[0] - start)
            cache.main[start:start + count].copy_(pack_fp4(torch.randn(count, 512).bfloat16(), 16).to("hpu"))
            cache.index[start:start + count].copy_(pack_fp4(torch.randn(count, 128).bfloat16(), 32).to("hpu"))
    buffers.extend((cache.main, cache.index))
    original = [x.clone() for x in buffers]

    def restore():
        for dest, source in zip(buffers, original):
            dest.copy_(source)

    if args.prefix_checkpoint_contract:
        from deepseek_v41_prefix_micro import check_prefix_consumers
        from deepseek_v41_native_fragment import NativeBatchFragment
        inputs = hidden, pre, positions, ids, (), slots, pages, selections, pool, ready, ready
        native = NativeBatchFragment(grouped, candidate, buffers, restore,
                                     sum(2 for layer in layers if layer.attention.owns_index), pos_cpu.tolist())
        with torch.inference_mode():
            report = check_prefix_consumers(native, inputs, layers, buffers, restore, capacity, rank)
        args.output.write_text(json.dumps(report, indent=2))
        native.close()
        native.metadata.native_completion = None
        from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
        from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
        shutdown_prepared_group_plans()
        destroy_model_parallel()
        destroy_distributed_environment()
        context.__exit__(None, None, None)
        return

    if (args.compare_reindex or args.compare_concurrent_moe or args.compare_bounded_reindex or args.compare_main_fusions
            or args.compare_expert_finalize or args.compare_vector_codecs or args.compare_expert_transpose
            or args.compare_packed_vector):
        old_layers = make_layers()
        for old, current in zip(old_layers, layers):
            old.attention.batch_state = current.attention.batch_state
            # Normal preparation releases the original split weights. Reuse
            # the same immutable prepared buffers, as the C1 oracle does.
            for name in ("woa_fp8", "_fused_qkv_weight", "_fused_qkv_quantized", "_fused_compressor_weight",
                         "_fused_compressor_kv_width"):
                setattr(old.attention, name, getattr(current.attention, name))
            old.attention.set_search_length(1048576)
            if args.compare_packed_vector:
                old.attention.batch_packed_mla = old.attention.batch_packed_mla_sram = True
                old.attention.batch_packed_mla_vector = False
            if args.compare_main_fusions:
                old.batch_main_fusions = False
                old.batch_mhc_fusion = old.batch_ffn_fusion = False
                old.batch_control_reuse = False
                old.batch_control_prefetch = args.fusion_scope in ("full_index", "route_pack", "index_tiled_keys",
                                                                   "packed_mla", "packed_mla_sram",
                                                                   "index_mla_sram") and b >= 32
                old.attention.batch_full_index_mme = args.fusion_scope in ("route_pack", "index_tiled_keys",
                                                                           "packed_mla", "packed_mla_sram",
                                                                           "index_mla_sram")
                old.attention.batch_index_tiled_keys = args.fusion_scope == "index_mla_sram"
                old.attention.batch_packed_mla = args.fusion_scope == "packed_mla_sram"
                old.attention.batch_packed_mla_sram = False
                old.attention.batch_compressor_pair = False
                old.attention.batch_compressor_gather = False
                old.moe.batch_prefetch_w2 = False
                old.moe.batch_route_pack = False
                old.moe.batch_expert_reuse = False
                old.moe.batch_w13_horizontal = args.fusion_scope in ("full_index", "route_pack", "index_tiled_keys",
                                                                     "packed_mla", "packed_mla_sram", "index_mla_sram")
            if args.compare_expert_transpose:
                old.moe.batch_expert_transpose_mme = False
            if args.compare_expert_finalize:
                old.moe.batch_expert_direct_finalize = False
            if args.compare_reindex:
                old.attention.batch_reindex_mme = False
                current.attention.batch_reindex_mme = True
            if args.compare_bounded_reindex:
                old.attention.bounded_reindex = False
                current.attention.bounded_reindex = True
            if args.compare_concurrent_moe:
                old.moe.concurrent_moe_rows = 0
                current.moe.concurrent_moe_rows = args.compare_concurrent_moe
            old.moe.shared_gate_up_weight = current.moe.shared_gate_up_weight
        old_stage = SimpleNamespace(layers=old_layers,
                                    shared=shared,
                                    config={"text_config": config},
                                    dspark=False,
                                    pp_rank=0)
        old_grouped = PreparedBatchLayerGroup(old_stage, 0, args.layers)
        reference = torch.compile(old_grouped, backend=backend, fullgraph=True, dynamic=False)
        inputs = hidden, pre, positions, ids, (), slots, pages, selections, pool, ready, ready
        native_compare = (args.compare_bounded_reindex or args.compare_main_fusions or args.compare_expert_finalize
                          or args.compare_vector_codecs or args.compare_expert_transpose or args.compare_packed_vector)
        if native_compare:
            if (not args.tp2 or args.layers != 4
                    or (args.compare_bounded_reindex and not envs.VLLM_HPU_DSV41_REINDEX_BOUNDED_PLAN)):
                raise ValueError("Bounded group qualification requires real TP, four layers and the production flag")
            from deepseek_v41_native_fragment import NativeBatchFragment
            candidate = NativeBatchFragment(grouped, candidate, buffers, restore,
                                            sum(2 for layer in layers if layer.attention.owns_index), pos_cpu.tolist())
            reference = NativeBatchFragment(old_grouped, reference, buffers, restore,
                                            sum(2 for layer in old_layers if layer.attention.owns_index),
                                            pos_cpu.tolist())

        def select_codec(fn):
            if args.compare_vector_codecs:
                # Select the documented diagnostic switch before each arm's
                # normal compilation/capture. Once prepared, native replay
                # retains that graph; no operator/function is monkeypatched.
                os.environ["VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT"] = "1" if fn is candidate else "0"

        report = dict(
            batch=b,
            fusion_scope=args.fusion_scope if args.compare_main_fusions else None,
            context_profile=args.context_profile,
            initial_positions=context_positions,
            cache_pages_per_request=context_pages,
            first_layer=first,
            layers=args.layers,
            comparison=(
                "PR38 vector group codecs, both native joint replay" if args.compare_vector_codecs else
                "Vector packed KV decoding, same QK/PV through native batch TP consumers"
                if args.compare_packed_vector else "Batch direct expert finalize, both native joint replay"
                if args.compare_expert_finalize else "Transposed expert MME through real TP consumers" if args.
                compare_expert_transpose else "Main mHC/FFN fusions through native batch TP consumers" if args.
                compare_main_fusions else "Original batched TPC scorer versus bounded MME Reindex" if args.
                compare_reindex else "Retained versus bounded Reindex, both native joint replay" if args.
                compare_bounded_reindex else "Retained batch versus SRAM-bounded MoE, identical surrounding TP chain"),
            concurrent_moe_rows=args.compare_concurrent_moe,
            long_reindex=args.long_reindex,
            candidate_pool_layout=args.candidate_pool_layout,
            effective_candidate_moe_rows=layers[0].moe.concurrent_moe_rows,
            effective_reference_moe_rows=old_layers[0].moe.concurrent_moe_rows,
            checks=[],
            timings={})
        with torch.inference_mode():
            if native_compare:
                for fn in (reference, candidate):
                    select_codec(fn)
                    for _ in range(3):
                        restore()
                        fn(*inputs)
                        torch.hpu.synchronize()
                    fn.require_ready()
            lengths = (16383, 511, 8191, 8192, 2047, 530, 16383) if args.long_reindex else None
            for change in range(len(lengths) if lengths else 3 if native_compare else 2):
                if change:
                    hidden.copy_(torch.randn(b, 4, 5120).bfloat16())
                if native_compare:
                    host_positions = (torch.full_like(pos_cpu, lengths[change])
                                      if lengths else torch.full_like(pos_cpu, 511) if change == 1 else pos_cpu)
                    positions.copy_(host_positions)
                    candidate.host_positions = reference.host_positions = host_positions.tolist()
                    if change == 2:
                        slots.copy_(slots_cpu.roll(1))
                        pages.copy_(page_cpu.roll(1, 0))
                outputs, states = {}, {}
                for name, fn in (("reference", reference), ("candidate", candidate)):
                    select_codec(fn)
                    restore()
                    value = fn(*inputs)
                    if native_compare:
                        outputs[name] = tuple(v.cpu() for v in value)
                    else:
                        outputs[name] = (value[0].cpu(), value[1].cpu(), *(v.cpu() for v in value[2]), value[3].cpu())
                    states[name] = [v.cpu() for v in buffers]
                check = dict(
                    change=change,
                    positions=host_positions.tolist() if native_compare else pos_cpu.tolist(),
                    outputs=[
                        torch.equal(a, z) for a, z in zip(outputs["reference"], outputs["candidate"], strict=True)
                    ],
                    states=[torch.equal(a, z) for a, z in zip(states["reference"], states["candidate"], strict=True)])
                report["checks"].append(check)
                args.output.write_text(json.dumps(report, indent=2))
                torch.save(outputs, args.output.parent / f"rank{rank}-reindex-{change}.pt")
                if not all(check["outputs"] + check["states"]):
                    if args.diagnose_fusion_replay and args.compare_main_fusions:
                        from torch.utils._pytree import tree_flatten
                        from unittest.mock import patch
                        from vllm_gaudi.ops.tp2_prepared_plan import PreparedGroupModule
                        direct = {}
                        for name, fn in (("reference", reference), ("candidate", candidate)):
                            restore()
                            for dst, src in zip(tree_flatten(fn.fixed)[0], tree_flatten(inputs)[0], strict=True):
                                dst.copy_(src)
                            # Diagnostic only: invoke the existing recipes and
                            # collectives, without preparing another ownerless
                            # native plan. Restore the method before proceeding.
                            with patch.object(PreparedGroupModule, "forward",
                                              lambda module, values: module.original(values)):
                                direct[name] = tuple(v.cpu() for v in fn.flatten(fn.compiled(*fn.fixed)))
                        report["replay_diagnostic"] = dict(
                            scope="Same compiled recipes with ordinary submission; no graph exports or timing",
                            direct_equal=[
                                torch.equal(a, z) for a, z in zip(direct["reference"], direct["candidate"], strict=True)
                            ],
                            native_vs_direct={
                                name: [torch.equal(a, z) for a, z in zip(outputs[name], direct[name], strict=True)]
                                for name in direct
                            })
                        torch.save(direct, args.output.parent / f"rank{rank}-direct-{change}.pt")
                        args.output.write_text(json.dumps(report, indent=2))
                    raise RuntimeError(f"Real TP candidate group differs: {check}")
            for name, fn in (("reference", reference), ("candidate", candidate)):
                if name == "reference" and not args.reference_timing:
                    continue
                select_codec(fn)
                samples = []
                for _ in range(7):
                    torch.hpu.synchronize()
                    begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start = time.perf_counter()
                    begin.record()
                    fn(*inputs)
                    end.record()
                    end.synchronize()
                    samples.append(dict(device_ms=begin.elapsed_time(end),
                                        wall_ms=(time.perf_counter() - start) * 1000))
                report["timings"][name] = samples
                args.output.write_text(json.dumps(report, indent=2))
        if native_compare:
            from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
            report["native_stats"] = prepared_group_stats()
            args.output.write_text(json.dumps(report, indent=2))
            candidate.close()
            candidate.metadata.native_completion = None
            reference.close()
            reference.metadata.native_completion = None
        if args.tp2:
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
            from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
            shutdown_prepared_group_plans()
            destroy_model_parallel()
            destroy_distributed_environment()
            context.__exit__(None, None, None)
        return

    if args.fragmented_reference:
        if args.reuse_reference or args.diagnose_boundaries or args.diagnose_layer or args.reference_directory:
            p.error("Fragmented scheduling qualification is a separate complete-chain experiment")
        single = torch.compile(PreparedBatchLayerGroup(stage, 0, args.layers),
                               backend=backend,
                               fullgraph=True,
                               dynamic=False)
        one_inputs = tuple(v[:1].clone() for v in (hidden, pre, positions, ids, slots, pages))
        one_selections = tuple(v[:1].clone() for v in selections)
        one_pool, one_ready = pool[:1].clone(), tuple(v[:1].clone() for v in ready)
        remaining_slots = slots.clone()
        remaining_positions = positions.clone()
        remaining_slots[0] = -1
        remaining_positions[0] = -1

        def joined():
            return candidate(hidden, pre, positions, ids, (), slots, pages, selections, pool, ready, ready)[:2]

        def fragmented():
            x, p0, pos, token, slot, page = one_inputs
            first = single(x, p0, pos, token, (), slot, page, one_selections, one_pool, one_ready, one_ready)[:2]
            rest = candidate(hidden, pre, remaining_positions, ids, (), remaining_slots, pages, selections, pool, ready,
                             ready)[:2]
            return first, rest

        report = dict(batch=b, layers=args.layers, comparison="B1 + padded B(B-1) versus full B", checks=[], timings={})
        with torch.inference_mode():
            for change in range(2):
                if change:
                    hidden.copy_(torch.randn(b, 4, 5120).bfloat16())
                    one_inputs[0].copy_(hidden[:1])
                restore()
                expected = tuple(v.cpu() for v in joined())
                expected_state = [v.cpu() for v in buffers]
                restore()
                first, rest = fragmented()
                actual = tuple(torch.cat((a.cpu(), z[1:].cpu())) for a, z in zip(first, rest))
                check = dict(outputs=[torch.equal(a, z) for a, z in zip(expected, actual)],
                             states=[
                                 torch.equal(v.cpu()[64:] if i >= len(buffers) - 2 else v.cpu(),
                                             z[64:] if i >= len(buffers) - 2 else z)
                                 for i, (v, z) in enumerate(zip(buffers, expected_state))
                             ])
                report["checks"].append(check)
                args.output.write_text(json.dumps(report, indent=2))
                if not all(check["outputs"] + check["states"]):
                    raise RuntimeError(f"Fragmented request-batch contract failed: {check}")
            for name, fn in (("fragmented", fragmented), ("joined", joined)):
                samples = []
                for _ in range(7):
                    torch.hpu.synchronize()
                    begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start = time.perf_counter()
                    begin.record()
                    fn()
                    end.record()
                    end.synchronize()
                    samples.append(dict(device_ms=begin.elapsed_time(end),
                                        wall_ms=(time.perf_counter() - start) * 1000))
                report["timings"][name] = samples
                args.output.write_text(json.dumps(report, indent=2))
                print(name, statistics.median(v["device_ms"] for v in samples), flush=True)
        if args.tp2:
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
            from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
            shutdown_prepared_group_plans()
            destroy_model_parallel()
            destroy_distributed_environment()
            context.__exit__(None, None, None)
        return

    if args.archived_output_only:
        import hashlib
        archived = args.reference_directory / f"rank{rank}-output-0.pt"
        previous = torch.load(archived, weights_only=True)
        if not torch.equal(hidden.cpu(), previous["input"]):
            raise RuntimeError("Archived production input differs from current fixture")
        outputs = []
        with torch.inference_mode():
            restore()
            for start in range(0, b, args.request_chunk):
                stop = min(start + args.request_chunk, b)
                result = candidate(hidden[start:stop], pre[start:stop], positions[start:stop],
                                   ids[start:stop], (), slots[start:stop], pages[start:stop],
                                   tuple(v[start:stop] for v in selections), pool[start:stop],
                                   tuple(v[start:stop] for v in ready), tuple(v[start:stop] for v in ready))
                outputs.append(tuple(v.cpu() for v in result[:2]))
        actual = tuple(torch.cat([v[i] for v in outputs]) for i in range(2))
        checks = [
            dict(exact=torch.equal(x, y),
                 max_abs=float((x.float() - y.float()).abs().max()),
                 different_rows=(x != y).flatten(1).any(1).nonzero().flatten().tolist())
            for x, y in zip(previous["reference"], actual, strict=True)
        ]
        args.output.write_text(
            json.dumps(dict(batch=b,
                            chunk=args.request_chunk,
                            layers=args.layers,
                            oracle=str(archived),
                            oracle_sha256=hashlib.sha256(archived.read_bytes()).hexdigest(),
                            checks=checks,
                            scope="Uninstrumented production graph outputs only; no state or timing qualification"),
                       indent=2))
        torch.save(dict(input=hidden.cpu(), reference=previous["reference"], actual=actual),
                   args.output.parent / f"rank{rank}-output-0.pt")
        if args.tp2:
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
            from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
            shutdown_prepared_group_plans()
            destroy_model_parallel()
            destroy_distributed_environment()
            context.__exit__(None, None, None)
        return

    refs, reference_layers = [], []
    for row, slot in enumerate(slots_cpu.tolist()[:1] if args.reuse_reference else slots_cpu.tolist()):
        old_layers = make_layers()
        for old, current in zip(old_layers, layers):
            src, dst = current.attention, old.attention
            state = src.batch_state
            for name in ("woa_fp8", "_fused_qkv_weight", "_fused_qkv_quantized", "_fused_compressor_weight",
                         "_fused_compressor_kv_width"):
                setattr(dst, name, getattr(src, name))
            dst.swa = state.swa[slot * 256:(slot + 1) * 256]
            if hasattr(state, "kv_history"):
                dst.kv_history = state.kv_history[slot * 8:(slot + 1) * 8]
                dst.score_history = state.score_history[slot * 8:(slot + 1) * 8]
            if args.reuse_reference:
                for name in ("swa", "kv_history", "score_history"):
                    if hasattr(state, name):
                        setattr(dst, name, getattr(dst, name).clone())
            dst.set_search_length(1048576)
            old.moe.shared_gate_up_weight = current.moe.shared_gate_up_weight
        old_stage = SimpleNamespace(layers=old_layers, pp_rank=0, config={"text_config": config})
        reference_layers.append(old_layers)
        old_group = PreparedLayerGroup(old_stage, 0, args.layers, decode=True)

        def ref(x, p, pos, token, page, module=old_group):
            shared.block_table = page
            if args.diagnose_boundaries:
                results = ()
                for layer in module.layers:
                    x, p, _ = layer(x, p, pos, torch.zeros_like(token, dtype=torch.bool), decode=True)
                    results += (x, p)
                return results
            return module(x, p, pos, token, ())[:2]

        entry = FunctionType(ref.__code__.replace(co_name=f"serial_block_request_{row}"),
                             ref.__globals__,
                             argdefs=ref.__defaults__,
                             closure=ref.__closure__)
        refs.append(torch.compile(entry, backend=backend, fullgraph=True, dynamic=False))

    reference_inputs = tuple(x.clone() for x in (hidden[:1], pre[:1], positions[:1], ids[:1], pages[0]))

    def transfer_reference(row, *, save=False):
        if not save and first > config["candidate_source_layer_id"]:
            # This fragment starts after the Full layer that produces the
            # candidate pool. Give the serial oracle the same request-owned
            # pool as the batch; its default all-invalid pool is not a valid
            # reference for Reindex attention.
            shared.candidate_pool[:1].copy_(pool[row:row + 1])
        if not args.reuse_reference:
            return
        slot = int(slots_cpu[row])
        for old, current in zip(reference_layers[0], layers):
            state = current.attention.batch_state
            for name, width in (("swa", 256), ("kv_history", 8), ("score_history", 8)):
                if hasattr(state, name):
                    bank = getattr(state, name)[slot * width:(slot + 1) * width]
                    fixed = getattr(old.attention, name)
                    (bank if save else fixed).copy_(fixed if save else bank)

    def row_inputs(row):
        values = (hidden[row:row + 1], pre[row:row + 1], positions[row:row + 1], ids[row:row + 1], pages[row])
        if not args.reuse_reference:
            return values
        for dst, src in zip(reference_inputs, values):
            dst.copy_(src)
        return reference_inputs

    if args.diagnose_layer:
        from vllm_gaudi.ops.deepseek_v41_math import hc_pre, hc_post, rms_norm

        def intermediates(layer, x, previous, pos, page, batch_request):
            w = layer.weights
            val, nxt, post, comb = hc_pre(x,
                                          previous,
                                          w.hc_attn_fn,
                                          w.hc_attn_scale,
                                          w.hc_attn_base,
                                          layer.eps,
                                          layer.hc_eps,
                                          layer.iterations,
                                          request_batch=batch_request)
            norm = rms_norm(val, w.attn_norm.weight, layer.eps, request_batch=batch_request)
            before = (val, nxt, post, comb, norm)
            dependency = (post, comb) if envs.VLLM_HPU_DSV41_MHC_SCHEDULE else ()
            if batch_request:
                attn = layer.attention.forward_batch(norm,
                                                     pos,
                                                     slots,
                                                     page,
                                                     selections[0],
                                                     pool,
                                                     ready[0],
                                                     ready[0],
                                                     ready_outputs=dependency)[0]
            else:
                shared.block_table = page
                attn = layer.attention(norm, pos, ready_outputs=dependency, decode=True)
            residual = hc_post(attn, x, post, comb)
            val, nxt, post, comb = hc_pre(residual,
                                          nxt,
                                          w.hc_ffn_fn,
                                          w.hc_ffn_scale,
                                          w.hc_ffn_base,
                                          layer.eps,
                                          layer.hc_eps,
                                          layer.iterations,
                                          request_batch=batch_request)
            norm = rms_norm(val, w.ffn_norm.weight, layer.eps, request_batch=batch_request)
            dependency = (post, comb) if envs.VLLM_HPU_DSV41_MHC_SCHEDULE else ()
            moe = layer.moe(norm,
                            torch.zeros_like(pos, dtype=torch.bool),
                            ready_outputs=dependency,
                            ordinary_decode=batch_request)
            return (*before, attn, residual, val, nxt, post, comb, norm, moe, hc_post(moe, residual, post, comb))

        with torch.inference_mode():
            restore()
            reference = []
            diagnostic_reference = None
            for i in range(b):
                group = reference_layers[0 if args.reuse_reference else i]

                def reference_entry(x, previous, pos, page, layer=group[0]):
                    return intermediates(layer, x, previous, pos, page, False)

                entry = FunctionType(reference_entry.__code__.replace(co_name=f"sublayer_{i}"),
                                     reference_entry.__globals__,
                                     argdefs=reference_entry.__defaults__,
                                     closure=reference_entry.__closure__)
                if not args.reuse_reference or diagnostic_reference is None:
                    diagnostic_reference = torch.compile(entry, backend=backend, fullgraph=True, dynamic=False)
                transfer_reference(i)
                x, prev, pos, _, page = row_inputs(i)
                reference.append(tuple(v.cpu() for v in diagnostic_reference(x, prev, pos, page)))
            restore()
            fn = torch.compile(lambda x, p, pos, page: intermediates(layers[0], x, p, pos, page, True),
                               backend=backend,
                               fullgraph=True,
                               dynamic=False)
            actual = tuple(v.cpu() for v in fn(hidden, pre, positions, pages))
        expected = tuple(torch.cat([row[i] for row in reference]) for i in range(len(actual)))
        checks = [
            dict(name=name,
                 exact=torch.equal(x, y),
                 max_abs=float((x.float() - y.float()).abs().max()),
                 different=int((x != y).sum()),
                 elements=x.numel()) for name, x, y in zip(("collapsed", "pre", "post", "comb", "norm", "attention",
                                                            "attn_residual", "ffn_collapsed", "ffn_pre", "ffn_post",
                                                            "ffn_comb", "ffn_norm", "moe", "final"), expected, actual)
        ]
        args.output.write_text(json.dumps(checks, indent=2))
        torch.save(dict(reference=expected, actual=actual), args.output.parent / f"rank{rank}-sublayers.pt")
        print(json.dumps(checks), flush=True)
        return

    if args.diagnose_boundaries:

        def traced(x, p, pos, token, engram, slots, pages, selections, pool, main_ready, index_ready):
            sels, main, index = list(selections), list(main_ready), list(index_ready)
            result = ()
            for layer, (isrc, ksrc) in zip(grouped.layers, grouped.mapping):
                x, p, s, pool, mdone, idone = layer.forward_batch(x, p, pos, torch.zeros_like(token, dtype=torch.bool),
                                                                  None, slots, pages, sels[isrc], pool, main[ksrc],
                                                                  index[ksrc])
                sels[isrc], main[ksrc], index[ksrc] = s, mdone, idone
                result += (x, p)
            return result

        candidate = torch.compile(traced, backend="hpu_backend", fullgraph=True, dynamic=False)

    def serial():
        values = []
        for i in range(b):
            transfer_reference(i)
            row = refs[0 if args.reuse_reference else i](*row_inputs(i))
            values.append(tuple(v.clone() for v in row) if args.reuse_reference else row)
            transfer_reference(i, save=True)
        return tuple(torch.cat([v[j] for v in values]) for j in range(len(values[0])))

    def batch():
        result = candidate(hidden, pre, positions, ids, (), slots, pages, selections, pool, ready, ready)
        return result if args.diagnose_boundaries else result[:2]

    if observer is not None:
        row = args.recipe_output_row
        saved = torch.load(args.reference_directory / f"rank{rank}-output-0.pt", weights_only=True)
        if not torch.equal(saved["input"], hidden.cpu()):
            raise RuntimeError("Recipe diagnostic input differs from the frozen oracle")
        record = {}
        with torch.inference_mode():
            for name, fn in (("serial", lambda: refs[0](*row_inputs(row))), ("batch", batch)):
                restore()
                transfer_reference(row)
                observer.records = []
                outputs = tuple(v.cpu() for v in fn())
                record[name] = dict(outputs=outputs, recipes=observer.records)
                observer.records = None
        expected_serial = tuple(v[row:row + 1] for v in saved["reference"])
        expected_batch = saved["actual"]
        exact = dict(serial=all(torch.equal(a, z) for a, z in zip(record["serial"]["outputs"], expected_serial)),
                     batch=all(torch.equal(a, z) for a, z in zip(record["batch"]["outputs"], expected_batch)))
        torch.save(record, args.output.parent / f"rank{rank}-recipe-outputs.pt")
        args.output.write_text(
            json.dumps(dict(row=row,
                            uninstrumented_output_exact=exact,
                            recipes={
                                k: [dict(name=x["name"], shapes=x["shapes"]) for x in v["recipes"]]
                                for k, v in record.items()
                            }),
                       indent=2))
        if not all(exact.values()):
            raise RuntimeError("Recipe observation changed final outputs; diagnostic is invalid")
        if args.tp2:
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
            from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
            shutdown_prepared_group_plans()
            destroy_model_parallel()
            destroy_distributed_environment()
            context.__exit__(None, None, None)
        return

    report = dict(batch=b,
                  layers=args.layers,
                  layer_start=first,
                  scope=__doc__,
                  context_profile=args.context_profile,
                  positions=context_positions,
                  correctness_only=args.correctness_only,
                  native_replay=args.native_replay,
                  checks=[],
                  timings={})
    with torch.inference_mode():
        for change in range(2):
            if change:
                hidden.copy_(torch.randn(b, 4, 5120).bfloat16())
            restore()
            expected = tuple(v.cpu() for v in serial())
            if args.reference_directory is not None:
                archived = args.reference_directory / f"rank{rank}-output-{change}.pt"
                if archived.exists():
                    previous = torch.load(archived, weights_only=True)
                    assert torch.equal(hidden.cpu(), previous["input"]), "Reference input fixture changed"
                    assert all(torch.equal(a, z) for a, z in zip(expected, previous["reference"])), \
                        "Reused C1 reference differs from archived graph outputs"
            expected_state = [v.cpu() for v in buffers]
            restore()
            snap = BatchSnapshot(stage, positions, slots, pages)
            actual = tuple(v.cpu() for v in batch())
            equal_states = [
                torch.equal(v.cpu()[64:] if i >= len(buffers) - 2 else v.cpu(), r[64:] if i >= len(buffers) - 2 else r)
                for i, (v, r) in enumerate(zip(buffers, expected_state))
            ]
            check = dict(change=change,
                         output_exact=[torch.equal(x, y) for x, y in zip(expected, actual)],
                         state_exact=equal_states,
                         output_changed_elements=[int(torch.count_nonzero(x != y)) for x, y in zip(expected, actual)],
                         output_changed_rows=[
                             torch.nonzero((x != y).flatten(1).any(1)).flatten().tolist()
                             for x, y in zip(expected, actual)
                         ],
                         max_abs=[float((x.float() - y.float()).abs().max()) for x, y in zip(expected, actual)])
            report["checks"].append(check)
            args.output.write_text(json.dumps(report, indent=2))
            torch.save(dict(input=hidden.cpu(), reference=expected, actual=actual),
                       args.output.parent / f"rank{rank}-output-{change}.pt")
            if not all(check["output_exact"]) or not all(equal_states):
                raise RuntimeError(f"Four-layer request contract failed: {check}")
            snap.restore()
            assert all(torch.equal(v.cpu(), r.cpu()) for v, r in zip(buffers, original)), "Capture restore failed"
        if args.native_replay:
            from deepseek_v41_native_fragment import NativeBatchFragment
            candidate = NativeBatchFragment(grouped, candidate, buffers, restore,
                                            sum(2 for layer in layers if layer.attention.owns_index), pos_cpu.tolist())
            for _ in range(3):
                restore()
                batch()
                torch.hpu.synchronize()
            candidate.require_ready()
            restore()
            native_output = tuple(v.cpu() for v in batch())
            native_check = dict(outputs=[torch.equal(x, y) for x, y in zip(actual, native_output, strict=True)],
                                states=[
                                    torch.equal(v.cpu()[64:] if i >= len(buffers) - 2 else v.cpu(),
                                                r[64:] if i >= len(buffers) - 2 else r)
                                    for i, (v, r) in enumerate(zip(buffers, expected_state, strict=True))
                                ])
            report["native_check"] = native_check
            args.output.write_text(json.dumps(report, indent=2))
            if not all(native_check["outputs"] + native_check["states"]):
                raise RuntimeError(f"Native replay differs from the qualified ordinary candidate: {native_check}")
        for name, fn in (("serial", serial), ("batch", batch)):
            if args.correctness_only:
                continue
            if name == "serial" and not args.reference_timing:
                continue
            samples = []
            for _ in range(7 if args.native_replay else 5):
                torch.hpu.synchronize()
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start = time.perf_counter()
                begin.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(dict(device_ms=begin.elapsed_time(end), wall_ms=(time.perf_counter() - start) * 1000))
            report["timings"][name] = samples
            args.output.write_text(json.dumps(report, indent=2))
            print(name, statistics.median(v["device_ms"] for v in samples), flush=True)
    if args.native_replay:
        from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
        report["native_stats"] = prepared_group_stats()
        args.output.write_text(json.dumps(report, indent=2))
        candidate.close()
        candidate.metadata.native_completion = None
    if args.tp2:
        from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
        from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
        shutdown_prepared_group_plans()
        destroy_model_parallel()
        destroy_distributed_environment()
        context.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            import traceback
            import sys
            traceback.print_exc()
            sys.stderr.flush()
            # A failed rank must not enter device teardown while peers are
            # waiting on its collective. torchrun owns peer termination.
            os._exit(1)
        raise
