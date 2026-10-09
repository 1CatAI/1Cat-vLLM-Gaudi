# SPDX-License-Identifier: Apache-2.0
"""Check native C6 target replay on real decoder weights before full serving."""
import argparse
import copy
import json
import os
from pathlib import Path
from types import MethodType
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warm-steps", type=int, default=8)
    parser.add_argument("--audit-native-staging", action="store_true")
    parser.add_argument("--engram-upload-ab",
                        action="store_true",
                        help="Include changing packed Engram uploads and their real target consumer in both arms")
    parser.add_argument("--eager-input",
                        action="store_true",
                        help="Match the current DSpark eager embedding/residual/pre input producer")
    parser.add_argument("--index-mirror-ab",
                        action="store_true",
                        help="Target-only native C6 decoded index mirror A/B, ABABAB")
    parser.add_argument("--mla-vector-ab",
                        action="store_true",
                        help="Target-only logical vector A/B; index mirror enabled in both arms")
    parser.add_argument("--moe-sat-ab", action="store_true", help="SAT expert A/B on the qualified attention parent")
    parser.add_argument("--kv-norm-rope-ab",
                        action="store_true",
                        help="KV norm/RoPE A/B on the mirror/vector/SAT parent")
    parser.add_argument("--index-reduce-ab",
                        action="store_true",
                        help="Per-query head reduction A/B on the four qualified compute changes")
    parser.add_argument("--index-wide-ab", action="store_true",
                        help="Wide per-query scores A/B on the five qualified compute changes")
    parser.add_argument("--moe-prefetch-ab", action="store_true",
                        help="Early W2 SAT decode A/B on the six qualified compute changes")
    parser.add_argument("--mhc-batch-control-ab", action="store_true",
                        help="FP32 mHC weight reuse A/B on the six qualified compute changes")
    parser.add_argument("--mhc-mme-pair", action="store_true")
    parser.add_argument("--mhc-mme-ab", action="store_true",
                        help="BF16 operands/FP32 MME control versus current packed-RRMS C6 parent")
    parser.add_argument("--mhc-control-ab",
                        action="store_true",
                        help="Packed mHC control A/B; report internal errors and diagnostic head token IDs")
    parser.add_argument("--mhc-collapse-ab", action="store_true",
                        help="C6 post/collapse and interlayer handoff A/B on the six-change parent")
    parser.add_argument("--combined-compute-ab",
                        action="store_true",
                        help="Original device path vs mirror/vector/SAT together, sharing resident weights")
    parser.add_argument("--measure-combined-after",
                        action="store_true",
                        help="Measure the combined original/candidate after SAT A/B without reloading weights")
    parser.add_argument("--unique-expert-ab", action="store_true",
                        help="Device actual-M unique expert A/B on the current C6 parent")
    parser.add_argument("--threshold-selection-ab", action="store_true")
    parser.add_argument("--router-native-ab", action="store_true")
    parser.add_argument("--fp4-cache-rows-ab", action="store_true")
    parser.add_argument("--query-fusion-ab", action="store_true")
    parser.add_argument("--dense-restore-ab", action="store_true",
                        help="Common dense FP8 attention projections versus BF16 on the fixed C6 parent")
    parser.add_argument("--dense-sidecar", type=Path)
    parser.add_argument("--shared-restore-ab", action="store_true",
                        help="Common packed shared FP8 FFN on the fixed C6 parent")
    parser.add_argument("--trace-steps", type=int, default=0,
                        help="Capture candidate device activity after unprofiled A/B; exclude it from timings")
    args = parser.parse_args()
    mhc_numeric_ab = (args.mhc_control_ab or args.mhc_batch_control_ab or args.mhc_collapse_ab or args.mhc_mme_ab
                      or args.threshold_selection_ab or args.router_native_ab or args.fp4_cache_rows_ab
                      or args.query_fusion_ab or args.dense_restore_ab or args.shared_restore_ab)
    if (args.dense_restore_ab or args.shared_restore_ab) and args.dense_sidecar is None:
        parser.error("Dense restore A/B requires the existing C1 sidecar")
    parent_compute_ab = mhc_numeric_ab or args.moe_prefetch_ab or args.unique_expert_ab
    if args.mhc_batch_control_ab or args.moe_prefetch_ab or args.mhc_collapse_ab or args.mhc_mme_ab:
        variants = (args.index_mirror_ab, args.mla_vector_ab, args.moe_sat_ab, args.kv_norm_rope_ab,
                    args.index_reduce_ab, args.index_wide_ab, args.mhc_control_ab, args.mhc_batch_control_ab,
                    args.moe_prefetch_ab, args.combined_compute_ab, args.mhc_collapse_ab, args.mhc_mme_ab)
        if sum(variants) != 1 or args.measure_combined_after:
            parser.error("FP32 control and early W2 each compare one change on the fixed six-change parent")
    if args.trace_steps < 0:
        parser.error("Trace steps must be nonnegative")
    if args.measure_combined_after and not (args.moe_sat_ab or args.kv_norm_rope_ab or args.index_reduce_ab
                                            or parent_compute_ab or args.index_wide_ab):
        parser.error("The second measurement requires an incremental compute A/B")
    if args.layers % 4 or not 4 <= args.layers <= 40:
        parser.error("A target fixture must contain complete four-layer groups")
    if args.layer_start < 0 or args.layer_start % 4 or args.layer_start + args.layers > 40:
        parser.error("The target interval must contain aligned decoder groups within layers0..39")
    layer_ids = range(args.layer_start, args.layer_start + args.layers)
    rank, tp_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG",
                                                              "").replace("{rank}", str(rank))
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared,
                        tensor_parallel_size=tp_size,
                        pipeline_parallel_size=manifest["pipeline_parallel_size"])
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import (
        CompiledStage,
        PreparedDecoderLayer,
        PreparedInput,
        PreparedStage,
        _weight_tree,
        load_weight_tree,
    )
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, _Snapshot, stage_collectives, stage_state_tensors
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_verify import vocab_parallel_argmax
    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp_size))
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="running",
                  tp=tp_size,
                  rank=rank,
                  layers=list(layer_ids),
                  rows=6,
                  target_context=16384,
                  capacity=1048576,
                  engram="fixed zero BF16 rows; no host lookup",
                  endpoint="real target -> vocab-parallel head/argmax -> next target embedding",
                  semantic_qualification=False,
                  formal_target_met=False)

    def save():
        (root / f"target-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    with set_current_vllm_config(config), torch.inference_mode():
        try:
            torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
            init_distributed_environment(world_size=tp_size,
                                         rank=rank,
                                         distributed_init_method="env://",
                                         local_rank=rank,
                                         backend="hccl")
            initialize_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            if os.environ.get("GRAPH_VISUALIZATION") == "1":
                from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

                configure_post_graph_directory(root / "graphs" / f"rank{rank}")
            reduce, gather = stage_collectives(rank, True, tp_size)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            prefixes = tuple(f"layers.{index}." for index in layer_ids)
            specs = {
                name: spec
                for name, spec in shard.specs.items()
                if name in ("norm.weight", "embed.weight", "head.weight") or name.startswith(prefixes)
            }
            tree = _weight_tree(specs)
            dense_sidecar = None
            dense_config = None
            if args.query_fusion_ab:
                from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar

                dense_sidecar = DenseFP8Sidecar(args.prepared / "sidecars/attention_dense_fp8", shard)
                dense_config = {"wq_b": list(layer_ids)}
            load_weight_tree(shard, tree, "hpu", specs, dense_sidecar=dense_sidecar, dense_config=dense_config)
            stage = torch.nn.Module()
            stage.weights, stage.config, stage.shard = tree, {"text_config": text}, shard
            stage.length, stage.search_length, stage.generation = 1048576, 32768, 1
            stage.pp_rank, stage.tp_rank, stage.tensor_parallel_size = 0, rank, tp_size
            stage.dspark, stage.is_last_stage = True, True
            stage.fp8_decode, stage.expert_n256, stage.bf16_head = True, True, False
            stage.precision_fingerprint = ("dspark", "n256-fp8", "six-target-rows")
            stage.reduce, stage.all_gather = reduce, gather
            stage.shared = PagedCSA2SharedState(text,
                                                args.layer_start,
                                                args.layer_start + args.layers,
                                                "hpu",
                                                stage.length,
                                                tensor_parallel_size=tp_size)
            stage.runtime_indexer = stage.shared.runtime_indexer
            stage.shared.block_table[:256].copy_(torch.arange(1, 257, dtype=torch.int32, device="hpu"))
            for cache in stage.shared.sources.values():
                cache.main = torch.zeros(257 * 128 // cache.ratio, 288, dtype=torch.uint8, device="hpu")
                cache.index = torch.zeros(257 * 128 // cache.ratio, 68, dtype=torch.uint8, device="hpu")
            lookup = mxfp4_bf16_lut(torch.device("hpu"))
            stage.layers = torch.nn.ModuleList()
            dense_variants = []
            shared_variants = []
            restore_sidecar = None
            if args.dense_restore_ab or args.shared_restore_ab:
                from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
                restore_sidecar = DenseFP8Sidecar(args.dense_sidecar, shard)
            for index in layer_ids:
                block = PreparedDecoderLayer(tree.layers.get_submodule(str(index)),
                                             text,
                                             index,
                                             stage.shared,
                                             shard.manifest["normal_scales"][f"layers.{index}.ffn.experts"][rank],
                                             lookup,
                                             reduce,
                                             gather,
                                             "hpu",
                                             tensor_parallel_size=tp_size)
                if args.query_fusion_ab:
                    block.attention.native_rope = True
                    block.attention.q_scale_rope = True
                    block.attention.fused_norm = True
                    block.attention._rotary_native_name = f"{block.attention._rotary_name}_native"
                block.attention.set_search_length(stage.search_length)
                block.attention.prepare_qkv_input_weight()
                block.attention.prepare_compressor_input_weight()
                block.attention.prefill_tp_rank = rank
                if block.attention.prepared_output:
                    block.attention.prepare_output_weight()
                block.prepare_mhc_control_weights()
                if args.dense_restore_ab:
                    original = block.attention
                    proposed = copy.copy(original)
                    proposed._modules = dict(original._modules)
                    proposed._buffers = dict(original._buffers)
                    weights = copy.copy(original.weights)
                    weights._modules = dict(original.weights._modules)
                    weights._buffers = dict(original.weights._buffers)
                    for projection in ("wq_a", "wkv", "wq_b", "wo_b"):
                        module = copy.copy(getattr(original.weights, projection))
                        module._buffers = dict(module._buffers)
                        prefix = f"layers.{index}.attn.{projection}."
                        module.weight = restore_sidecar.tensor(prefix + "weight", "hpu")
                        channel = restore_sidecar.tensor(prefix + "channel_scale", "hpu")
                        module.register_buffer("channel_scale", channel, False)
                        module.dense_fp8 = True
                        module.dense_fp8_direct_input = projection in ("wq_a", "wkv")
                        weights.add_module(projection, module)
                    proposed.weights = weights
                    proposed.invalidate_qkv_input_weight()
                    proposed.prepare_qkv_input_weight()
                    dense_variants.append((original, proposed))
                if args.shared_restore_ab:
                    original = block.moe
                    proposed = copy.copy(original)
                    proposed._modules = dict(original._modules)
                    proposed._buffers = dict(original._buffers)
                    weights = copy.copy(original.weights)
                    weights._modules = dict(original.weights._modules)
                    weights._buffers = dict(original.weights._buffers)
                    shared = copy.copy(original.weights.shared_experts)
                    shared._modules = dict(original.weights.shared_experts._modules)
                    shared._buffers = dict(original.weights.shared_experts._buffers)
                    for projection in ("w1", "w3", "w2"):
                        module = copy.copy(getattr(original.weights.shared_experts, projection))
                        module._buffers = dict(module._buffers)
                        prefix = f"layers.{index}.ffn.shared_experts.{projection}."
                        module.weight = restore_sidecar.tensor(prefix + "weight", "hpu")
                        module.register_buffer("channel_scale", restore_sidecar.tensor(prefix + "channel_scale", "hpu"),
                                               False)
                        module.dense_fp8 = module.dense_fp8_direct_input = True
                        shared.add_module(projection, module)
                    weights.shared_experts = shared
                    proposed.weights = weights
                    proposed.shared_gate_up = True
                    proposed.prepare_shared_gate_up_weight()
                    shared_variants.append((original, proposed))
                stage.layers.append(block)
            for name in ("_forward_impl", "forward"):
                setattr(stage, name, MethodType(getattr(PreparedStage, name), stage))
            input_module = PreparedInput(tree.embed, rank, reduce)
            embedding = input_module if args.eager_input else torch.compile(
                input_module, backend="hpu_backend", fullgraph=True, dynamic=False)
            report["input_producer"] = "eager" if args.eager_input else "compiled"

            def sample(hidden):
                return vocab_parallel_argmax(torch.nn.functional.linear(hidden.float(), tree.head.weight), rank, gather)

            sampler = torch.compile(sample, backend="hpu_backend", fullgraph=True, dynamic=False)
            reference, native = CompiledStage(stage, native=True), StageReplay(stage)
            engram = tuple(
                torch.zeros(6, 24 // tp_size, 256, dtype=torch.bfloat16, device="hpu") for _ in (1, 14)
                if _ in layer_ids)
            ids = torch.arange(6, dtype=torch.int64, device="hpu") + 7
            positions = torch.arange(16384, 16390, dtype=torch.int32, device="hpu")
            initial = _Snapshot(stage_state_tensors(stage))

            def reset():
                torch.hpu.synchronize()
                initial.restore()
                ids.copy_(torch.arange(6, dtype=torch.int64, device="hpu") + 7)
                positions.copy_(torch.arange(16384, 16390, dtype=torch.int32, device="hpu"))
                torch.hpu.synchronize()

            unique_inputs = []
            if args.fp4_cache_rows_ab:
                from vllm_gaudi.ops.deepseek_v41_unique_experts import load_unique_expert_operators

                load_unique_expert_operators()
            if args.mhc_mme_ab:
                from vllm_gaudi.ops.deepseek_v41_unique_experts import load_unique_expert_operators

                load_unique_expert_operators()
                for layer in stage.layers:
                    for name in ("hc_attn_fn", "hc_ffn_fn"):
                        packed = layer._pack_mhc_control_weight(getattr(layer.weights, name))
                        high = packed.bfloat16()
                        if args.mhc_mme_pair:
                            low = (packed - high.float()).bfloat16()
                            high = torch.cat((high, low), dim=0).contiguous()
                        setattr(layer, name + "_mme", high)
            if args.unique_expert_ab:
                from vllm_gaudi.ops.deepseek_v41_unique_experts import UniqueExpertInputs, load_unique_expert_operators
                from vllm_gaudi.ops.tp2_prepared_plan import _runtime
                bridge, _ = _runtime()
                if not hasattr(bridge.NativeDecodeGraph, "set_compact_inputs"):
                    raise RuntimeError("Unique expert A/B requires its private row-program bridge")
                load_unique_expert_operators()
                unique_inputs = [UniqueExpertInputs("hpu") for _ in stage.layers]

            def step(execute):
                residual, pre = embedding(ids)
                hidden = execute(residual, pre, positions, ids, engram)[0]
                selected = sampler(hidden)
                ids.copy_(selected)
                positions.add_(6)
                return hidden, selected

            if (args.index_mirror_ab or args.mla_vector_ab or args.moe_sat_ab or args.kv_norm_rope_ab
                    or args.index_reduce_ab or parent_compute_ab or args.index_wide_ab or args.combined_compute_ab):
                report.update(endpoint="Target C6 only; input preparation and sampling excluded",
                              input_producer="prepared before each device interval",
                              index_mirror_ab=dict(blocks=[]))
                report["candidate"] = ("c6_dense_shared_restore" if args.dense_restore_ab
                                       and args.shared_restore_ab else
                                       "c6_shared_restore" if args.shared_restore_ab else
                                       "c6_dense_restore" if args.dense_restore_ab else
                                       "c6_query_fusion" if args.query_fusion_ab else
                                       "c6_fp4_cache_rows" if args.fp4_cache_rows_ab else
                                       "c6_threshold_selection" if args.threshold_selection_ab else
                                       "c6_native_router" if args.router_native_ab else
                                       "mhc_bf16_mme_control" if args.mhc_mme_ab else
                                       "unique_experts_actual_m" if args.unique_expert_ab else
                                       "mhc_c6_post_collapse" if args.mhc_collapse_ab else
                                       "moe_sat_prefetch_w2" if args.moe_prefetch_ab else
                                       "mhc_batch_control_f32" if args.mhc_batch_control_ab else
                                       "combined_mirror_vector_sat" if args.combined_compute_ab else "mhc_control_rrms"
                                       if args.mhc_control_ab else "c6_index_wide_scores" if args.index_wide_ab else
                                       "c6_index_reduce" if args.index_reduce_ab else
                                       "kv_norm_rope" if args.kv_norm_rope_ab else "moe_token_wide_sat" if args.
                                       moe_sat_ab else "logical_mla_vector" if args.mla_vector_ab else "index_mirror")
                variant_ab = (args.mla_vector_ab or args.moe_sat_ab or args.kv_norm_rope_ab or args.index_reduce_ab
                              or parent_compute_ab or args.index_wide_ab or args.combined_compute_ab)
                vector_replays = [StageReplay(stage), StageReplay(stage)] if variant_ab else None
                qualified_sat = [bool(getattr(layer.moe, "c6_token_wide_sat", False)) for layer in stage.layers]
                qualified_mhc = [(layer.hc_attn_fn_packed, layer.hc_ffn_fn_packed) for layer in stage.layers]
                qualified_collapse = [(layer.mhc_post_collapse, layer.mhc_interlayer_collapse)
                                      for layer in stage.layers]
                if args.mhc_control_ab and any(a is None or b is None for a, b in qualified_mhc):
                    raise RuntimeError("mHC A/B requires prepared checkpoint control weights")
                combined_phase = args.combined_compute_ab
                report["sat_eligible_layers"] = dict(zip(map(str, layer_ids), qualified_sat, strict=True))
                save()

                def prepare_arm(use_mirror):
                    reset()
                    if args.dense_restore_ab:
                        for layer, variants in zip(stage.layers, dense_variants, strict=True):
                            layer.attention = variants[int(use_mirror)]
                        stage.precision_fingerprint = ("dspark", "n256-fp8", "six-target-rows",
                                                       "dense_fp8" if use_mirror else "dense_bf16")
                    if args.shared_restore_ab:
                        for layer, variants in zip(stage.layers, shared_variants, strict=True):
                            layer.moe = variants[int(use_mirror)]
                        stage.precision_fingerprint = (
                            "dspark", "n256-fp8", "six-target-rows",
                            "dense_fp8" if args.dense_restore_ab and use_mirror else "dense_bf16",
                            "shared_fp8" if use_mirror else "shared_bf16")
                    stage.shared.invalidate_index_mirror()
                    if use_mirror or (not combined_phase and
                                      (args.mla_vector_ab or args.moe_sat_ab or args.kv_norm_rope_ab
                                       or args.index_reduce_ab or parent_compute_ab or args.index_wide_ab)):
                        stage.shared.prepare_index_mirror(16384)
                    use_parent_compute = (use_mirror if combined_phase else
                                          parent_compute_ab or args.index_wide_ab or use_mirror)
                    if args.mla_vector_ab or combined_phase or parent_compute_ab or args.index_wide_ab:
                        for layer in stage.layers:
                            layer.attention.logical_mla_vector = use_parent_compute
                    if args.mla_vector_ab or args.index_mirror_ab:
                        for layer in stage.layers:
                            layer.moe.c6_token_wide_sat = False
                    if args.moe_sat_ab or combined_phase or parent_compute_ab or args.index_wide_ab:
                        for layer, eligible in zip(stage.layers, qualified_sat, strict=True):
                            layer.moe.c6_token_wide_sat = eligible and use_parent_compute
                    for layer in stage.layers:
                        layer.attention.kv_norm_rope_decode = args.kv_norm_rope_ab and use_mirror
                        layer.attention.c6_index_reduce = args.index_reduce_ab and use_mirror
                        if args.index_reduce_ab:
                            layer.attention.kv_norm_rope_decode = use_mirror if combined_phase else True
                    for layer, packed in zip(stage.layers, qualified_mhc, strict=True):
                        layer.hc_attn_fn_packed, layer.hc_ffn_fn_packed = (
                            packed if args.mhc_batch_control_ab or args.query_fusion_ab or args.fp4_cache_rows_ab
                            or args.dense_restore_ab or args.shared_restore_ab
                            or args.threshold_selection_ab
                            or args.router_native_ab
                            or args.unique_expert_ab
                            or args.mhc_mme_ab or args.mhc_collapse_ab or
                            (args.mhc_control_ab and use_mirror) else (None, None))
                        layer.attention.c6_index_wide_scores = (
                            (args.index_wide_ab and use_mirror) or args.mhc_batch_control_ab
                            or args.moe_prefetch_ab or args.mhc_collapse_ab or args.unique_expert_ab or args.mhc_mme_ab
                            or args.threshold_selection_ab or args.router_native_ab or args.fp4_cache_rows_ab
                            or args.query_fusion_ab or args.dense_restore_ab or args.shared_restore_ab
                        )
                        layer.batch_control_reuse = (args.mhc_batch_control_ab and use_mirror or
                                                     (args.dense_restore_ab or args.shared_restore_ab) and
                                                     os.environ.get("VLLM_HPU_DSV41_MHC_BATCH_REUSE") == "1")
                        layer.mhc_control_mme = args.mhc_mme_ab and use_mirror
                        layer.attention.decode_threshold_selection = args.threshold_selection_ab and use_mirror
                        layer.attention.c6_fp4_cache_write = args.fp4_cache_rows_ab and use_mirror
                        layer.attention.c6_query_fusion = args.query_fusion_ab and use_mirror
                        if args.router_native_ab:
                            layer.moe.router_top6 = use_mirror
                        layer.moe.c6_prefetch_w2 = args.moe_prefetch_ab and use_mirror
                        if parent_compute_ab or args.index_wide_ab:
                            layer.attention.kv_norm_rope_decode = use_mirror if combined_phase else True
                            layer.attention.c6_index_reduce = use_mirror if combined_phase else True
                    if args.mhc_collapse_ab:
                        for layer, eligible in zip(stage.layers, qualified_collapse, strict=True):
                            layer.mhc_post_collapse = eligible[0] and use_mirror
                            layer.mhc_interlayer_collapse = eligible[1] and use_mirror
                    if args.unique_expert_ab:
                        for layer, inputs in zip(stage.layers, unique_inputs, strict=True):
                            layer.moe.unique_expert_inputs = inputs if use_mirror else None
                        stage.compact_pipeline_inputs = tuple(
                            tensor for inputs in unique_inputs for tensor in inputs.tensors()) if use_mirror else ()
                    torch.hpu.synchronize()

                def invoke(use_candidate, *inputs):
                    execute = vector_replays[int(use_candidate)] if vector_replays is not None else native
                    return execute(*inputs)

                def prepare_step(index):
                    ids.copy_(torch.arange(6, dtype=torch.int64, device="hpu") + 7 + index % 16)
                    positions.copy_(torch.arange(16384, 16390, dtype=torch.int32, device="hpu") + index * 6)
                    inputs = embedding(ids)
                    torch.hpu.synchronize()
                    return inputs

                base_candidate = report["candidate"]
                report["measurements"] = {}
                phases = (False, True) if args.measure_combined_after else (combined_phase, )
                for combined_phase in phases:
                    report["candidate"] = (
                        "combined_mirror_vector_sat_kv_norm_rope_index_reduce_wide"
                        if combined_phase and args.index_wide_ab else
                        "combined_mirror_vector_sat_kv_norm_rope_index_reduce_mhc" if combined_phase
                        and args.mhc_control_ab else "combined_mirror_vector_sat_kv_norm_rope_index_reduce"
                        if combined_phase and args.index_reduce_ab else
                        "combined_mirror_vector_sat_kv_norm_rope" if combined_phase and args.kv_norm_rope_ab else
                        "combined_mirror_vector_sat" if combined_phase else base_candidate)
                    report["index_mirror_ab"] = dict(blocks=[])
                    report["measurements"][report["candidate"]] = report["index_mirror_ab"]
                    outputs = []
                    for use_mirror in (False, True):
                        prepare_arm(use_mirror)
                        values = []
                        for index in range(4):
                            residual, pre = prepare_step(index)
                            hidden, final_pre, aux = invoke(use_mirror, residual, pre, positions, ids, engram)
                            selected = sampler(hidden).cpu() if mhc_numeric_ab else None
                            values.append((hidden.cpu(), final_pre.cpu(), selected)
                                          if mhc_numeric_ab else (hidden.cpu(), final_pre.cpu()))
                        outputs.append(values)
                    errors = []
                    byte_exact = True
                    for old, new in zip(*outputs, strict=True):
                        exact = all(torch.equal(lhs.contiguous().view(torch.uint8),
                                                rhs.contiguous().view(torch.uint8))
                                    for lhs, rhs in zip(old[:2], new[:2], strict=True))
                        byte_exact = byte_exact and exact
                        if args.fp4_cache_rows_ab and not exact:
                            raise AssertionError("FP4 cache writer changed real16 hidden/pre bytes")
                        if mhc_numeric_ab:
                            step_error = {}
                            for name, lhs, rhs in zip(("hidden", "pre"), old[:2], new[:2], strict=True):
                                difference = (lhs.float() - rhs.float()).abs()
                                if not torch.isfinite(rhs).all():
                                    raise AssertionError("mHC candidate produced non-finite target output")
                                step_error[name] = dict(max_abs=float(difference.max()),
                                                        mean_abs=float(difference.mean()),
                                                        rms=float(difference.square().mean().sqrt()),
                                                        changed_elements=int((lhs != rhs).sum()))
                            step_error["head_tokens_reference"] = old[2].tolist()
                            step_error["head_tokens_candidate"] = new[2].tolist()
                            step_error["head_tokens_exact"] = bool(torch.equal(old[2], new[2]))
                            errors.append(step_error)
                        else:
                            for lhs, rhs in zip(old, new, strict=True):
                                torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
                                if not torch.equal(lhs.contiguous().view(torch.uint8),
                                                   rhs.contiguous().view(torch.uint8)):
                                    raise AssertionError("Target C6 consumer differs at the byte level")
                    report["index_mirror_ab"]["four_changing_steps_exact"] = byte_exact
                    report["index_mirror_ab"]["byte_exact"] = byte_exact
                    if mhc_numeric_ab:
                        report["index_mirror_ab"]["internal_errors"] = errors
                        report["index_mirror_ab"]["diagnostic_head_tokens_exact"] = all(value["head_tokens_exact"]
                                                                                        for value in errors)
                        report["index_mirror_ab"]["generated_tokens_qualified"] = False
                        report["index_mirror_ab"]["token_scope"] = (
                            "Head after a partial layer interval with synthetic history; "
                            "formal EOS tokens still required"
                        )
                    for use_mirror in (False, True, False, True, False, True):
                        prepare_arm(use_mirror)
                        for index in range(args.warm_steps):
                            residual, pre = prepare_step(index)
                            invoke(use_mirror, residual, pre, positions, ids, engram)
                        torch.hpu.synchronize()
                        before = prepared_group_stats()
                        samples = []
                        for index in range(args.steps):
                            residual, pre = prepare_step(index)
                            start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                            start.record()
                            started = time.perf_counter_ns()
                            invoke(use_mirror, residual, pre, positions, ids, engram)
                            stop.record()
                            stop.synchronize()
                            samples.append(
                                dict(device_ms=start.elapsed_time(stop),
                                     drained_host_ms=(time.perf_counter_ns() - started) / 1e6))
                        report["index_mirror_ab"]["blocks"].append(
                            dict(mirror=use_mirror, samples=samples, before=before, after=prepared_group_stats()))
                        save()
                if args.trace_steps:
                    from vllm_gaudi.ops.deepseek_v41_native_trace import NativeTrace, scope

                    prepare_arm(True)
                    trace_inputs = prepare_step(0)
                    invoke(True, *trace_inputs, positions, ids, engram)
                    torch.hpu.synchronize()
                    trace = NativeTrace(cpu_trace_dir=root / "traces" / f"rank{rank}", scope_only=True)
                    trace.start()
                    try:
                        for arm in ((False, True) if args.unique_expert_ab or args.mhc_mme_ab
                                    or args.threshold_selection_ab or args.router_native_ab
                                    or args.fp4_cache_rows_ab or args.query_fusion_ab or args.dense_restore_ab
                                    or args.shared_restore_ab
                                    or args.mhc_batch_control_ab else (True,)):
                            prepare_arm(arm)
                            for index in range(args.trace_steps):
                                residual, pre = prepare_step(index)
                                variant = ("dense_shared_restore" if args.dense_restore_ab
                                           and args.shared_restore_ab else
                                           "shared_restore" if args.shared_restore_ab else
                                           "dense_restore" if args.dense_restore_ab else
                                           "batch_control" if args.mhc_batch_control_ab else
                                           "query" if args.query_fusion_ab else
                                           "cache_write" if args.fp4_cache_rows_ab else
                                           "selection" if args.threshold_selection_ab else
                                           "router" if args.router_native_ab else
                                           "mme" if args.mhc_mme_ab else "unique")
                                arm_name = variant if arm else "reference"
                                with scope(f"v41::target_c6::{arm_name}::layers{args.layer_start}:"
                                           f"{args.layer_start + args.layers}::rank{rank}::step{index}"):
                                    invoke(arm, residual, pre, positions, ids, engram)
                                torch.hpu.synchronize()
                    finally:
                        trace.stop()
                    report["trace"] = dict(steps=args.trace_steps, metadata=trace.metadata,
                                           included_in_ab_timing=False, history="synthetic zero prefix")
                report.update(status="passed_target_device_ab",
                              timed_steps=args.steps,
                              warm_steps=args.warm_steps,
                              peak_device_bytes=torch.hpu.max_memory_allocated())
                save()
                return

            if args.engram_upload_ab:
                from vllm_gaudi.ops.deepseek_v41_host import _TransferSlot
                if args.layer_start != 0 or args.layers < 16:
                    parser.error("Engram upload A/B requires both real consumers in layers0..15")
                slots = [[_TransferSlot(6, 24 // tp_size, 256, "hpu") for _ in range(2)] for _ in engram]
                transfer_stream = torch.hpu.Stream()

                def upload_step(index, separate_stream):
                    packet_slots = [ring[index % 2] for ring in slots]
                    for layer_index, slot in enumerate(packet_slots):
                        slot.reuse()
                        slot.host[:, :, :256].fill_((index + layer_index) % 7)
                        slot.host[:, :, 256:].fill_(120)
                    residual, pre = embedding(ids)
                    before_upload = time.perf_counter_ns()
                    for slot in packet_slots:
                        slot.upload(6, transfer_stream if separate_stream else None)
                    after_upload = time.perf_counter_ns()
                    hidden = native(residual, pre, positions, ids, tuple(s.device for s in packet_slots))[0]
                    selected = sampler(hidden)
                    ids.copy_(selected)
                    positions.add_(6)
                    embedding(ids)
                    for slot in packet_slots:
                        slot.consumer_done.record(torch.hpu.current_stream())
                        slot.inflight = True
                    return hidden, selected, (after_upload - before_upload) / 1e6

                reset()
                expected_uploads = [tuple(v.cpu() for v in upload_step(i, True)[:2]) for i in range(4)]
                reset()
                actual_uploads = [tuple(v.cpu() for v in upload_step(i, False)[:2]) for i in range(4)]
                for old, new in zip(expected_uploads, actual_uploads, strict=True):
                    for lhs, rhs in zip(old, new, strict=True):
                        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
                report["engram_upload_ab"] = dict(four_feedback_steps_exact=True, arms={})
                for separate_stream in (True, False):
                    reset()
                    for i in range(args.warm_steps):
                        upload_step(i, separate_stream)
                    torch.hpu.synchronize()
                    samples = []
                    for i in range(args.steps):
                        start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        start.record()
                        started = time.perf_counter_ns()
                        _, _, upload_ms = upload_step(i, separate_stream)
                        stop.record()
                        stop.synchronize()
                        samples.append(
                            dict(host_ms=(time.perf_counter_ns() - started) / 1e6,
                                 device_ms=start.elapsed_time(stop),
                                 upload_host_ms=upload_ms))
                    name = "transfer_stream" if separate_stream else "compute_stream"
                    report["engram_upload_ab"]["arms"][name] = samples
                    save()
                report.update(status="passed_engram_upload_chain",
                              timed_steps=args.steps,
                              warm_steps=args.warm_steps,
                              after=prepared_group_stats(),
                              peak_device_bytes=torch.hpu.max_memory_allocated())
                save()
                return

            save()
            reset()
            expected = [tuple(value.cpu() for value in step(reference)) for _ in range(2)]
            reset()
            actual = [tuple(value.cpu() for value in step(native)) for _ in range(2)]
            for old, new in zip(expected, actual, strict=True):
                for lhs, rhs in zip(old, new, strict=True):
                    torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
            report["two_feedback_steps_exact"] = True
            report["native_prepared_before_timing"] = prepared_group_stats()
            reset()
            for _ in range(args.warm_steps):
                step(native)
            torch.hpu.synchronize()
            before = prepared_group_stats()
            staging_before = None
            if args.audit_native_staging:
                from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
                graphs = [entry[0] for entry in _native_entries.values()]
                staging_before = dict(copies=sum(graph.input_update_copies() for graph in graphs),
                                      bytes=sum(graph.input_update_bytes() for graph in graphs))
            start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            start.record()
            clock_start = time.perf_counter_ns()
            for _ in range(args.steps):
                step(native)
            embedding(ids)
            stop.record()
            stop.synchronize()
            report.update(status="passed_target_chain",
                          device_ms=start.elapsed_time(stop) / args.steps,
                          host_ms=(time.perf_counter_ns() - clock_start) / 1e6 / args.steps,
                          warm_steps=args.warm_steps,
                          timed_steps=args.steps,
                          before=before,
                          after=prepared_group_stats(),
                          peak_device_bytes=torch.hpu.max_memory_allocated())
            if args.audit_native_staging:
                staging_after = dict(copies=sum(graph.input_update_copies() for graph in graphs),
                                     bytes=sum(graph.input_update_bytes() for graph in graphs))
                report["native_input_staging"] = dict(
                    before=staging_before,
                    after=staging_after,
                    copies_per_step=(staging_after["copies"] - staging_before["copies"]) / args.steps,
                    bytes_per_step=(staging_after["bytes"] - staging_before["bytes"]) / args.steps)
            save()
        except Exception as exc:
            report.update(status="failed", error=repr(exc))
            save()
            raise
        finally:
            if torch.distributed.is_initialized():
                shutdown_prepared_group_plans()
                destroy_model_parallel()
                destroy_distributed_environment()


if __name__ == "__main__":
    main()
