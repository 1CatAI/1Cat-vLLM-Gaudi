# SPDX-License-Identifier: Apache-2.0
"""Additive operator registration against the locked serving Bridge headers."""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir


def directory(name):
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        raise RuntimeError(f"{name} must identify the matching Bridge headers")
    return Path(value).resolve()


bridge = directory("GAUDI_PYTORCH_BRIDGE_ROOT")
build = directory("GAUDI_PYTORCH_BRIDGE_BUILD_ROOT")
deps = directory("GAUDI_PYTORCH_BRIDGE_DEPS_ROOT")
private_hw_bias = os.environ.get("DSV41_HW_DIRECT_BIAS", "0") == "1"
includes = [get_include_dir(), "/usr/include/habanalabs", "/usr/include/habanalabs/hl_logger", str(bridge),
            str(bridge / "pytorch_helpers"), str(bridge / "python_packages/habana_frameworks/torch/jit/csrc"),
            str(bridge / "pytorch_helpers/habana_helpers/habana_serialization/include"), str(build), str(deps)]
includes += [str(deps / name) for name in (
    "abseil-cpp-src", "magic_enum-src/include", "fmt-src/include", "exprtk-src/include", "nlohmann_json-src/include")]
sources = ["register.cpp", "logical_union.cpp", "logical_decoded.cpp",
           "fp4_cache_rows.cpp", "logical_write_ordered.cpp", "index_write_ordered.cpp",
           "logical_slotmap.cpp", "mla_joined_gemm.cpp", "q_projection_tiled.cpp", "sat_prefetch16.cpp",
           "sat_active_k.cpp", "logical_pair_pv.cpp", "expert_decode_probe.cpp", "sat_k_first.cpp",
           "sat_small_sram.cpp", "sat_bytes.cpp", "sat_silu640.cpp", "vocab_radix.cpp",
           "logical_cohesive.cpp", "logical_merged.cpp", "vocab_partition.cpp", "sat_feature_tiles.cpp",
           "sampling_wire_view.cpp", "vocab_filter.cpp", "sat_three_route_w2.cpp", "peer_post_norm.cpp",
           "sat_silu_unroll.cpp", "kv_publish.cpp", "logical_pv_rope.cpp", "nucleus_sampler.cpp",
           "mtp_k128.cpp"]
# The serving mHC bundle already owns this schema. A private additive build
# must not register it again; older diagnostic bundles still include it.
only_attention_interleave = os.environ.get("DSV41_UNIQUE_ONLY_ATTENTION_INTERLEAVE", "0") == "1"
only_fp8_prologue = os.environ.get("DSV41_UNIQUE_ONLY_FP8_PROLOGUE", "0") == "1"
only_source_dense = os.environ.get("DSV41_UNIQUE_ONLY_SOURCE_DENSE", "0") == "1"
only_vocab_softmax = os.environ.get("DSV41_UNIQUE_ONLY_VOCAB_SOFTMAX", "0") == "1"
only_peer_signal_probe = os.environ.get("DSV41_UNIQUE_ONLY_PEER_SIGNAL_PROBE", "0") == "1"
only_peer_post_collapse = os.environ.get("DSV41_UNIQUE_ONLY_PEER_POST_COLLAPSE", "0") == "1"
only_ordered_peer_sum = os.environ.get("DSV41_UNIQUE_ONLY_ORDERED_PEER_SUM", "0") == "1"
only_journal_batch_direct = os.environ.get("DSV41_UNIQUE_ONLY_JOURNAL_BATCH_DIRECT", "0") == "1"
only_journal_batch = os.environ.get("DSV41_UNIQUE_ONLY_JOURNAL_BATCH", "0") == "1"
only_stream_exp = os.environ.get("DSV41_UNIQUE_ONLY_STREAM_EXP", "0") == "1"
only_qk_flat_direct = os.environ.get("DSV41_UNIQUE_ONLY_QK_FLAT_DIRECT", "0") == "1"
only_qk_flat = os.environ.get("DSV41_UNIQUE_ONLY_QK_FLAT", "0") == "1"
only_slicer_stitch = os.environ.get("DSV41_UNIQUE_ONLY_SLICER_STITCH", "0") == "1"
only_mhc_gate_packet = os.environ.get("DSV41_UNIQUE_ONLY_MHC_GATE_PACKET", "0") == "1"
only_w2_k_pipeline = os.environ.get("DSV41_UNIQUE_ONLY_W2_K_PIPELINE", "0") == "1"
only_w13_k_pipeline = os.environ.get("DSV41_UNIQUE_ONLY_W13_K_PIPELINE", "0") == "1"
only_router_batched = os.environ.get("DSV41_UNIQUE_ONLY_ROUTER_BATCHED", "0") == "1"
only_w13_boundary = os.environ.get("DSV41_UNIQUE_ONLY_W13_BOUNDARY", "0") == "1"
only_dense_bf16_pair = os.environ.get("DSV41_UNIQUE_ONLY_DENSE_BF16_PAIR", "0") == "1"
only_cooperative_silu = os.environ.get("DSV41_UNIQUE_ONLY_COOPERATIVE_SILU", "0") == "1"
only_w2_channels = os.environ.get("DSV41_UNIQUE_ONLY_W2_CHANNELS", "0") == "1"
only_w2_reduce_n256 = os.environ.get("DSV41_UNIQUE_ONLY_W2_REDUCE_N256", "0") == "1"
only_w2_shared_scale_n256 = os.environ.get("DSV41_UNIQUE_ONLY_W2_SHARED_SCALE_N256", "0") == "1"
only_w2_ready_scale = os.environ.get("DSV41_UNIQUE_ONLY_W2_READY_SCALE", "0") == "1"
only_w13_k_pipeline_four = os.environ.get("DSV41_UNIQUE_ONLY_W13_K_PIPELINE_FOUR", "0") == "1"
only_w13_k_pipeline_k512 = os.environ.get("DSV41_UNIQUE_ONLY_W13_K_PIPELINE_K512", "0") == "1"
only_split_scale_planes = os.environ.get("DSV41_UNIQUE_ONLY_SPLIT_SCALE_PLANES", "0") == "1"
only_silu_decode_affine = os.environ.get("DSV41_UNIQUE_ONLY_SILU_DECODE_AFFINE", "0") == "1"
only_mhc_post_stats = os.environ.get("DSV41_UNIQUE_ONLY_MHC_POST_STATS", "0") == "1"
only_main_single_bank = os.environ.get("DSV41_UNIQUE_ONLY_MAIN_SINGLE_BANK", "0") == "1"
only_coherent_swa = os.environ.get("DSV41_UNIQUE_ONLY_COHERENT_SWA", "0") == "1"
only_input_quant_remat = os.environ.get("DSV41_UNIQUE_ONLY_INPUT_QUANT_REMAT", "0") == "1"
only_moe_peer_post = os.environ.get("DSV41_UNIQUE_ONLY_MOE_PEER_POST", "0") == "1"
only_mixed_dense_probe = os.environ.get("DSV41_UNIQUE_ONLY_MIXED_DENSE_PROBE", "0") == "1"
only_scaled_w13 = os.environ.get("DSV41_UNIQUE_ONLY_SCALED_W13", "0") == "1"
only_control_fp8 = os.environ.get("DSV41_UNIQUE_ONLY_CONTROL_FP8", "0") == "1"
only_control_fp8_pair = os.environ.get("DSV41_UNIQUE_ONLY_CONTROL_FP8_PAIR", "0") == "1"
only_explicit_steps = os.environ.get("DSV41_UNIQUE_ONLY_EXPLICIT_STEPS", "0") == "1"
only_mhc_deferred = os.environ.get("DSV41_UNIQUE_ONLY_MHC_DEFERRED", "0") == "1"
only_silu_decode = os.environ.get("DSV41_UNIQUE_ONLY_SILU_DECODE", "0") == "1"
only_norm_roundtrip = os.environ.get("DSV41_UNIQUE_ONLY_NORM_ROUNDTRIP", "0") == "1"
only_probability_draw = os.environ.get("DSV41_UNIQUE_ONLY_PROBABILITY_DRAW", "0") == "1"
only_probability_full_draw = os.environ.get("DSV41_UNIQUE_ONLY_PROBABILITY_FULL_DRAW", "0") == "1"
only_mtp_cached = os.environ.get("DSV41_UNIQUE_ONLY_MTP_CACHED", "0") == "1"
only_hw_dense_fused = os.environ.get("DSV41_UNIQUE_ONLY_HW_DENSE_FUSED", "0") == "1"
only_router_shared_fused = os.environ.get("DSV41_UNIQUE_ONLY_ROUTER_SHARED_FUSED", "0") == "1"
only_hw_dense = os.environ.get("DSV41_UNIQUE_ONLY_HW_DENSE", "0") == "1"
only_rope_coherent = os.environ.get("DSV41_UNIQUE_ONLY_ROPE_COHERENT", "0") == "1"
only_scheduled_peer = os.environ.get("DSV41_UNIQUE_ONLY_SCHEDULED_PEER", "0") == "1"
only_n512_decode = os.environ.get("DSV41_UNIQUE_ONLY_N512_DECODE", "0") == "1"
only_mtp_sat = os.environ.get("DSV41_UNIQUE_ONLY_MTP_SAT", "0") == "1"
only_mhc_statistics = os.environ.get("DSV41_UNIQUE_ONLY_MHC_STATISTICS", "0") == "1"
only_input_norm = os.environ.get("DSV41_UNIQUE_ONLY_INPUT_NORM", "0") == "1"
only_silu_full_rows = os.environ.get("DSV41_UNIQUE_ONLY_SILU_FULL_ROWS", "0") == "1"
only_compressor_sequence = os.environ.get("DSV41_UNIQUE_ONLY_COMPRESSOR_SEQUENCE", "0") == "1"
only_pair_silu = os.environ.get("DSV41_UNIQUE_ONLY_PAIR_SILU", "0") == "1"
only_swa_cache = os.environ.get("DSV41_UNIQUE_ONLY_SWA_CACHE", "0") == "1"
only_split_feature_silu = os.environ.get("DSV41_UNIQUE_ONLY_SPLIT_FEATURE_SILU", "0") == "1"
only_layer_main_split = os.environ.get("DSV41_UNIQUE_ONLY_LAYER_MAIN_SPLIT", "0") == "1"
only_layer_main_reuse = os.environ.get("DSV41_UNIQUE_ONLY_LAYER_MAIN_REUSE", "0") == "1"
only_compact_stream_mme = os.environ.get("DSV41_UNIQUE_ONLY_COMPACT_STREAM_MME", "0") == "1"
only_unpaired_feature_silu = os.environ.get("DSV41_UNIQUE_ONLY_UNPAIRED_FEATURE_SILU", "0") == "1"
only_unpaired_w13 = os.environ.get("DSV41_UNIQUE_ONLY_UNPAIRED_W13", "0") == "1"
only_mhc_epilogue = os.environ.get("DSV41_UNIQUE_ONLY_MHC_EPILOGUE", "0") == "1"
only_merge_cache = os.environ.get("DSV41_UNIQUE_ONLY_MERGE_CACHE", "0") == "1"
only_control_reuse = os.environ.get("DSV41_UNIQUE_ONLY_CONTROL_REUSE", "0") == "1"
only_lane_candidates = os.environ.get("DSV41_UNIQUE_ONLY_LANE_CANDIDATES", "0") == "1"
only_bounded_nucleus = os.environ.get("DSV41_UNIQUE_ONLY_BOUNDED_NUCLEUS", "0") == "1"
only_journal_copy = os.environ.get("DSV41_UNIQUE_ONLY_JOURNAL_COPY", "0") == "1"
only_physical_role = os.environ.get("DSV41_UNIQUE_ONLY_PHYSICAL_ROLE", "0") == "1"
only_control_tiles = os.environ.get("DSV41_UNIQUE_ONLY_CONTROL_TILES", "0") == "1"
only_exp_pv = os.environ.get("DSV41_UNIQUE_ONLY_EXP_PV", "0") == "1"
only_q_bf16_rope = os.environ.get("DSV41_UNIQUE_ONLY_Q_BF16_ROPE", "0") == "1"
only_full_row = os.environ.get("DSV41_UNIQUE_ONLY_FULL_ROW", "0") == "1"
only_silu_scalar_cache = os.environ.get("DSV41_UNIQUE_ONLY_SILU_SCALAR_CACHE", "0") == "1"
only_fp4_table = os.environ.get("DSV41_UNIQUE_ONLY_FP4_TABLE", "0") == "1"
only_row_cache = os.environ.get("DSV41_UNIQUE_ONLY_ROW_CACHE", "0") == "1"
only_shared_finalize = os.environ.get("DSV41_UNIQUE_ONLY_SHARED_FINALIZE", "0") == "1"
only_mtp_fp8 = os.environ.get("DSV41_UNIQUE_ONLY_MTP_FP8", "0") == "1"
only_mtp_k128 = os.environ.get("DSV41_UNIQUE_ONLY_MTP_K128", "0") == "1"
only_router_pair = os.environ.get("DSV41_UNIQUE_ONLY_ROUTER_PAIR", "0") == "1"
only_future_peer_receive = os.environ.get("DSV41_UNIQUE_ONLY_FUTURE_PEER_RECEIVE", "0") == "1"
only_mla_fp16_pv = os.environ.get("DSV41_UNIQUE_ONLY_MLA_FP16_PV", "0") == "1"
only_mla_fp16_direct = os.environ.get("DSV41_UNIQUE_ONLY_MLA_FP16_DIRECT", "0") == "1"
only_router_ready = os.environ.get("DSV41_UNIQUE_ONLY_ROUTER_READY", "0") == "1"
only_main_adjacent_pv = os.environ.get("DSV41_UNIQUE_ONLY_MAIN_ADJACENT_PV", "0") == "1"
only_swa_source_reuse = os.environ.get("DSV41_UNIQUE_ONLY_SWA_SOURCE_REUSE", "0") == "1"
only_weighted_static_mask = os.environ.get("DSV41_UNIQUE_ONLY_WEIGHTED_STATIC_MASK", "0") == "1"
only_weighted_sparse = os.environ.get("DSV41_UNIQUE_ONLY_WEIGHTED_SPARSE", "0") == "1"
only_weighted_nucleus = os.environ.get("DSV41_UNIQUE_ONLY_WEIGHTED_NUCLEUS", "0") == "1"
only_paged_decoded = os.environ.get("DSV41_UNIQUE_ONLY_PAGED_DECODED", "0") == "1"
only_wo_handoff = os.environ.get("DSV41_UNIQUE_ONLY_WO_HANDOFF", "0") == "1"
if only_source_dense:
    sources = ["source_dense.cpp"]
elif only_fp8_prologue:
    sources = ["fp8_qkv_prologue.cpp"]
elif only_attention_interleave:
    sources = ["attention_interleave.cpp"]
elif only_w13_boundary:
    sources = ["w13_boundary_probe.cpp"]
elif only_router_batched:
    sources = ["router_batched.cpp"]
elif only_dense_bf16_pair:
    sources = ["dense_bf16_pair.cpp"]
elif only_mla_fp16_direct:
    sources = ["mla_fp16_direct.cpp"]
elif only_mla_fp16_pv:
    sources = ["mla_fp16_pv.cpp"]
elif only_future_peer_receive:
    sources = ["future_peer_receive.cpp", "future_epoch.cpp"]
elif only_router_pair:
    sources = ["router_pair.cpp"]
elif only_router_ready:
    sources = ["router_ready.cpp"]
elif only_mtp_k128:
    sources = ["mtp_k128.cpp"]
elif only_vocab_softmax:
    sources = ["softmax.cpp"]
elif only_peer_signal_probe:
    sources = ["peer_signal_probe.cpp"]
elif only_peer_post_collapse:
    sources = ["peer_post_collapse.cpp"]
elif only_ordered_peer_sum:
    sources = ["../../deepseek_v4/pytorch/hpu_dsv41_ordered_peer_sum_pt2.cpp"]
elif only_journal_batch_direct:
    sources = ["journal_batch_direct.cpp"]
elif only_journal_batch:
    sources = ["journal_batch.cpp"]
elif only_stream_exp:
    sources = ["shared_main_stream_exp.cpp"]
elif only_qk_flat_direct:
    sources = ["shared_main_qk_flat_direct.cpp"]
elif only_qk_flat:
    sources = ["shared_main_qk_flat.cpp"]
elif only_slicer_stitch:
    sources = ["sat_slicer_stitch.cpp", "sat_w13_k_pipeline_stitch.cpp"]
elif only_mhc_gate_packet:
    sources = ["../../deepseek_v4/pytorch/hpu_dsv41_mhc_gates_post_pt2.cpp"]
elif only_w13_k_pipeline_k512:
    sources = ["sat_w13_k_pipeline_k512.cpp"]
elif only_w13_k_pipeline_four:
    sources = ["sat_w13_k_pipeline_four.cpp"]
elif only_w2_k_pipeline:
    sources = ["sat_w2_k_pipeline.cpp"]
elif only_cooperative_silu:
    sources = ["sat_cooperative_silu.cpp"]
elif only_w2_channels:
    sources = ["sat_w2_channels.cpp"]
elif only_w2_reduce_n256:
    sources = ["sat_w2_reduce_n256.cpp"]
elif only_w2_shared_scale_n256:
    sources = ["sat_w2_shared_scale_n256.cpp"]
elif only_w2_ready_scale:
    sources = ["sat_w2_ready_scale.cpp"]
elif only_w13_k_pipeline:
    sources = ["sat_w13_k_pipeline.cpp"]
elif only_split_scale_planes:
    sources = ["sat_split_scale_planes.cpp"]
elif only_silu_decode_affine:
    sources = ["sat_silu_decode_affine.cpp"]
elif only_mhc_post_stats:
    sources = ["mhc_post_stats.cpp"]
elif only_main_single_bank:
    sources = ["main_single_bank.cpp"]
elif only_coherent_swa:
    sources = ["coherent_swa.cpp"]
elif only_input_quant_remat:
    sources = ["input_quant_remat.cpp"]
elif only_moe_peer_post:
    sources = ["moe_peer_post.cpp"]
elif only_mixed_dense_probe:
    sources = ["mixed_dense_probe.cpp"]
elif only_scaled_w13:
    sources = ["sat_scaled_w13.cpp"]
elif only_control_fp8_pair:
    sources = ["control_fp8_pair.cpp"]
elif only_control_fp8:
    sources = ["control_fp8_rrms.cpp"]
elif only_explicit_steps:
    sources = ["sat_explicit_steps.cpp"]
elif only_mhc_deferred:
    sources = ["mhc_deferred_post.cpp"]
elif only_silu_decode:
    sources = ["sat_silu_decode.cpp"]
elif only_norm_roundtrip:
    sources = ["norm_roundtrip.cpp"]
elif only_probability_full_draw:
    sources = ["probability_full_draw.cpp"]
elif only_probability_draw:
    sources = ["probability_draw.cpp"]
elif only_mtp_cached:
    sources = ["mtp_cached.cpp"]
elif only_hw_dense_fused:
    sources = ["hw_dense_fused.cpp"]
elif only_router_shared_fused:
    sources = ["router_shared_fused.cpp"]
elif only_hw_dense:
    sources = ["hw_dense.cpp"]
elif only_rope_coherent:
    sources = ["rope_coherent.cpp"]
elif only_scheduled_peer:
    sources = ["peer_post_norm.cpp"]
elif only_n512_decode:
    sources = ["sat_n512_decode.cpp"]
elif only_mtp_sat:
    sources = ["mtp_sat.cpp"]
elif only_mhc_statistics:
    sources = ["mhc_statistics.cpp"]
elif only_input_norm:
    sources = ["input_norm.cpp"]
elif only_main_adjacent_pv:
    sources = ["shared_main_adjacent_pv.cpp"]
elif only_swa_source_reuse:
    sources = ["swa_source_reuse.cpp"]
elif only_weighted_static_mask:
    sources = ["weighted_sparse.cpp", "weighted_finish_mask.cpp"]
elif only_silu_full_rows:
    sources = ["sat_silu_full_rows.cpp"]
elif only_compressor_sequence:
    sources = ["compressor_sequence.cpp"]
elif only_pair_silu:
    sources = ["sat_pair_silu.cpp"]
elif only_swa_cache:
    sources = ["swa_cached_main.cpp", "swa_batch_cache.cpp"]
elif only_split_feature_silu:
    sources = ["sat_split_feature_silu.cpp"]
elif only_layer_main_split:
    sources = ["shared_main_split.cpp"]
elif only_layer_main_reuse:
    sources = ["shared_main_batch.cpp"]
elif only_compact_stream_mme:
    sources = ["logical_compact_stream_mme.cpp"]
elif only_unpaired_feature_silu:
    sources = ["sat_unpaired_feature_silu.cpp"]
elif only_q_bf16_rope:
    sources = ["q_bf16_rope.cpp"]
elif only_full_row:
    sources = ["logical_full_row.cpp"]
elif only_silu_scalar_cache:
    sources = ["sat_silu_scalar_cache.cpp"]
elif only_fp4_table:
    sources = ["logical_fp4_table.cpp"]
elif only_row_cache:
    sources = ["logical_row_cache.cpp"]
elif only_shared_finalize:
    sources = ["sat_shared_finalize.cpp"]
elif only_mtp_fp8:
    sources = ["mtp_fp8.cpp"]
elif only_weighted_sparse:
    sources = ["weighted_sparse.cpp"]
elif only_weighted_nucleus:
    sources = ["weighted_nucleus.cpp"]
elif only_paged_decoded:
    sources = ["paged_decoded_rows.cpp"]
elif only_wo_handoff:
    sources = ["woa_dense_handoff.cpp"]
elif only_exp_pv:
    sources = ["logical_scale_exp_pv.cpp"]
elif only_control_tiles:
    sources = ["mhc_control_tiles.cpp"]
elif only_physical_role:
    sources = ["sat_physical_role_silu.cpp"]
elif only_journal_copy:
    sources = ["journal_copy.cpp"]
elif only_bounded_nucleus:
    sources = ["bounded_nucleus.cpp"]
elif only_lane_candidates:
    sources = ["vocab_lane_candidates.cpp"]
elif only_control_reuse:
    sources = ["mhc_control_reuse.cpp"]
elif only_merge_cache:
    sources = ["logical_merge_cache.cpp"]
elif only_unpaired_w13:
    sources = ["sat_unpaired_w13.cpp"]
elif only_mhc_epilogue:
    sources = ["mhc_mme_epilogue.cpp"]
elif os.environ.get("DSV41_UNIQUE_CONTROL_MME_FROM_BASE", "0") != "1":
    sources.append("control_mme.cpp")
extension_name = ("hpu_dsv41_source_dense_pt2" if only_source_dense else
                  "hpu_dsv41_fp8_qkv_prologue_pt2" if only_fp8_prologue else
                  "hpu_dsv41_attention_interleave_pt2" if only_attention_interleave else
                  "hpu_dsv41_w13_boundary_probe_pt2" if only_w13_boundary else
                  "hpu_dsv41_router_batched_pt2" if only_router_batched else
                  "hpu_dsv41_dense_bf16_pair_pt2" if only_dense_bf16_pair else
                  "hpu_dsv41_mla_fp16_direct_pt2" if only_mla_fp16_direct else
                  "hpu_dsv41_mla_fp16_pv_pt2" if only_mla_fp16_pv else
                  "hpu_dsv41_future_peer_receive_pt2" if only_future_peer_receive else
                  "hpu_dsv41_router_pair_pt2" if only_router_pair else
                  "hpu_dsv41_router_ready_pt2" if only_router_ready else
                  "hpu_dsv41_vocab_softmax_pt2" if only_vocab_softmax else
                  "hpu_dsv41_peer_signal_probe_pt2" if only_peer_signal_probe else
                  "hpu_dsv41_peer_post_collapse_pt2" if only_peer_post_collapse else
                  "hpu_dsv41_ordered_peer_sum_pt2" if only_ordered_peer_sum else
                  "hpu_dsv41_journal_batch_direct_pt2" if only_journal_batch_direct else
                  "hpu_dsv41_journal_batch_pt2" if only_journal_batch else
                  "hpu_dsv41_stream_exp_pt2" if only_stream_exp else
                  "hpu_dsv41_qk_flat_direct_pt2" if only_qk_flat_direct else
        "hpu_dsv41_qk_flat_pt2" if only_qk_flat else
                  "hpu_dsv41_slicer_stitch_pt2" if only_slicer_stitch else
                  "hpu_dsv41_mhc_gate_packet_pt2" if only_mhc_gate_packet else
                  "hpu_dsv41_w13_k_pipeline_k512_pt2" if only_w13_k_pipeline_k512 else
                  "hpu_dsv41_w13_k_pipeline_four_pt2" if only_w13_k_pipeline_four else
                  "hpu_dsv41_w2_k_pipeline_pt2" if only_w2_k_pipeline else
                  "hpu_dsv41_cooperative_silu_pt2" if only_cooperative_silu else
                  "hpu_dsv41_w2_channels_pt2" if only_w2_channels else
                  "hpu_dsv41_w2_reduce_n256_pt2" if only_w2_reduce_n256 else
                  "hpu_dsv41_w2_shared_scale_n256_pt2" if only_w2_shared_scale_n256 else
                  "hpu_dsv41_w2_ready_scale_pt2" if only_w2_ready_scale else
                  "hpu_dsv41_w13_k_pipeline_pt2" if only_w13_k_pipeline else
                  "hpu_dsv41_split_scale_planes_pt2" if only_split_scale_planes else
                  "hpu_dsv41_silu_decode_affine_pt2" if only_silu_decode_affine else
                  "hpu_dsv41_mhc_post_stats_pt2" if only_mhc_post_stats else
                  "hpu_dsv41_main_single_bank_pt2" if only_main_single_bank else
                  "hpu_dsv41_coherent_swa_pt2" if only_coherent_swa else
                  "hpu_dsv41_input_quant_remat_pt2" if only_input_quant_remat else
                  "hpu_dsv41_moe_peer_post_pt2" if only_moe_peer_post else
                  "hpu_dsv41_mixed_dense_probe_pt2" if only_mixed_dense_probe else
                  "hpu_dsv41_scaled_w13_pt2" if only_scaled_w13 else
                  "hpu_dsv41_control_fp8_pair_pt2" if only_control_fp8_pair else
                  "hpu_dsv41_control_fp8_pt2" if only_control_fp8 else
                  "hpu_dsv41_explicit_steps_pt2" if only_explicit_steps else
                  "hpu_dsv41_mhc_deferred_pt2" if only_mhc_deferred else
                  "hpu_dsv41_silu_decode_pt2" if only_silu_decode else
                  "hpu_dsv41_norm_roundtrip_pt2" if only_norm_roundtrip else
                  "hpu_dsv41_probability_full_draw_pt2" if only_probability_full_draw else
                  "hpu_dsv41_probability_draw_pt2" if only_probability_draw else
                  "hpu_dsv41_mtp_cached_pt2" if only_mtp_cached else
                  "hpu_dsv41_hw_dense_fused_pt2" if only_hw_dense_fused else
                  "hpu_dsv41_router_shared_fused_pt2" if only_router_shared_fused else
                  "hpu_dsv41_hw_dense_pt2" if only_hw_dense else
                  "hpu_dsv41_rope_coherent_pt2" if only_rope_coherent else
                  "hpu_dsv41_scheduled_peer_pt2" if only_scheduled_peer else
                  "hpu_dsv41_n512_decode_pt2" if only_n512_decode else
                  "hpu_dsv41_mtp_sat_pt2" if only_mtp_sat else
                  "hpu_dsv41_mhc_statistics_pt2" if only_mhc_statistics else
                  "hpu_dsv41_input_norm_pt2" if only_input_norm else
                  "hpu_dsv41_main_adjacent_pv_pt2" if only_main_adjacent_pv else
                  "hpu_dsv41_swa_source_reuse_pt2" if only_swa_source_reuse else
                  "hpu_dsv41_weighted_static_mask_pt2" if only_weighted_static_mask else
                  "hpu_dsv41_silu_full_rows_pt2" if only_silu_full_rows else
                  "hpu_dsv41_compressor_sequence_pt2" if only_compressor_sequence else
                  "hpu_dsv41_pair_silu_pt2" if only_pair_silu else
                  "hpu_dsv41_swa_cache_pt2" if only_swa_cache else
                  "hpu_dsv41_split_feature_silu_pt2" if only_split_feature_silu else
                  "hpu_dsv41_layer_main_split_pt2" if only_layer_main_split else
                  "hpu_dsv41_layer_main_reuse_pt2" if only_layer_main_reuse else
                  "hpu_dsv41_compact_stream_mme_pt2" if only_compact_stream_mme else
                  "hpu_dsv41_unpaired_feature_silu_pt2" if only_unpaired_feature_silu else
                  "hpu_dsv41_q_bf16_rope_pt2" if only_q_bf16_rope else
                  "hpu_dsv41_full_row_pt2" if only_full_row else
                  "hpu_dsv41_silu_scalar_cache_pt2" if only_silu_scalar_cache else
                  "hpu_dsv41_fp4_table_pt2" if only_fp4_table else
                  "hpu_dsv41_row_cache_pt2" if only_row_cache else
                  "hpu_dsv41_shared_finalize_pt2" if only_shared_finalize else
                  "hpu_dsv41_mtp_k128_pt2" if only_mtp_k128 else
                  "hpu_dsv41_mtp_fp8_pt2" if only_mtp_fp8 else
                  "hpu_dsv41_weighted_sparse_pt2" if only_weighted_sparse else
                  "hpu_dsv41_weighted_nucleus_pt2" if only_weighted_nucleus else
                  "hpu_dsv41_paged_decoded_pt2" if only_paged_decoded else
                  "hpu_dsv41_wo_handoff_pt2" if only_wo_handoff else
                  "hpu_dsv41_exp_pv_pt2" if only_exp_pv else
                  "hpu_dsv41_control_tiles_pt2" if only_control_tiles else
                  "hpu_dsv41_physical_role_pt2" if only_physical_role else
                  "hpu_dsv41_journal_copy_pt2" if only_journal_copy else
                  "hpu_dsv41_bounded_nucleus_pt2" if only_bounded_nucleus else
                  "hpu_dsv41_lane_candidates_pt2" if only_lane_candidates else
                  "hpu_dsv41_control_reuse_pt2" if only_control_reuse else
                  "hpu_dsv41_merge_cache_pt2" if only_merge_cache else
                  "hpu_dsv41_unpaired_w13_pt2" if only_unpaired_w13 else
                  "hpu_dsv41_mhc_epilogue_pt2" if only_mhc_epilogue else "hpu_dsv41_unique_pt2")
setup(name=extension_name, ext_modules=[CppExtension(
    extension_name, [str(Path(__file__).resolve().parent / source) for source in sources],
    include_dirs=includes,
    library_dirs=[get_lib_dir()], libraries=["habana_pytorch2_plugin.upstream", "habana_pytorch_backend.upstream"],
    extra_compile_args=["-O2", "-std=c++17", "-DFMT_HEADER_ONLY=1", "-DGENERIC_HELPERS"]
    + (["-DDSV41_DENSE_BITS12=1"]
       if only_source_dense and os.environ.get("DSV41_UNIQUE_SOURCE_DENSE_BITS12", "0") == "1" else [])
    + (["-DDSV41_HW_DIRECT_BIAS=1"] if private_hw_bias else [])
    + (["-DDSV41_MHC_CARRIED_RRMS_ONLY=1"]
       if only_mhc_deferred and os.environ.get("DSV41_MHC_QUERY_OWNER", "0") == "1" else [])
    + (["-DDSV41_NORM_ROUNDTRIP_TILED=1"]
       if only_norm_roundtrip and os.environ.get("DSV41_NORM_ROUNDTRIP_TILED", "0") == "1" else []),
)], cmdclass={"build_ext": BuildExtension})
