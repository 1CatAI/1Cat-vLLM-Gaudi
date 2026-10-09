// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <initializer_list>
#include "../deepseek_v4/host/deepseek_v41_mhc_weighted_stats_gaudi2.hpp"
#include <vector>
#include "../deepseek_v4/host/deepseek_v41_mhc_gates_post_gaudi2.hpp"
#include "../deepseek_v4/host/deepseek_v41_silu_tile_gaudi2.hpp"
#include "../deepseek_v4/host/deepseek_v41_expert_route_pack.hpp"
#include "../deepseek_v4/host/deepseek_v41_peer_post_norm_quant_gaudi2.hpp"
#include "../deepseek_v4/host/deepseek_v41_peer_post_collapse_gaudi2.hpp"
#include "../deepseek_v4/host/deepseek_v41_kv_norm_rope_publish_gaudi2.hpp"
#include "../deepseek_v4/host/deepseek_v41_qkv_norm_publish_gaudi2.hpp"
extern unsigned char _binary___deepseek_v41_router_shared_scaled_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_router_shared_scaled_gaudi2_o_end;
#define DeepseekV41Mxfp4PreparedDequantFP8Gaudi2 DeepseekV41MtpDequantFP8Gaudi2
#include "../deepseek_v4/host/deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2.hpp"
#undef DeepseekV41Mxfp4PreparedDequantFP8Gaudi2

#define DeepseekV41MainPublishGatherGaudi2 DeepseekV41BatchedMainPublishGatherGaudi2
#include "../deepseek_v4/host/deepseek_v41_main_publish_gather_gaudi2.hpp"
#undef DeepseekV41MainPublishGatherGaudi2
#define DeepseekV41MainReuseGatherGaudi2 DeepseekV41BatchedMainReuseGatherGaudi2
#include "../deepseek_v4/host/deepseek_v41_main_reuse_gather_gaudi2.hpp"
#undef DeepseekV41MainReuseGatherGaudi2

class DeepseekV41SwaOnlyReuseGatherGaudi2 {
public:
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};

namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

#define ELF(NAME) extern unsigned char _binary___##NAME##_o_start, _binary___##NAME##_o_end
ELF(deepseek_v41_expert_w13_split_scale_gaudi2);
ELF(deepseek_v41_expert_w2_split_scale_gaudi2);
ELF(deepseek_v41_expert_w13_k512_gaudi2);
ELF(deepseek_v41_expert_w2_k512_gaudi2);
ELF(deepseek_v41_expert_pair_channels_gaudi2);
ELF(deepseek_v41_expert_scaled_silu_quant_gaudi2);
ELF(deepseek_v41_control_statistics_gaudi2);
ELF(deepseek_v41_control_quant_rrms_gaudi2);
ELF(deepseek_v41_weighted_row_parameters_gaudi2);
ELF(deepseek_v41_norm_prepared_weighted_stats_gaudi2);
ELF(deepseek_v41_main_single_bank_gather_gaudi2);
ELF(deepseek_v41_control_pair_quant_rrms_gaudi2);
ELF(deepseek_v41_control_pair_finish_gaudi2);
ELF(deepseek_v41_expert_token_wide_explicit_steps_gaudi2);
ELF(deepseek_v41_mhc_deferred_post_gaudi2);
ELF(deepseek_v41_silu_decode_c6_gaudi2);
ELF(deepseek_v41_norm_roundtrip_tiled_gaudi2);
ELF(deepseek_v41_norm_statistics_gaudi2);
ELF(deepseek_v41_norm_roundtrip_gaudi2);
ELF(deepseek_v41_probability_parts_gaudi2);
ELF(deepseek_v41_probability_draw_gaudi2);
ELF(deepseek_v41_expert_cached_gather_bf16_gaudi2);
ELF(deepseek_v41_fixed_dense_quant_gaudi2);
ELF(deepseek_v41_engram_batch_gaudi2);
ELF(deepseek_v41_mla_stream_metadata_gaudi2);
ELF(deepseek_v41_mla_stream_decode_gaudi2);
ELF(deepseek_v41_mla_stream_softmax_gaudi2);
ELF(deepseek_v41_mhc_mme_epilogue_gaudi2);
ELF(deepseek_v41_expert_physical_silu_gaudi2);
ELF(deepseek_v41_mla_hash_owners_gaudi2);
ELF(deepseek_v41_mla_hash_decode_gaudi2);
ELF(deepseek_v41_mla_hash_gather_gaudi2);
ELF(deepseek_v41_nucleus_mass_gaudi2);
ELF(deepseek_v41_nucleus_draw_gaudi2);
ELF(deepseek_v41_nucleus_finish_gaudi2);
ELF(deepseek_v41_pv_rope_f32_gaudi2);
ELF(deepseek_v41_silu_static_gaudi2);
ELF(deepseek_v41_unique_control_gaudi2);
ELF(deepseek_v41_unique_program_gaudi2);
ELF(deepseek_v41_unique_ids_gaudi2);
ELF(deepseek_v41_unique_owner_pack_fp8_gaudi2);
ELF(deepseek_v41_unique_silu_quant_fp8_gaudi2);
ELF(deepseek_v41_unique_scale_fp8_gaudi2);
ELF(deepseek_v41_unique_restore_simple_bf16_gaudi2);
ELF(deepseek_v41_mla_union_metadata_gaudi2);
ELF(deepseek_v41_mla_union_scatter_gaudi2);
ELF(deepseek_v41_mla_merged_gaudi2);
ELF(deepseek_v41_vocab_filter_gaudi2);
ELF(deepseek_v41_vocab_filter_emit_gaudi2);
ELF(deepseek_v41_logical_mla_decoded_gaudi2);
ELF(deepseek_v41_fp4_cache_write_rows_gaudi2);
ELF(deepseek_v41_logical_mla_write_ordered_gaudi2);
ELF(deepseek_v41_index_keys_write_ordered_gaudi2);
ELF(deepseek_v41_mla_slotmap_gaudi2);
ELF(deepseek_v41_mla_slotmap_scatter_gaudi2);
ELF(deepseek_v41_q_scale_rope_tiled_gaudi2);
ELF(deepseek_v41_expert_token_wide_sat16_fp8_gaudi2);
ELF(deepseek_v41_expert_n256_sat_active_k_gaudi2);
ELF(deepseek_v41_logical_mla_bf16_only_gaudi2);
ELF(deepseek_v41_selected_mla_pair_softmax_gaudi2);
ELF(deepseek_v41_mla_pair_reduce_bf16_gaudi2);
ELF(deepseek_v41_expert_probe_read_gaudi2);
ELF(deepseek_v41_expert_probe_decode_gaudi2);
ELF(deepseek_v41_expert_probe_store_gaudi2);
ELF(deepseek_v41_expert_token_wide_k_first_gaudi2);
ELF(deepseek_v41_expert_token_wide_bytes_gaudi2);
ELF(deepseek_v41_expert_n256_bytes_gaudi2);
ELF(deepseek_v41_silu640_gaudi2);
ELF(deepseek_v41_vocab_radix_threshold_gaudi2);
ELF(deepseek_v41_vocab_radix_emit_gaudi2);
ELF(deepseek_v41_expert_three_route_sat_fp8_gaudi2);
ELF(deepseek_v41_expert_three_route_scale_reduce_gaudi2);
ELF(deepseek_v41_logical_mla_coord_cached_gaudi2);
ELF(deepseek_v41_mla_shared_owners_gaudi2);
ELF(deepseek_v41_mla_shared_decode_gaudi2);
ELF(deepseek_v41_mla_shared_softmax_gaudi2);
ELF(deepseek_v41_expert_token_wide_group_pipe_gaudi2);
ELF(deepseek_v41_expert_down_group_pipe_gaudi2);
ELF(deepseek_v41_mla_compact_prefix_gaudi2);
ELF(deepseek_v41_mla_compact_metadata_gaudi2);
ELF(deepseek_v41_mla_compact_decode_gaudi2);
ELF(deepseek_v41_mla_compact_softmax_gaudi2);
ELF(deepseek_v41_silu_channel_vector_gaudi2);
ELF(deepseek_v41_expert_token_wide_unroll_steps_gaudi2);
ELF(deepseek_v41_logical_main_mirror_gaudi2);
ELF(deepseek_v41_logical_scale_cache_gaudi2);
ELF(deepseek_v41_logical_merge_cache_gaudi2);
ELF(deepseek_v41_mhc_control_reuse_gaudi2);
ELF(deepseek_v41_vocab_lane_candidates_gaudi2);
ELF(deepseek_v41_bounded_nucleus_parts_gaudi2);
ELF(deepseek_v41_bounded_local_distribution_gaudi2);
ELF(deepseek_v41_journal_copy_gaudi2);
ELF(deepseek_v41_expert_physical_role_silu_gaudi2);
ELF(deepseek_v41_woa_scale_dense_quant_gaudi2);
ELF(deepseek_v41_fp4_paged_decoded_rows_gaudi2);
ELF(deepseek_v41_logical_row_cache_gaudi2);
ELF(deepseek_v41_logical_fp4_table_gaudi2);
ELF(deepseek_v41_silu_scalar_cache_gaudi2);
ELF(deepseek_v41_logical_full_row_gaudi2);
ELF(deepseek_v41_q_bf16_rope_tiled_gaudi2);
ELF(deepseek_v41_scale_reduce_shared_gaudi2);
ELF(deepseek_v41_swa_cached_publish_gaudi2);
ELF(deepseek_v41_swa_cached_reuse_gaudi2);
ELF(deepseek_v41_swa_batch_cache_write_gaudi2);
ELF(deepseek_v41_silu_split_activate_gaudi2);
ELF(deepseek_v41_weighted_mass_bins_gaudi2);
ELF(deepseek_v41_input_norm_bf16_gaudi2);
ELF(deepseek_v41_mhc_statistics_gaudi2);
ELF(deepseek_v41_mhc_statistics_finish_gaudi2);
ELF(deepseek_v41_rope_all_rows_gaudi2);
ELF(deepseek_v41_rope_inverse_all_rows_gaudi2);
ELF(deepseek_v41_expert_token_wide_n512_gaudi2);
ELF(deepseek_v41_expert_down_n512_gaudi2);
ELF(deepseek_v41_weighted_mass_advance_gaudi2);
ELF(deepseek_v41_weighted_mass_finish_gaudi2);
ELF(deepseek_v41_weighted_finish_mask_gaudi2);
ELF(deepseek_v41_swa_source_reuse_gaudi2);
ELF(deepseek_v41_mla_adjacent_softmax_gaudi2);
ELF(deepseek_v41_mla_adjacent_finish_gaudi2);
ELF(deepseek_v41_swa_keys_reuse_gaudi2);
ELF(deepseek_v41_weighted_score_bounds_gaudi2);
ELF(deepseek_v41_weighted_sparse_bins_gaudi2);
ELF(deepseek_v41_weighted_active_blocks_gaudi2);
ELF(deepseek_v41_expert_pair_silu_gaudi2);
ELF(deepseek_v41_compressor_sequence_bf16_gaudi2);
ELF(deepseek_v41_mhc_control_tiles_gaudi2);
ELF(deepseek_v41_mhc_control_tiles_finish_gaudi2);
ELF(deepseek_v41_mla_exp_bf16_gaudi2);
ELF(deepseek_v41_mla_exp_normalize_gaudi2);
ELF(deepseek_v41_logical_scale_bf16_gaudi2);
ELF(deepseek_v41_selected_mla_stacked_softmax_gaudi2);
ELF(deepseek_v41_mla_stacked_reduce_bf16_gaudi2);
ELF(deepseek_v41_candidate_mirror_keys_gaudi2);
ELF(deepseek_v41_coherent_swa_keys_gaudi2);
ELF(deepseek_v41_coherent_swa_pair_softmax_gaudi2);
ELF(deepseek_v41_coherent_swa_finish_gaudi2);
ELF(deepseek_v41_coherent_swa_decode_gaudi2);
ELF(deepseek_v41_coherent_swa_softmax_gaudi2);
ELF(deepseek_v41_expert_token_wide_affine_route_gaudi2);
ELF(deepseek_v41_expert_token_wide_k64_sat_gaudi2);
ELF(deepseek_v41_expert_n256_k64_sat_gaudi2);
#undef ELF

namespace {
using namespace tpc_lib_api;
constexpr const char* guids[] = {
    "custom_deepseek_v41_unique_control_i32_gaudi2",
    "custom_deepseek_v41_unique_program_i32_gaudi2",
    "custom_deepseek_v41_unique_ids_i32_gaudi2",
    "custom_deepseek_v41_unique_pack_fp8_gaudi2",
    "custom_deepseek_v41_unique_silu_quant_fp8_gaudi2",
    "custom_deepseek_v41_unique_scale_fp8_gaudi2",
    "custom_deepseek_v41_unique_restore_bf16_gaudi2",
    "custom_deepseek_v41_mla_union_metadata_i32_gaudi2",
    "custom_deepseek_v41_mla_union_scatter_gaudi2",
    "custom_deepseek_v41_logical_mla_decoded_gaudi2",
    "custom_deepseek_v41_fp4_cache_write_rows_gaudi2",
    "custom_deepseek_v41_logical_mla_write_ordered_gaudi2",
    "custom_deepseek_v41_index_keys_write_ordered_gaudi2",
    "custom_deepseek_v41_mla_slotmap_gaudi2",
    "custom_deepseek_v41_mla_slotmap_scatter_gaudi2",
    "custom_deepseek_v41_q_scale_rope_tiled_gaudi2",
    "custom_deepseek_v41_expert_token_wide_sat16_fp8_gaudi2",
    "custom_deepseek_v41_expert_n256_sat_active_k_gaudi2",
    "custom_deepseek_v41_logical_mla_bf16_only_gaudi2",
    "custom_deepseek_v41_selected_mla_pair_softmax_gaudi2",
    "custom_deepseek_v41_mla_pair_reduce_bf16_gaudi2",
    "custom_deepseek_v41_expert_probe_read_gaudi2",
    "custom_deepseek_v41_expert_probe_decode_gaudi2",
    "custom_deepseek_v41_expert_probe_store_gaudi2",
    "custom_deepseek_v41_expert_token_wide_k_first_gaudi2",
    "custom_deepseek_v41_expert_token_wide_small_sram_gaudi2",
    "custom_deepseek_v41_expert_token_wide_bytes_gaudi2",
    "custom_deepseek_v41_expert_n256_bytes_gaudi2",
    "custom_deepseek_v41_silu640_gaudi2",
    "custom_deepseek_v41_vocab_radix_threshold_gaudi2",
    "custom_deepseek_v41_vocab_radix_emit_gaudi2",
    "custom_deepseek_v41_logical_mla_cohesive_vector_gaudi2",
    "custom_deepseek_v41_dspark_silu_activate_tile_gaudi2",
    "custom_deepseek_v41_dspark_silu_quant_tile_gaudi2",
    "custom_deepseek_v41_mla_merged_gaudi2",
    "custom_deepseek_v41_vocab_filter_mask_gaudi2",
    "custom_deepseek_v41_vocab_filter_emit_gaudi2",
    "custom_deepseek_v41_expert_three_route_sat_fp8_gaudi2",
    "custom_deepseek_v41_expert_three_route_scale_reduce_gaudi2",
    "custom_deepseek_v41_peer_post_norm_quant_gaudi2",
    "custom_deepseek_v41_peer_post_collapse_gaudi2",
    "custom_deepseek_v41_dspark_silu_static_gaudi2",
    "custom_deepseek_v41_kv_norm_rope_publish_gaudi2",
    "custom_deepseek_v41_pv_rope_f32_gaudi2",
    "custom_deepseek_v41_nucleus_mass_gaudi2",
    "custom_deepseek_v41_nucleus_draw_gaudi2",
    "custom_deepseek_v41_nucleus_finish_gaudi2",
    "custom_deepseek_v41_dspark_qkv_norm_publish_gaudi2",
    "custom_deepseek_v41_mla_hash_owners_gaudi2",
    "custom_deepseek_v41_mla_hash_decode_gaudi2",
    "custom_deepseek_v41_mla_hash_gather_gaudi2",
    "custom_deepseek_v41_expert_physical_silu_gaudi2",
    "custom_deepseek_v41_logical_mla_coord_cached_gaudi2",
    "custom_deepseek_v41_mla_shared_owners_gaudi2",
    "custom_deepseek_v41_mla_shared_decode_gaudi2",
    "custom_deepseek_v41_mla_shared_softmax_gaudi2",
    "custom_deepseek_v41_expert_token_wide_group_pipe_gaudi2",
    "custom_deepseek_v41_expert_down_group_pipe_gaudi2",
    "custom_deepseek_v41_mla_compact_prefix_gaudi2",
    "custom_deepseek_v41_mla_compact_metadata_gaudi2",
    "custom_deepseek_v41_mla_compact_decode_gaudi2",
    "custom_deepseek_v41_mla_compact_softmax_gaudi2",
    "custom_deepseek_v41_silu_channel_vector_gaudi2",
    "custom_deepseek_v41_expert_token_wide_unroll_steps_gaudi2",
    "custom_deepseek_v41_logical_main_mirror_gaudi2",
    "custom_deepseek_v41_logical_scale_cache_gaudi2",
    "custom_deepseek_v41_silu_affine_gaudi2",
    "custom_deepseek_v41_candidate_mirror_keys_gaudi2",
    "custom_deepseek_v41_expert_token_wide_affine_route_gaudi2",
    "custom_deepseek_v41_logical_scale_bf16_gaudi2",
    "custom_deepseek_v41_expert_token_wide_k64_sat_gaudi2",
    "custom_deepseek_v41_expert_n256_k64_sat_gaudi2",
    "custom_deepseek_v41_selected_mla_stacked_softmax_gaudi2",
    "custom_deepseek_v41_mla_stacked_reduce_bf16_gaudi2",
    "custom_deepseek_v41_logical_scale_cache_batch6_gaudi2",
    "custom_deepseek_v41_mhc_mme_epilogue_gaudi2",
    "custom_deepseek_v41_logical_merge_cache_gaudi2",
    "custom_deepseek_v41_mhc_control_reuse_gaudi2",
    "custom_deepseek_v41_vocab_lane_candidates_gaudi2",
    "custom_deepseek_v41_bounded_nucleus_parts_gaudi2",
    "custom_deepseek_v41_bounded_local_distribution_gaudi2",
    "custom_deepseek_v41_journal_copy_gaudi2",
    "custom_deepseek_v41_expert_physical_role_silu_gaudi2",
    "custom_deepseek_v41_mhc_control_tiles_gaudi2",
    "custom_deepseek_v41_mhc_control_tiles_finish_gaudi2",
    "custom_deepseek_v41_mla_exp_bf16_gaudi2",
    "custom_deepseek_v41_mla_exp_normalize_gaudi2",
    "custom_deepseek_v41_woa_scale_dense_quant_gaudi2",
    "custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2",
    "custom_deepseek_v41_weighted_mass_bins_gaudi2",
    "custom_deepseek_v41_weighted_mass_advance_gaudi2",
    "custom_deepseek_v41_weighted_mass_finish_gaudi2",
    "custom_deepseek_v41_mtp_dequant_fp8_gaudi2",
    "custom_deepseek_v41_scale_reduce_shared_gaudi2",
    "custom_deepseek_v41_logical_row_cache_gaudi2",
    "custom_deepseek_v41_logical_fp4_table_gaudi2",
    "custom_deepseek_v41_silu_scalar_cache_gaudi2",
    "custom_deepseek_v41_logical_full_row_gaudi2",
    "custom_deepseek_v41_q_bf16_rope_tiled_gaudi2",
    "custom_deepseek_v41_mla_stream_metadata_gaudi2",
    "custom_deepseek_v41_mla_stream_decode_gaudi2",
    "custom_deepseek_v41_mla_stream_softmax_gaudi2",
    "custom_deepseek_v41_main_batch_publish_gather_gaudi2",
    "custom_deepseek_v41_main_batch_reuse_gather_gaudi2",
    "custom_deepseek_v41_swa_only_reuse_gather_gaudi2",
    "custom_deepseek_v41_silu_split_activate_gaudi2",
    "custom_deepseek_v41_swa_cached_publish_gaudi2",
    "custom_deepseek_v41_swa_cached_reuse_gaudi2",
    "custom_deepseek_v41_swa_batch_cache_write_gaudi2",
    "custom_deepseek_v41_weighted_score_bounds_gaudi2",
    "custom_deepseek_v41_weighted_sparse_bins_gaudi2",
    "custom_deepseek_v41_weighted_active_blocks_gaudi2",
    "custom_deepseek_v41_expert_pair_silu_gaudi2",
    "custom_deepseek_v41_compressor_sequence_bf16_gaudi2",
    "custom_deepseek_v41_silu_full_rows_gaudi2",
    "custom_deepseek_v41_weighted_finish_mask_gaudi2",
    "custom_deepseek_v41_swa_source_reuse_gaudi2",
    "custom_deepseek_v41_mla_adjacent_softmax_gaudi2",
    "custom_deepseek_v41_mla_adjacent_finish_gaudi2",
    "custom_deepseek_v41_swa_keys_reuse_gaudi2",
    "custom_deepseek_v41_engram_batch_gaudi2",
    "custom_deepseek_v41_input_norm_bf16_gaudi2",
    "custom_deepseek_v41_mhc_statistics_gaudi2",
    "custom_deepseek_v41_mhc_statistics_finish_gaudi2",
    "custom_deepseek_v41_expert_token_wide_n512_gaudi2",
    "custom_deepseek_v41_expert_down_n512_gaudi2",
    "custom_deepseek_v41_rope_coherent_bf16_gaudi2",
    "custom_deepseek_v41_rope_inverse_coherent_bf16_gaudi2",
    "custom_deepseek_v41_router_shared_scaled_gaudi2",
    "custom_deepseek_v41_shared_silu_full_product_gaudi2",
    "custom_deepseek_v41_fixed_dense_quant_gaudi2",
    "custom_deepseek_v41_expert_cached_gather_bf16_gaudi2",
    "custom_deepseek_v41_probability_parts_gaudi2",
    "custom_deepseek_v41_probability_draw_gaudi2",
    "custom_deepseek_v41_norm_statistics_gaudi2",
    "custom_deepseek_v41_norm_roundtrip_gaudi2",
    "custom_deepseek_v41_norm_roundtrip_tiled_gaudi2",
    "custom_deepseek_v41_expert_silu_decode_fp8_gaudi2",
    "custom_deepseek_v41_mhc_mme_post_collapse_gaudi2",
    "custom_deepseek_v41_expert_token_wide_explicit_steps_gaudi2",
    "custom_deepseek_v41_control_statistics_gaudi2",
    "custom_deepseek_v41_control_quant_rrms_gaudi2",
    "custom_deepseek_v41_expert_pair_channels_gaudi2",
    "custom_deepseek_v41_expert_scaled_silu_quant_gaudi2",
    "custom_deepseek_v41_attention_norm_quant_gaudi2_remat",
    "custom_deepseek_v41_coherent_swa_decode_gaudi2",
    "custom_deepseek_v41_coherent_swa_softmax_gaudi2",
    "custom_deepseek_v41_coherent_swa_keys_gaudi2",
    "custom_deepseek_v41_coherent_swa_pair_softmax_gaudi2",
    "custom_deepseek_v41_coherent_swa_finish_gaudi2",
    "custom_deepseek_v41_control_pair_quant_rrms_gaudi2",
    "custom_deepseek_v41_control_pair_finish_gaudi2",
    "custom_deepseek_v41_main_single_bank_gather_gaudi2",
    "custom_deepseek_v41_mhc_post_weighted_stats_gaudi2",
    "custom_deepseek_v41_norm_from_weighted_stats_gaudi2",
    "custom_deepseek_v41_weighted_row_parameters_gaudi2",
    "custom_deepseek_v41_norm_prepared_weighted_stats_gaudi2",
    "custom_deepseek_v41_dspark_silu_decode_affine_gaudi2",
    "custom_deepseek_v41_expert_w13_split_scale_gaudi2",
    "custom_deepseek_v41_expert_w2_split_scale_gaudi2",
    "custom_deepseek_v41_mhc_gates_post_gaudi2",
    "custom_deepseek_v41_expert_w13_k512_gaudi2",
    "custom_deepseek_v41_expert_w2_k512_gaudi2"};
constexpr unsigned guid_count = sizeof(guids) / sizeof(guids[0]);
int kind(const char* name) {
    for (unsigned i = 0; i < guid_count; ++i) if (!std::strcmp(name, guids[i])) return i;
    return -1;
}
template<typename T> T base(const char* name) {
    static void* handle = [] {
        const char* path = std::getenv("VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL");
        return path && *path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
    }();
    return handle ? reinterpret_cast<T>(dlsym(handle, name)) : nullptr;
}
bool shape(const Tensor& tensor, TensorDataType type, std::initializer_list<uint64_t> dimensions) {
    if (tensor.geometry.dataType != type || tensor.geometry.dims != dimensions.size()) return false;
    unsigned index = 0;
    for (auto size : dimensions) if (tensor.geometry.maxSizes[index++] != size) return false;
    return true;
}
GlueCodeReturn loadElf(HabanaKernelInstantiation* output, const unsigned char* first, const unsigned char* last) {
    const auto capacity = output->kernel.elfSize;
    output->kernel.elfSize = last - first;
    if (capacity < output->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(output->kernel.kernelElf, first, output->kernel.elfSize);
    return GLUE_SUCCESS;
}
}

extern "C" {
uint64_t GetLibVersion() {
    auto next = base<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");
    return next ? next() : 0;
}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId device, uint32_t* count,
                                         tpc_lib_api::GuidInfo* output) {
    using namespace tpc_lib_api;
    if (!count) return GLUE_FAILED;
    auto next = base<decltype(&GetKernelGuids)>("GetKernelGuids");
    if (!next) return GLUE_FAILED;
    if (device != DEVICE_ID_GAUDI2) return next(device, count, output);
    uint32_t inherited = 0;
    auto status = next(device, &inherited, nullptr);
    if (status != GLUE_SUCCESS) return status;
    const auto capacity = *count;
    *count = inherited + guid_count;
    if (!output || !capacity) return GLUE_SUCCESS;
    if (capacity < *count) return GLUE_FAILED;
    std::memset(output, 0, guid_count * sizeof(*output));
    for (unsigned i = 0; i < guid_count; ++i) std::strcpy(output[i].name, guids[i]);
    return next(device, &inherited, output + guid_count);
}
extern unsigned char _binary___deepseek_v41_attention_norm_quant_remat_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_attention_norm_quant_remat_gaudi2_o_end;
static tpc_lib_api::GlueCodeReturn instantiate_private(tpc_lib_api::HabanaKernelParams* input,
                                                      tpc_lib_api::HabanaKernelInstantiation* output);
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* input,
                                               tpc_lib_api::HabanaKernelInstantiation* output) {
    return instantiate_private(input,output);
}
static tpc_lib_api::GlueCodeReturn instantiate_private(tpc_lib_api::HabanaKernelParams* input,
                                                      tpc_lib_api::HabanaKernelInstantiation* output) {
    using namespace tpc_lib_api;
    if (!input || !output) return GLUE_FAILED;
    const auto operation = kind(input->guid.name);
    if (operation < 0) {
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        return next ? next(input, output) : GLUE_FAILED;
    }
    if (input->deviceId != DEVICE_ID_GAUDI2 || operation < 0) return GLUE_FAILED;
    if(operation==160)return DeepseekV41MhcGatesPostGaudi2().GetGcDefinitions(input,output);
    if(operation==158 || operation==159 || operation==161 || operation==162) {
        const bool up=operation==158 || operation==161;
        const bool coarse=operation==161 || operation==162;
        if(input->inputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& q=input->inputTensors[1].geometry;
        const auto& groups=input->inputTensors[2].geometry;
        const bool compact=groups.maxSizes[0]*16==q.maxSizes[0];
        if(groups.dims!=3 || (!compact && groups.maxSizes[0]*8!=q.maxSizes[0]) ||
           !shape(input->inputTensors[4],DATA_I16,{128,q.maxSizes[1],q.maxSizes[2]}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        // Reuse the qualified decoder's shape checks and all existing axes.
        // Only the scale K map and the new immutable channel input differ.
        auto inherited=*input;inherited.inputTensorNr=4;
        std::vector<Tensor> tensors(input->inputTensors,input->inputTensors+4);
        inherited.inputTensors=tensors.data();
        if(compact)inherited.inputTensors[2].geometry.maxSizes[0]+=128;
        std::strcpy(inherited.guid.name,up?
            "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2":
            "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto requestedKernel=output->kernel;
        // Query inherited maps without using the caller's new-ELF capacity:
        // the old binary can be larger than the selected split-scale binary.
        output->kernel.elfSize=0;output->kernel.kernelElf=nullptr;
        const auto status=next(&inherited,output);
        output->kernel=requestedKernel;
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        auto& scale=output->inputTensorAccessPattern[2];
        const int tile=coarse?512:128;
        const int scaleWords=tile*(compact?4:8);
        scale.mapping[0]={2,float(scaleWords),0,float(scaleWords-1),false};
        auto& code=output->inputTensorAccessPattern[4];code={};
        code.mapping[0]={2,0,0,127,false};
        code.mapping[1]={0,float(up?0:1),0,float(up?q.maxSizes[1]-1:0),false};
        code.mapping[2]={1,0,0,float(q.maxSizes[2]-1),false};
        output->kernel.paramsNr=0;
        if(coarse) {
            const auto outputK=input->outputTensors[0].geometry.maxSizes[1];
            if(!outputK || outputK%32 || outputK>0x7fffffff)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            output->indexSpaceGeometry[2]=(outputK+tile-1)/tile;
            output->inputTensorAccessPattern[1].mapping[0]={2,float(tile*64),0,float(tile*64-1),false};
            output->outputTensorAccessPattern[0].mapping[1]={2,float(tile),0,float(tile-1),false};
            output->kernel.paramsNr=1;output->kernel.scalarParams[0]=outputK;
            return up?loadElf(output,&_binary___deepseek_v41_expert_w13_k512_gaudi2_o_start,
                &_binary___deepseek_v41_expert_w13_k512_gaudi2_o_end):
                loadElf(output,&_binary___deepseek_v41_expert_w2_k512_gaudi2_o_start,
                &_binary___deepseek_v41_expert_w2_k512_gaudi2_o_end);
        }
        return operation==158?loadElf(output,&_binary___deepseek_v41_expert_w13_split_scale_gaudi2_o_start,
            &_binary___deepseek_v41_expert_w13_split_scale_gaudi2_o_end):
            loadElf(output,&_binary___deepseek_v41_expert_w2_split_scale_gaudi2_o_start,
            &_binary___deepseek_v41_expert_w2_split_scale_gaudi2_o_end);
    }
    if(operation==157) {
        // Same C1 binary, accurate route-local scalar reads. The legacy GUID
        // and its access map remain unchanged for existing replay plans.
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_silu_decode_fp8_gaudi2");
        const auto status=instantiate_private(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        const auto slots=input->inputTensors[0].geometry.maxSizes[0];
        const unsigned route_tile=slots==3?3:2;
        auto& scale=output->inputTensorAccessPattern[5];scale={};
        scale.mapping[0]={0,0,0,0,false};
        scale.mapping[1]={1,float(route_tile),0,float(route_tile-1),false};
        return status;
    }
    if(operation==155) {
        if(input->inputTensorNr!=1 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[2];
        if(rows<1 || rows>6 || !shape(input->inputTensors[0],DATA_F32,{40,2,rows}) ||
           !shape(input->outputTensors[0],DATA_F32,{5,rows}) || !input->nodeParams.nodeParams ||
           input->nodeParams.nodeParamsSize!=8)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto* p=static_cast<const float*>(input->nodeParams.nodeParams);
        if(p[0]<=0 || p[1]!=1.f/5120)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        auto& a=output->inputTensorAccessPattern[0];
        a.mapping[0]={0,0,0,39,false};a.mapping[1]={0,0,0,1,false};a.mapping[2]={0,1,0,0,false};
        auto& o=output->outputTensorAccessPattern[0];
        o.mapping[0]={0,0,0,4,false};o.mapping[1]={0,1,0,0,false};
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,p,8);
        return loadElf(output,&_binary___deepseek_v41_weighted_row_parameters_gaudi2_o_start,
                             &_binary___deepseek_v41_weighted_row_parameters_gaudi2_o_end);
    }
    if(operation==156) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(!shape(input->inputTensors[2],DATA_F32,{5,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto inherited=*input;
        std::vector<Tensor> ins(input->inputTensors,input->inputTensors+3);
        ins[2].geometry.dims=3;ins[2].geometry.maxSizes[0]=ins[2].geometry.minSizes[0]=40;
        ins[2].geometry.maxSizes[1]=ins[2].geometry.minSizes[1]=2;
        ins[2].geometry.maxSizes[2]=ins[2].geometry.minSizes[2]=rows;inherited.inputTensors=ins.data();
        const auto capacity=output->kernel.elfSize;
        const auto status=DeepseekV41MhcWeightedStatsGaudi2(true).GetGcDefinitions(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        auto& a=output->inputTensorAccessPattern[2];a={};
        a.mapping[0]={0,0,0,4,false};a.mapping[1]={1,1,0,0,false};
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_norm_prepared_weighted_stats_gaudi2_o_start,
                             &_binary___deepseek_v41_norm_prepared_weighted_stats_gaudi2_o_end);
    }
    if(operation==153 || operation==154)
        return DeepseekV41MhcWeightedStatsGaudi2(operation==154).GetGcDefinitions(input,output);
    if(operation==152) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        if(!shape(input->outputTensors[0],DATA_BF16,{512,640,tokens}) ||
           !shape(input->outputTensors[1],DATA_F32,{640,tokens}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto inherited=*input;
        std::vector<Tensor> outs{input->outputTensors[0],input->outputTensors[0],
                                 input->outputTensors[1],input->outputTensors[0]};
        outs[1].geometry.dataType=DATA_F32;
        inherited.outputTensors=outs.data();inherited.outputTensorNr=4;
        TensorAccessPattern inherited_access[4]{};
        auto adapted=*output;adapted.outputTensorAccessPattern=inherited_access;
        const auto capacity=output->kernel.elfSize;
        const auto status=DeepseekV41BatchedMainPublishGatherGaudi2().GetGcDefinitions(&inherited,&adapted);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        auto* destination=output->outputTensorAccessPattern;
        *output=adapted;output->outputTensorAccessPattern=destination;
        destination[0]=inherited_access[0];destination[1]=inherited_access[2];
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_main_single_bank_gather_gaudi2_o_start,
                             &_binary___deepseek_v41_main_single_bank_gather_gaudi2_o_end);
    }
    if(operation==150) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<2 || rows>6 || !shape(input->inputTensors[0],DATA_BF16,{20480,rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{160,2,rows}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{20480,rows*2}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,rows*2}) ||
           !shape(input->outputTensors[2],DATA_F32,{1,rows}) ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=8)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=40;output->indexSpaceGeometry[1]=rows;
        auto& x=output->inputTensorAccessPattern[0];
        x.mapping[0]={0,512,0,511,false};x.mapping[1]={1,1,0,0,false};
        auto& stats=output->inputTensorAccessPattern[1];
        stats.mapping[0]={0,0,0,159,false};stats.mapping[1]={0,0,0,1,false};
        stats.mapping[2]={1,1,0,0,false};
        auto& q=output->outputTensorAccessPattern[0];
        q.mapping[0]={0,512,0,511,false};q.mapping[1]={1,1,0,float(rows),false};q.sparseAccess=true;
        for(unsigned i=1;i<3;++i) {
            auto& scalar=output->outputTensorAccessPattern[i];
            scalar.mapping[0]={0,0,0,0,false};
            scalar.mapping[1]={1,1,0,float(i==1?rows:0),false};scalar.sparseAccess=true;
        }
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,8);
        return loadElf(output,&_binary___deepseek_v41_control_pair_quant_rrms_gaudi2_o_start,
                             &_binary___deepseek_v41_control_pair_quant_rrms_gaudi2_o_end);
    }
    if(operation==151) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->outputTensors[0].geometry.maxSizes[1];
        if(rows<2 || rows>6 || !shape(input->outputTensors[0],DATA_F32,{24,rows}) ||
           !shape(input->inputTensors[0],DATA_F32,{48,rows*2}) ||
           !shape(input->inputTensors[1],DATA_F32,{1,rows*2}) ||
           !shape(input->inputTensors[2],DATA_F32,{48,1}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,0,0,float(i==0?47:0),false};
            ap.mapping[1]={0,1,0,float(rows),false};ap.sparseAccess=true;
        }
        output->inputTensorAccessPattern[2].allRequired=true;
        auto& y=output->outputTensorAccessPattern[0];
        y.mapping[0]={0,0,0,23,false};y.mapping[1]={0,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_control_pair_finish_gaudi2_o_start,
                             &_binary___deepseek_v41_control_pair_finish_gaudi2_o_end);
    }
    if(operation==145 || operation==147) {
        const unsigned outputs=operation==147?1u:2u;
        if(input->inputTensorNr!=2 || input->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto tokens=input->inputTensors[1].geometry.maxSizes[0];
        if(tokens<2 || tokens>6 || !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
           !shape(input->inputTensors[1],DATA_I32,{tokens}) ||
           !shape(input->outputTensors[0],DATA_BF16,{512,192}) ||
           (outputs==2 && !shape(input->outputTensors[1],DATA_F32,{512,192})))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=4;output->indexSpaceGeometry[1]=192;
        output->inputTensorAccessPattern[0].allRequired=true;
        output->inputTensorAccessPattern[0].sparseAccess=true;
        output->inputTensorAccessPattern[1].allRequired=true;
        for(unsigned i=0;i<outputs;++i) {
            auto& ap=output->outputTensorAccessPattern[i];
            ap.mapping[0]={0,128,0,127,false};ap.mapping[1]={1,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return operation==147?loadElf(output,&_binary___deepseek_v41_coherent_swa_keys_gaudi2_o_start,
                                            &_binary___deepseek_v41_coherent_swa_keys_gaudi2_o_end):
                              loadElf(output,&_binary___deepseek_v41_coherent_swa_decode_gaudi2_o_start,
                                             &_binary___deepseek_v41_coherent_swa_decode_gaudi2_o_end);
    }
    if(operation==146 || operation==148) {
        const bool pair=operation==148;
        if(input->inputTensorNr!=7 || input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& q=input->inputTensors[0].geometry;
        const auto heads=q.maxSizes[1],tokens=q.maxSizes[2];
        if(heads<1 || heads>64 || tokens<2 || tokens>6 ||
           !shape(input->inputTensors[0],DATA_F32,{192,heads,tokens}) ||
           !shape(input->inputTensors[1],DATA_F32,{512,heads,tokens}) ||
           !shape(input->inputTensors[2],DATA_F32,{640,tokens}) ||
           !shape(input->inputTensors[3],DATA_F32,{heads}) ||
           !shape(input->inputTensors[4],DATA_F32,{1}) ||
           !shape(input->inputTensors[5],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[6],DATA_I32,{tokens}) ||
           !shape(input->outputTensors[0],pair?DATA_BF16:DATA_F32,{192,heads*(pair?2:1),tokens}) ||
           !shape(input->outputTensors[1],DATA_F32,{512,heads,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=heads;output->indexSpaceGeometry[1]=tokens;
        for(unsigned i:{0u,1u}) {
            auto& ip=output->inputTensorAccessPattern[i];ip.mapping[0]={0,0,0,float(i==0?191:511),false};
            ip.mapping[1]={0,1,0,0,false};ip.mapping[2]={1,1,0,0,false};
            auto& op=output->outputTensorAccessPattern[i];op.mapping[0]=ip.mapping[0];
            op.mapping[1]=ip.mapping[1];op.mapping[2]=ip.mapping[2];
            if(pair && i==0)op.mapping[1]={0,2,0,1,false};
        }
        auto& mask=output->inputTensorAccessPattern[2];mask.mapping[0]={0,0,128,639,false};
        mask.mapping[1]={1,1,0,0,false};
        output->inputTensorAccessPattern[3].mapping[0]={0,1,0,0,false};
        output->inputTensorAccessPattern[4].allRequired=true;
        output->inputTensorAccessPattern[5].allRequired=true;
        output->inputTensorAccessPattern[6].mapping[0]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return pair?loadElf(output,&_binary___deepseek_v41_coherent_swa_pair_softmax_gaudi2_o_start,
                                  &_binary___deepseek_v41_coherent_swa_pair_softmax_gaudi2_o_end):
                    loadElf(output,&_binary___deepseek_v41_coherent_swa_softmax_gaudi2_o_start,
                                   &_binary___deepseek_v41_coherent_swa_softmax_gaudi2_o_end);
    }
    if(operation==149) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& g=input->outputTensors[0].geometry;
        const auto heads=g.maxSizes[1],tokens=g.maxSizes[2];
        if(heads<1 || heads>64 || tokens<2 || tokens>6 ||
           !shape(input->inputTensors[0],DATA_F32,{512,heads*2,tokens}) ||
           !shape(input->inputTensors[1],DATA_F32,{512,heads,tokens}) ||
           !shape(input->outputTensors[0],DATA_BF16,{512,heads,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;output->indexSpaceGeometry[0]=4;
        output->indexSpaceGeometry[1]=heads;output->indexSpaceGeometry[2]=tokens;
        for(unsigned i:{0u,1u}) {
            auto& ip=output->inputTensorAccessPattern[i];
            ip.mapping[0]={0,128,0,127,false};ip.mapping[1]={1,float(i==0?2:1),0,float(i==0?1:0),false};
            ip.mapping[2]={2,1,0,0,false};
        }
        auto& op=output->outputTensorAccessPattern[0];op.mapping[0]={0,128,0,127,false};
        op.mapping[1]={1,1,0,0,false};op.mapping[2]={2,1,0,0,false};output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_coherent_swa_finish_gaudi2_o_start,
                             &_binary___deepseek_v41_coherent_swa_finish_gaudi2_o_end);
    }
    if(operation==144) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const unsigned capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_attention_norm_quant_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_attention_norm_quant_remat_gaudi2_o_start,
                             &_binary___deepseek_v41_attention_norm_quant_remat_gaudi2_o_end);
    }
    if(operation==129) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& product=input->inputTensors[0].geometry;
        const auto columns=input->inputTensors[3].geometry.maxSizes[1]*256;
        if(product.dims!=3 || product.maxSizes[0]!=columns+512 || product.maxSizes[1]!=1 ||
           product.maxSizes[2]<2 || product.maxSizes[2]>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        auto inherited=*input;
        std::vector<Tensor> validation(input->inputTensors,input->inputTensors+input->inputTensorNr);
        inherited.inputTensors=validation.data();
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_shared_silu_quant_gaudi2");
        // The unchanged kernel derives its loop width from the output and
        // channel plane. Only its validation geometry needs the shared prefix;
        // actual tensor addressing retains the complete producer stride.
        inherited.inputTensors[0].geometry.maxSizes[0]=columns;
        inherited.inputTensors[0].geometry.minSizes[0]=columns;
        return next(&inherited,output);
    }
    if(operation==142) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto pairs=input->outputTensors[0].geometry.maxSizes[2];
        const auto columns=input->outputTensors[0].geometry.maxSizes[0];
        const auto planes=input->inputTensors[1].geometry.maxSizes[1];
        if(pairs<6 || pairs>18 || pairs%3 || columns!=planes*512 ||
           !shape(input->inputTensors[0],DATA_I32,{pairs*2,1}) ||
           !shape(input->inputTensors[1],DATA_BF16,{256,planes,384}) ||
           !shape(input->outputTensors[0],DATA_F32,{columns,1,pairs}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=columns/128;
        output->indexSpaceGeometry[1]=pairs;
        auto& ids=output->inputTensorAccessPattern[0];
        ids.mapping[0]={1,2,0,1,false};ids.mapping[1]={0,0,0,0,false};
        output->inputTensorAccessPattern[1].allRequired=true;
        output->inputTensorAccessPattern[1].sparseAccess=true;
        auto& y=output->outputTensorAccessPattern[0];
        y.mapping[0]={0,128,0,127,false};y.mapping[1]={0,0,0,0,false};y.mapping[2]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_expert_pair_channels_gaudi2_o_start,
                             &_binary___deepseek_v41_expert_pair_channels_gaudi2_o_end);
    }
    if(operation==143) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=2 ||
           input->inputTensors[0].geometry.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const unsigned capacity=output->kernel.elfSize;
        auto inherited=*input;
        // Kernel instantiation must not change the caller-owned type array.
        std::vector<Tensor> tensors(input->inputTensors,input->inputTensors+input->inputTensorNr);
        inherited.inputTensors=tensors.data();
        inherited.inputTensors[0].geometry.dataType=DATA_F32;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_scaled_silu_quant_gaudi2_o_start,
                             &_binary___deepseek_v41_expert_scaled_silu_quant_gaudi2_o_end);
    }
    if(operation==140 || operation==141) {
        const bool emit=operation==141;
        if(input->inputTensorNr!=(emit?2u:1u) || input->outputTensorNr!=(emit?3u:1u))
            return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<2 || rows>6 || !shape(input->inputTensors[0],DATA_BF16,{20480,rows}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=emit?40:160;
        output->indexSpaceGeometry[1]=rows;
        auto& x=output->inputTensorAccessPattern[0];
        x.mapping[0]={0,emit?512.f:128.f,0,emit?511.f:127.f,false};x.mapping[1]={1,1,0,0,false};
        if(!emit) {
            if(!shape(input->outputTensors[0],DATA_F32,{160,2,rows}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            auto& y=output->outputTensorAccessPattern[0];
            y.mapping[0]={0,1,0,0,false};y.mapping[1]={0,0,0,1,false};y.mapping[2]={1,1,0,0,false};
            output->kernel.paramsNr=0;
            return loadElf(output,&_binary___deepseek_v41_control_statistics_gaudi2_o_start,
                                 &_binary___deepseek_v41_control_statistics_gaudi2_o_end);
        }
        if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=8 ||
           !shape(input->inputTensors[1],DATA_F32,{160,2,rows}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{20480,rows}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,rows}) ||
           !shape(input->outputTensors[2],DATA_F32,{1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto& stats=output->inputTensorAccessPattern[1];
        stats.mapping[0]={0,0,0,159,false};stats.mapping[1]={0,0,0,1,false};
        stats.mapping[2]={1,1,0,0,false};
        auto& q=output->outputTensorAccessPattern[0];
        q.mapping[0]={0,512,0,511,false};q.mapping[1]={1,1,0,0,false};
        for(unsigned i=1;i<3;++i) {
            auto& scalar=output->outputTensorAccessPattern[i];
            scalar.mapping[0]={0,0,0,0,false};scalar.mapping[1]={1,1,0,0,false};scalar.sparseAccess=true;
        }
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,8);
        return loadElf(output,&_binary___deepseek_v41_control_quant_rrms_gaudi2_o_start,
                             &_binary___deepseek_v41_control_quant_rrms_gaudi2_o_end);
    }
    if(operation==139) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_token_wide_explicit_steps_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_token_wide_explicit_steps_gaudi2_o_end);
    }
    if(operation==138) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=3 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(float))return GLUE_FAILED;
        const auto dims=input->inputTensors[0].geometry.dims;
        const auto tokens=input->inputTensors[0].geometry.maxSizes[1];
        const auto ranks=dims==3?input->inputTensors[0].geometry.maxSizes[2]:1;
        if(tokens<1 || tokens>6 || (dims!=2 && dims!=3) || (dims==3 && (ranks<2 || ranks>8)))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const bool value_ok=dims==2?shape(input->inputTensors[0],DATA_BF16,{5120,tokens}):
            shape(input->inputTensors[0],DATA_BF16,{5120,tokens,ranks});
        if(!value_ok || !shape(input->inputTensors[1],DATA_BF16,{5120,4,tokens}) ||
           !shape(input->inputTensors[2],DATA_F32,{25,tokens}) ||
           !shape(input->inputTensors[3],DATA_F32,{3}) || !shape(input->inputTensors[4],DATA_F32,{24}) ||
           !shape(input->outputTensors[0],DATA_BF16,{5120,4,tokens}) ||
           !shape(input->outputTensors[1],DATA_BF16,{5120,tokens}) ||
           !shape(input->outputTensors[2],DATA_F32,{24,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=3;i<5;++i)output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i:{0u,1u}) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,5120,0,5119,false};
            if(i==1)ap.mapping[1]={0,0,0,3,false};
            ap.mapping[i==1?2:1]={1,1,0,0,false};
            if(i==0 && dims==3)ap.mapping[2]={0,0,0,float(ranks-1),false};
        }
        auto& raw=output->inputTensorAccessPattern[2];
        raw.mapping[0]={0,0,0,24,false};raw.mapping[1]={1,1,0,0,false};
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->outputTensorAccessPattern[i];ap.mapping[0]={0,5120,0,5119,false};
            if(i==0)ap.mapping[1]={0,0,0,3,false};
            ap.mapping[i==0?2:1]={1,1,0,0,false};
        }
        auto& gate=output->outputTensorAccessPattern[2];
        gate.mapping[0]={0,0,0,23,false};gate.mapping[1]={1,1,0,0,false};
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=1;output->indexSpaceGeometry[1]=tokens;
        std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,sizeof(float));
        output->kernel.paramsNr=1;
        return loadElf(output,&_binary___deepseek_v41_mhc_deferred_post_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_deferred_post_gaudi2_o_end);
    }
    if(operation==137) {
        if(input->inputTensorNr!=8 || input->outputTensorNr!=3)return GLUE_FAILED;
        const auto& ids=input->inputTensors[0].geometry;
        const auto& q=input->inputTensors[1].geometry;
        const auto& scales=input->inputTensors[2].geometry;
        if(ids.dims!=2 || ids.dataType!=DATA_I32 || ids.maxSizes[1]!=1 || ids.maxSizes[0]<3 ||
           ids.maxSizes[0]>36 || (ids.maxSizes[0]%2 && ids.maxSizes[0]!=3) ||
           q.dims!=3 || q.dataType!=DATA_I16 || !q.maxSizes[0] || q.maxSizes[0]%8192 ||
           !q.maxSizes[1] || q.maxSizes[1]>20 || !q.maxSizes[2] || q.maxSizes[2]>384 ||
           scales.dims!=3 || scales.dataType!=DATA_I16 || scales.maxSizes[1]!=q.maxSizes[1] ||
           scales.maxSizes[2]!=q.maxSizes[2] ||
           (scales.maxSizes[0]!=q.maxSizes[0]/16+128 && scales.maxSizes[0]!=q.maxSizes[0]/8) ||
           !shape(input->inputTensors[3],DATA_BF16,{128}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto n=q.maxSizes[1]*256,k=q.maxSizes[0]/64,slots=ids.maxSizes[0];
        const unsigned route_tile=slots==3?3:2;
        if(!shape(input->inputTensors[4],DATA_F32,{k*2,1,slots}) ||
           !shape(input->inputTensors[5],DATA_F32,{1,slots}) ||
           !shape(input->inputTensors[6],DATA_BF16,{256,(k*2)/256,q.maxSizes[2]}) ||
           !shape(input->inputTensors[7],DATA_F32,{slots,1}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{n,k,slots}) ||
           !shape(input->outputTensors[1],DATA_F8_143,{k,1,slots}) ||
           !shape(input->outputTensors[2],DATA_F32,{1,1,slots}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;output->indexSpaceGeometry[0]=n/256;
        output->indexSpaceGeometry[1]=slots/route_tile;output->indexSpaceGeometry[2]=k/128;
        for(unsigned i:{1u,2u,3u,5u,6u})output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i:{0u,7u}) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={1,float(route_tile),0,float(route_tile-1),false};
            ap.mapping[1]={0,0,0,0,false};
        }
        auto& p=output->inputTensorAccessPattern[4];
        p.mapping[0]={0,0,0,float(k*2-1),false};p.mapping[1]={0,0,0,0,false};
        p.mapping[2]={1,float(route_tile),0,float(route_tile-1),false};
        auto& weight=output->outputTensorAccessPattern[0];
        weight.mapping[0]={0,256,0,255,false};weight.mapping[1]={2,128,0,127,false};
        weight.mapping[2]={1,float(route_tile),0,float(route_tile-1),false};
        for(unsigned i:{1u,2u}) {
            auto& ap=output->outputTensorAccessPattern[i];
            ap.mapping[0]={0,0,0,float(i==1?k-1:0),false};ap.mapping[1]={0,0,0,0,false};
            ap.mapping[2]={1,float(route_tile),0,float(route_tile-1),false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_silu_decode_c6_gaudi2_o_start,
                             &_binary___deepseek_v41_silu_decode_c6_gaudi2_o_end);
    }
    if(operation==136) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=2 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
        const auto columns=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<1 || rows>6 || (columns!=1280 && columns!=5120) ||
           !shape(input->inputTensors[0],DATA_BF16,{columns,rows}) ||
           !shape(input->inputTensors[1],DATA_BF16,{columns}) ||
           !shape(input->outputTensors[0],DATA_BF16,{columns,rows}) ||
           !shape(input->outputTensors[1],DATA_BF16,{columns,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=(columns+511)/512;
        output->indexSpaceGeometry[1]=rows;
        auto& x=output->inputTensorAccessPattern[0];
        x.mapping[0]={0,0,0,float(columns-1),false};x.mapping[1]={1,1,0,0,false};
        output->inputTensorAccessPattern[1].mapping[0]={0,512,0,511,false};
        for(int i=0;i<2;++i) {
            auto& ap=output->outputTensorAccessPattern[i];
            ap.mapping[0]={0,512,0,511,false};ap.mapping[1]={1,1,0,0,false};
        }
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,8);
        return loadElf(output,&_binary___deepseek_v41_norm_roundtrip_tiled_gaudi2_o_start,
                             &_binary___deepseek_v41_norm_roundtrip_tiled_gaudi2_o_end);
    }
    if(operation==134 || operation==135) {
        const bool statistics=operation==134;
        if(input->inputTensorNr!=(statistics?1:3) || input->outputTensorNr!=(statistics?1:2))return GLUE_FAILED;
        const auto columns=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<1 || rows>6 || (columns!=1280 && columns!=5120) ||
           !shape(input->inputTensors[0],DATA_BF16,{columns,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(statistics) {
            if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=8 ||
               !shape(input->outputTensors[0],DATA_F32,{1,rows}))return GLUE_FAILED;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
            auto& input_row=output->inputTensorAccessPattern[0];
            input_row.mapping[0]={0,0,0,float(columns-1),false};
            input_row.mapping[1]={0,1,0,0,false};
            auto& ap=output->outputTensorAccessPattern[0];
            ap.mapping[0]={0,0,0,0,false};ap.mapping[1]={0,1,0,0,false};
            output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,8);
            return loadElf(output,&_binary___deepseek_v41_norm_statistics_gaudi2_o_start,
                                 &_binary___deepseek_v41_norm_statistics_gaudi2_o_end);
        }
        if(!shape(input->inputTensors[1],DATA_BF16,{columns}) ||
           !shape(input->inputTensors[2],DATA_F32,{1,rows}) ||
           !shape(input->outputTensors[0],DATA_BF16,{columns,rows}) ||
           !shape(input->outputTensors[1],DATA_BF16,{columns,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=columns/64;output->indexSpaceGeometry[1]=rows;
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,64,0,63,false};
            if(i==0)ap.mapping[1]={1,1,0,0,false};
        }
        auto& rrms=output->inputTensorAccessPattern[2];
        rrms.mapping[0]={0,0,0,0,false};rrms.mapping[1]={1,1,0,0,false};
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->outputTensorAccessPattern[i];
            ap.mapping[0]={0,64,0,63,false};ap.mapping[1]={1,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_norm_roundtrip_gaudi2_o_start,
                             &_binary___deepseek_v41_norm_roundtrip_gaudi2_o_end);
    }
    if(operation==132 || operation==133) {
        const bool draw=operation==133;
        if(input->inputTensorNr!=(draw?3:1) || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto columns=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto parts=(columns+2047)/2048;
        if(rows<1 || rows>6 || columns<64 || columns>131072 || columns%64 ||
           !shape(input->inputTensors[0],DATA_F32,{columns,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(draw) {
            if(!shape(input->inputTensors[1],DATA_F32,{parts,rows}) ||
               !shape(input->inputTensors[2],DATA_F32,{4,rows}) ||
               !shape(input->outputTensors[0],DATA_I32,{2,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
            for(unsigned i=0;i<3;++i)output->inputTensorAccessPattern[i].allRequired=true;
            auto& ap=output->outputTensorAccessPattern[0];
            ap.mapping[0]={0,0,0,1,false};ap.mapping[1]={0,1,0,0,false};
            output->kernel.paramsNr=0;
            return loadElf(output,&_binary___deepseek_v41_probability_draw_gaudi2_o_start,
                                 &_binary___deepseek_v41_probability_draw_gaudi2_o_end);
        }
        if(!shape(input->outputTensors[0],DATA_F32,{parts,rows}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=parts;output->indexSpaceGeometry[1]=rows;
        auto& read=output->inputTensorAccessPattern[0];
        read.mapping[0]={0,2048,0,2047,false};read.mapping[1]={1,1,0,0,false};
        auto& write=output->outputTensorAccessPattern[0];
        write.mapping[0]={0,1,0,0,false};write.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_probability_parts_gaudi2_o_start,
                             &_binary___deepseek_v41_probability_parts_gaudi2_o_end);
    }
    if(operation==131) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1) return GLUE_FAILED;
        const auto n=input->inputTensors[1].geometry.maxSizes[0];
        const auto k=input->inputTensors[1].geometry.maxSizes[1];
        const auto experts=input->inputTensors[1].geometry.maxSizes[2];
        const auto slots=input->inputTensors[0].geometry.maxSizes[0];
        if(!n || n%128 || !k || k%32 || !experts || experts>128 || !slots || slots>18 ||
           !shape(input->inputTensors[0],DATA_I32,{slots,1}) ||
           !shape(input->inputTensors[1],DATA_BF16,{n,k,experts}) ||
           !shape(input->outputTensors[0],DATA_BF16,{n,k,slots}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;
        output->indexSpaceGeometry[0]=n/128;output->indexSpaceGeometry[1]=k/32;
        output->indexSpaceGeometry[2]=slots;
        output->inputTensorAccessPattern[0].mapping[0]={2,1,0,0,false};
        auto& bank=output->inputTensorAccessPattern[1];
        bank.mapping[0]={0,128,0,127,false};bank.mapping[1]={1,32,0,31,false};
        bank.mapping[2]={2,0,0,float(experts-1),false};
        auto& dst=output->outputTensorAccessPattern[0];
        dst.mapping[0]={0,128,0,127,false};dst.mapping[1]={1,32,0,31,false};
        dst.mapping[2]={2,1,0,0,false};
        return loadElf(output,&_binary___deepseek_v41_expert_cached_gather_bf16_gaudi2_o_start,
                              &_binary___deepseek_v41_expert_cached_gather_bf16_gaudi2_o_end);
    }
    if(operation==130) {
        if(input->inputTensorNr!=1 || input->outputTensorNr!=1 || (!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(float)))
            return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& g=input->inputTensors[0].geometry;
        const auto width=g.maxSizes[0],rows=g.maxSizes[1];
        if(g.dataType!=DATA_BF16 || g.dims!=2 || (width!=1280 && width!=5120) || rows<2 || rows>6)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(input->outputTensors[0],DATA_F8_143,{width,rows}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=width/32;output->indexSpaceGeometry[1]=rows;
        for(auto* p:{&output->inputTensorAccessPattern[0],&output->outputTensorAccessPattern[0]}) {
            p->mapping[0]={0,32,0,31,false};p->mapping[1]={1,1,0,0,false};
        }
        output->kernel.paramsNr=1;
        std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,sizeof(float));
        return loadElf(output,&_binary___deepseek_v41_fixed_dense_quant_gaudi2_o_start,
                              &_binary___deepseek_v41_fixed_dense_quant_gaudi2_o_end);
    }
    if(operation==128) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& product=input->inputTensors[0].geometry;
        const auto rows=product.maxSizes[1],columns=product.maxSizes[0];
        if(product.dataType!=DATA_F32 || product.dims!=2 || rows<2 || rows>6 ||
           (columns!=1792 && columns!=3072))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=1;i<4;++i) {
            if(!shape(input->inputTensors[i],i==3?DATA_I8:DATA_F32,{i==3?rows:384u}))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        if(!shape(input->inputTensors[4],DATA_F32,{384,1}) ||
           !shape(input->inputTensors[5],DATA_F32,{1,rows}) ||
           !shape(input->outputTensors[0],DATA_I32,{6,rows}) ||
           !shape(input->outputTensors[1],DATA_F32,{6,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        auto& p=output->inputTensorAccessPattern[0];
        p.mapping[0]={0,0,float(columns-512),float(columns-129),false};
        p.mapping[1]={0,1,0,0,false};
        for(unsigned i=1;i<3;++i)output->inputTensorAccessPattern[i].mapping[0]={0,0,0,383,false};
        output->inputTensorAccessPattern[3].mapping[0]={0,1,0,0,false};
        output->inputTensorAccessPattern[4].mapping[0]={0,0,0,383,false};
        output->inputTensorAccessPattern[4].mapping[1]={0,0,0,0,false};
        output->inputTensorAccessPattern[5].mapping[0]={0,0,0,0,false};
        output->inputTensorAccessPattern[5].mapping[1]={0,1,0,0,false};
        for(unsigned i=0;i<2;++i) {
            output->outputTensorAccessPattern[i].mapping[0]={0,0,0,5,false};
            output->outputTensorAccessPattern[i].mapping[1]={0,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_router_shared_scaled_gaudi2_o_start,
                              &_binary___deepseek_v41_router_shared_scaled_gaudi2_o_end);
    }
    if(operation==126 || operation==127) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[2];
        if(rows<2 || rows>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,operation==126 ? "custom_deepseek_v41_rope_bf16_gaudi2" :
                                                    "custom_deepseek_v41_rope_inverse_bf16_gaudi2");
        const auto status=next(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        // One head owns every C2-C6 row, matching the actual kernel stores.
        // Distinct head workpoints have disjoint footprints; unlike the
        // conservative row-point variant, no overlapping row writes remain.
        output->indexSpaceGeometry[2]=1;
        output->inputTensorAccessPattern[0].mapping[2]={2,0,0,float(rows-1),false};
        output->outputTensorAccessPattern[0].mapping[2]={2,0,0,float(rows-1),false};
        output->inputTensorAccessPattern[1].mapping[0]={2,0,0,float(rows-1),false};
        output->kernel.elfSize=capacity;
        return operation==126 ?
            loadElf(output,&_binary___deepseek_v41_rope_all_rows_gaudi2_o_start,
                           &_binary___deepseek_v41_rope_all_rows_gaudi2_o_end) :
            loadElf(output,&_binary___deepseek_v41_rope_inverse_all_rows_gaudi2_o_start,
                           &_binary___deepseek_v41_rope_inverse_all_rows_gaudi2_o_end);
    }
    if(operation==124) {
        return dsv41_route_pack::instantiate(input,output,2,
            &_binary___deepseek_v41_expert_token_wide_n512_gaudi2_o_start,
            &_binary___deepseek_v41_expert_token_wide_n512_gaudi2_o_end,2);
    }
    if(operation==125) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        auto inherited=*input;std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        const auto capacity=output->kernel.elfSize;
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        if(output->indexSpaceGeometry[0]%2)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceGeometry[0]/=2;
        for(unsigned i:{1u,2u}) {
            auto& map=output->inputTensorAccessPattern[i].mapping[1];map.a=2;map.end_b=1;
        }
        auto& map=output->outputTensorAccessPattern[0].mapping[0];map.a=512;map.end_b=511;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_down_n512_gaudi2_o_start,
                             &_binary___deepseek_v41_expert_down_n512_gaudi2_o_end);
    }
    if(operation==122) {
        if(input->inputTensorNr!=1 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<2 || rows>6 || !shape(input->inputTensors[0],DATA_BF16,{20480,rows}) ||
           !shape(input->outputTensors[0],DATA_F32,{40,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=40;output->indexSpaceGeometry[1]=rows;
        auto& x=output->inputTensorAccessPattern[0];
        x.mapping[0]={0,512,0,511,false};x.mapping[1]={1,1,0,0,false};
        auto& y=output->outputTensorAccessPattern[0];
        y.mapping[0]={0,1,0,0,false};y.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mhc_statistics_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_statistics_gaudi2_o_end);
    }
    if(operation==123) {
        if(input->inputTensorNr!=4 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto width=input->inputTensors[0].geometry.maxSizes[0];
        if(rows<2 || rows>6 || (width!=24 && width!=48) ||
           !shape(input->inputTensors[0],DATA_F32,{width,rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{40,rows}) ||
           !shape(input->inputTensors[2],DATA_F32,{3}) ||
           !shape(input->inputTensors[3],DATA_F32,{24}) ||
           !shape(input->outputTensors[0],DATA_F32,{24,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;output->kernel.paramsNr=0;
        for(unsigned i=0;i<2;++i) {
            auto& x=output->inputTensorAccessPattern[i];
            x.mapping[0]={0,0,0,float(input->inputTensors[i].geometry.maxSizes[0]-1),false};
            x.mapping[1]={0,1,0,0,false};
        }
        output->inputTensorAccessPattern[2].allRequired=true;
        output->inputTensorAccessPattern[3].allRequired=true;
        auto& y=output->outputTensorAccessPattern[0];
        y.mapping[0]={0,0,0,23,false};y.mapping[1]={0,1,0,0,false};
        return loadElf(output,&_binary___deepseek_v41_mhc_statistics_finish_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_statistics_finish_gaudi2_o_end);
    }
    if(operation==121) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<1 || rows>6 || !shape(input->inputTensors[0],DATA_BF16,{5120,rows}) ||
           !shape(input->inputTensors[1],DATA_BF16,{5120}) ||
           !shape(input->outputTensors[0],DATA_BF16,{5120,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_attention_norm_bf16_gaudi2");
        const auto capacity=output->kernel.elfSize;
        const auto status=next(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_input_norm_bf16_gaudi2_o_start,
                             &_binary___deepseek_v41_input_norm_bf16_gaudi2_o_end);
    }
    if(operation==120) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto count=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[4].geometry.maxSizes[1];
        const auto vocab=input->inputTensors[2].geometry.maxSizes[0];
        const auto heads=input->outputTensors[0].geometry.maxSizes[1];
        if(count<2 || count>6 || !rows || rows>INT32_MAX || !vocab || vocab>INT32_MAX ||
           !shape(input->inputTensors[0],DATA_I32,{count,1}) ||
           !shape(input->inputTensors[1],DATA_I32,{3,1}) ||
           !shape(input->inputTensors[2],DATA_I32,{vocab,1}) ||
           !shape(input->inputTensors[3],DATA_I32,{47,1}) ||
           !shape(input->inputTensors[4],DATA_U8,{256,rows}) ||
           !shape(input->inputTensors[5],DATA_U8,{8,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if((heads!=6 && heads!=12) || !shape(input->outputTensors[0],DATA_BF16,{256,heads,count}) ||
           !shape(input->outputTensors[1],DATA_I32,{3,count+1}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=heads;output->indexSpaceGeometry[1]=count;
        for(unsigned i=0;i<6;++i) {
            auto& ap=output->inputTensorAccessPattern[i];std::memset(&ap,0,sizeof(ap));ap.allRequired=true;
            for(unsigned d=0;d<input->inputTensors[i].geometry.dims;++d)
                ap.mapping[d]={0,0,0,float(input->inputTensors[i].geometry.maxSizes[d]-1),false};
        }
        auto& decoded=output->outputTensorAccessPattern[0];std::memset(&decoded,0,sizeof(decoded));
        decoded.mapping[0]={0,0,0,255,false};decoded.mapping[1]={0,1,0,0,false};
        decoded.mapping[2]={1,1,0,0,false};
        auto& history=output->outputTensorAccessPattern[1];std::memset(&history,0,sizeof(history));
        history.allRequired=true;history.mapping[0]={0,0,0,2,false};history.mapping[1]={1,0,0,float(count),false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_engram_batch_gaudi2_o_start,
                              &_binary___deepseek_v41_engram_batch_gaudi2_o_end);
    }
    if(operation==117 || operation==118 || operation==119) {
        const unsigned ni=operation==117?4u:operation==118?1u:5u;
        const unsigned no=operation==119?2u:1u;
        if(input->inputTensorNr!=ni || input->outputTensorNr!=no)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        if(operation==118) {
            const auto& g=input->outputTensors[0].geometry;
            if(!shape(input->outputTensors[0],DATA_BF16,{512,g.maxSizes[1],g.maxSizes[2]}) ||
               !g.maxSizes[1] || g.maxSizes[1]>64 || !g.maxSizes[2] || g.maxSizes[2]>6 ||
               !shape(input->inputTensors[0],DATA_F32,{512,2*g.maxSizes[1],g.maxSizes[2]}))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=3;output->indexSpaceGeometry[0]=4;
            output->indexSpaceGeometry[1]=g.maxSizes[1];output->indexSpaceGeometry[2]=g.maxSizes[2];
            auto& ip=output->inputTensorAccessPattern[0];
            ip.mapping[0]={0,128,0,127,false};ip.mapping[1]={1,2,0,1,false};ip.mapping[2]={2,1,0,0,false};
            auto& op=output->outputTensorAccessPattern[0];
            op.mapping[0]={0,128,0,127,false};op.mapping[1]={1,1,0,0,false};op.mapping[2]={2,1,0,0,false};
            output->kernel.paramsNr=0;
            return loadElf(output,&_binary___deepseek_v41_mla_adjacent_finish_gaudi2_o_start,
                                  &_binary___deepseek_v41_mla_adjacent_finish_gaudi2_o_end);
        }
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::vector<Tensor> outs(input->outputTensors,input->outputTensors+no);
        TensorAccessPattern original_out[3]{};
        auto adapted=*output;adapted.outputTensorAccessPattern=original_out;
        if(operation==117) {
            const auto& g=input->inputTensors[0].geometry;
            if(!shape(input->outputTensors[0],DATA_BF16,{g.maxSizes[0],2*g.maxSizes[1],g.maxSizes[2]}))
                return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            outs[0].geometry.dataType=DATA_F32;outs[0].geometry.maxSizes[1]=outs[0].geometry.minSizes[1]=g.maxSizes[1];
            inherited.outputTensors=outs.data();
            std::strcpy(inherited.guid.name,"custom_deepseek_v41_selected_mla_softmax_gaudi2");
            auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
            if(!next)return GLUE_FAILED;
            auto status=next(&inherited,&adapted);
            if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
            auto* destination=output->outputTensorAccessPattern;
            *output=adapted;output->outputTensorAccessPattern=destination;destination[0]=original_out[0];
            auto& op=destination[0];op.allRequired=false;op.mapping[1]={0,2,0,1,false};
            output->kernel.elfSize=capacity;
            return loadElf(output,&_binary___deepseek_v41_mla_adjacent_softmax_gaudi2_o_start,
                                  &_binary___deepseek_v41_mla_adjacent_softmax_gaudi2_o_end);
        }
        const auto count=input->inputTensors[3].geometry.maxSizes[0];
        if(!shape(input->outputTensors[0],DATA_BF16,{512,128,count}) ||
           !shape(input->outputTensors[1],DATA_F32,{128,count}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        outs.insert(outs.begin()+1,input->outputTensors[0]);outs[1].geometry.dataType=DATA_F32;
        inherited.outputTensors=outs.data();inherited.outputTensorNr=3;
        auto status=DeepseekV41SwaOnlyReuseGatherGaudi2().GetGcDefinitions(&inherited,&adapted);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        auto* destination=output->outputTensorAccessPattern;
        *output=adapted;output->outputTensorAccessPattern=destination;
        destination[0]=original_out[0];destination[1]=original_out[2];output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_swa_keys_reuse_gaudi2_o_start,
                              &_binary___deepseek_v41_swa_keys_reuse_gaudi2_o_end);
    }
    if(operation==116) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto count=input->inputTensors[3].geometry.maxSizes[0];
        if(count<1 || count>6 || !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
           !shape(input->inputTensors[1],DATA_BF16,{512,640,count}) ||
           !shape(input->inputTensors[2],DATA_F32,{640,count}) ||
           !shape(input->inputTensors[3],DATA_I32,{count}) ||
           !shape(input->inputTensors[4],DATA_I32,{count}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(input->outputTensors[0],DATA_BF16,{512,128,count}) ||
           !shape(input->outputTensors[1],DATA_F32,{512,128,count}) ||
           !shape(input->outputTensors[2],DATA_F32,{128,count}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=256;
        // The scatter is a device-dependent circular permutation. A full
        // output AP is required; consumers retain their original tensors.
        for(unsigned i=0;i<5;++i) {
            auto& ap=output->inputTensorAccessPattern[i];std::memset(&ap,0,sizeof(ap));
            ap.allRequired=true;
            for(unsigned j=0;j<input->inputTensors[i].geometry.dims;++j)
                ap.mapping[j]={0,0,0,float(input->inputTensors[i].geometry.maxSizes[j]-1),false};
        }
        for(unsigned i=0;i<3;++i) {
            auto& ap=output->outputTensorAccessPattern[i];std::memset(&ap,0,sizeof(ap));ap.allRequired=true;
            for(unsigned j=0;j<input->outputTensors[i].geometry.dims;++j)
                ap.mapping[j]={0,0,0,float(input->outputTensors[i].geometry.maxSizes[j]-1),false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_swa_source_reuse_gaudi2_o_start,
                              &_binary___deepseek_v41_swa_source_reuse_gaudi2_o_end);
    }
    if(operation==104)return DeepseekV41SwaOnlyReuseGatherGaudi2().GetGcDefinitions(input,output);
    if(operation==102)return DeepseekV41BatchedMainPublishGatherGaudi2().GetGcDefinitions(input,output);
    if(operation==103)return DeepseekV41BatchedMainReuseGatherGaudi2().GetGcDefinitions(input,output);
    if(operation>=99 && operation<=101) {
        const unsigned expected_inputs=operation==99?3u:operation==100?6u:5u;
        const unsigned expected_outputs=operation==101?1u:3u;
        if(input->inputTensorNr!=expected_inputs || input->outputTensorNr!=expected_outputs)
            return GLUE_INCOMPATIBLE_INPUT_COUNT;
        auto inherited=*input;
        std::vector<Tensor> operands(input->inputTensors,input->inputTensors+expected_inputs);
        std::vector<Tensor> results(input->outputTensors,input->outputTensors+expected_outputs);
        inherited.inputTensors=operands.data();inherited.outputTensors=results.data();
        auto resize=[](Tensor& t,unsigned dim,uint64_t size) {
            t.geometry.maxSizes[dim]=t.geometry.minSizes[dim]=size;
        };
        if(operation==99) {
            if(!shape(results[0],DATA_I32,{256,257}) || !shape(results[1],DATA_I32,{256,257}) ||
               !shape(results[2],DATA_I32,{257}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            for(unsigned i=0;i<2;++i){resize(results[i],0,1024);resize(results[i],1,65);}
            resize(results[2],0,65);
            std::strcpy(inherited.guid.name,"custom_deepseek_v41_mla_compact_metadata_gaudi2");
        } else if(operation==100) {
            if(!shape(operands[2],DATA_I32,{256,257}) || !shape(operands[3],DATA_I32,{256,257}) ||
               !shape(operands[4],DATA_I32,{258}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            for(unsigned i=2;i<4;++i){resize(operands[i],0,1024);resize(operands[i],1,65);}
            resize(operands[4],0,66);
            std::strcpy(inherited.guid.name,"custom_deepseek_v41_mla_compact_decode_gaudi2");
        } else {
            if(!shape(operands[4],DATA_I32,{258}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            resize(operands[4],0,66);
            std::strcpy(inherited.guid.name,"custom_deepseek_v41_mla_compact_softmax_gaudi2");
        }
        const auto capacity=output->kernel.elfSize;
        const auto status=instantiate_private(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        output->kernel.elfSize=capacity;
        if(operation==99) {
            output->indexSpaceGeometry[0]=257;
            for(unsigned i=0;i<2;++i)output->outputTensorAccessPattern[i].mapping[0].end_b=255;
            return loadElf(output,&_binary___deepseek_v41_mla_stream_metadata_gaudi2_o_start,
                           &_binary___deepseek_v41_mla_stream_metadata_gaudi2_o_end);
        }
        if(operation==100) {
            const auto width=input->outputTensors[2].geometry.maxSizes[0];
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=width/8;
            for(unsigned i=0;i<3;++i) {
                auto& ap=output->outputTensorAccessPattern[i];ap={};
                if(i<2){ap.mapping[0]={0,0,0,511,false};ap.mapping[1]={0,8,0,7,false};}
                else ap.mapping[0]={0,8,0,7,false};
            }
            for(unsigned i=0;i<2;++i)output->inputTensorAccessPattern[i].sparseAccess=true;
            return loadElf(output,&_binary___deepseek_v41_mla_stream_decode_gaudi2_o_start,
                           &_binary___deepseek_v41_mla_stream_decode_gaudi2_o_end);
        }
        const auto width=input->outputTensors[0].geometry.maxSizes[0];
        auto& ap=output->outputTensorAccessPattern[0];ap={};
        ap.mapping[0]={0,0,0,float(width-1),false};ap.mapping[1]={0,1,0,0,false};
        return loadElf(output,&_binary___deepseek_v41_mla_stream_softmax_gaudi2_o_start,
                       &_binary___deepseek_v41_mla_stream_softmax_gaudi2_o_end);
    }
    if(operation==98) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto width=input->inputTensors[0].geometry.maxSizes[0];
        const auto tokens=input->inputTensors[0].geometry.maxSizes[1];
        const auto length=input->inputTensors[2].geometry.maxSizes[1];
        if((width!=8192 && width!=16384) || tokens<1 || tokens>64 || !length || length>1048576 ||
           !shape(input->inputTensors[0],DATA_F32,{width,tokens}) ||
           !shape(input->inputTensors[1],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[2],DATA_F32,{64,length}) ||
           !shape(input->outputTensors[0],DATA_BF16,{width,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=width/128;output->indexSpaceGeometry[1]=tokens;
        for(auto* ap : {&output->inputTensorAccessPattern[0],&output->outputTensorAccessPattern[0]}) {
            dsv41_route_pack::map(*ap,0,0,128,0,127);
            dsv41_route_pack::map(*ap,1,1,1,0,0);
        }
        dsv41_route_pack::map(output->inputTensorAccessPattern[1],0,1,1,0,0);
        output->inputTensorAccessPattern[2].allRequired=true;
        output->inputTensorAccessPattern[2].sparseAccess=true;
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_q_bf16_rope_tiled_gaudi2_o_start,
                             &_binary___deepseek_v41_q_bf16_rope_tiled_gaudi2_o_end);
    }
    if(operation==97) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=3)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        if(tokens<2 || tokens>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const unsigned capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        // Geometry, indirect sparse accesses and SRAM slicing match parent.
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_full_row_gaudi2_o_start,
                             &_binary___deepseek_v41_logical_full_row_gaudi2_o_end);
    }
    if(operation==96) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=2 ||
           input->outputTensors[0].geometry.maxSizes[0]!=640)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const unsigned capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_silu_scalar_cache_gaudi2_o_start,
                             &_binary___deepseek_v41_silu_scalar_cache_gaudi2_o_end);
    }
    if(operation==95) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=3)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        if(tokens<2 || tokens>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const unsigned capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        // Geometry, indirect sparse accesses and SRAM slicing match parent.
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_fp4_table_gaudi2_o_start,
                             &_binary___deepseek_v41_logical_fp4_table_gaudi2_o_end);
    }
    if(operation==94) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=3)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        if(tokens<2 || tokens>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const unsigned capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        // Each slot workpoint owns all queries, allowing a bounded VRF row
        // cache to cross query boundaries. Keep the exact per-query outputs.
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=640;output->indexSpaceGeometry[1]=1;
        // Cache pages remain indirect sparse reads, exactly as in the parent
        // gather. Only query metadata spans every query owned by this slot.
        for(unsigned i : {2u,3u,5u}) {
            std::memset(&output->inputTensorAccessPattern[i],0,sizeof(TensorAccessPattern));
            output->inputTensorAccessPattern[i].allRequired=true;
        }
        for(unsigned i=0;i<3;++i) {
            auto& ap=output->outputTensorAccessPattern[i];std::memset(&ap,0,sizeof(ap));
            if(i<2) {
                dsv41_route_pack::map(ap,0,0,0,0,511);
                dsv41_route_pack::map(ap,1,0,1,0,0);
                dsv41_route_pack::map(ap,2,0,0,0,tokens-1);
            } else {
                dsv41_route_pack::map(ap,0,0,1,0,0);
                dsv41_route_pack::map(ap,1,0,0,0,tokens-1);
            }
        }
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_row_cache_gaudi2_o_start,
                             &_binary___deepseek_v41_logical_row_cache_gaudi2_o_end);
    }
    if(operation==93) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const TensorDataType types[]={DATA_F32,DATA_I32,DATA_F32,DATA_BF16,DATA_BF16};
        for(unsigned i=0;i<5;++i)
            if(input->inputTensors[i].geometry.dataType!=types[i])return GLUE_INCOMPATIBLE_DATA_TYPE;
        const auto& p=input->inputTensors[0].geometry;
        const auto& ids=input->inputTensors[1].geometry;
        const auto& sx=input->inputTensors[2].geometry;
        const auto& channel=input->inputTensors[3].geometry;
        const auto& shared=input->inputTensors[4].geometry;
        const uint64_t width=p.maxSizes[0],slots=p.maxSizes[2],tokens=slots/6;
        if(p.dims!=3 || p.maxSizes[1]!=1 || !width || width%256 || width>8192 ||
           slots<6 || slots>36 || slots%6 || ids.dims!=2 || ids.maxSizes[0]!=slots || ids.maxSizes[1]!=1 ||
           sx.dims!=2 || sx.maxSizes[0]!=1 || sx.maxSizes[1]!=slots ||
           channel.dims!=3 || channel.maxSizes[0]!=256 || channel.maxSizes[1]*256!=width ||
           !channel.maxSizes[2] || channel.maxSizes[2]>384 ||
           shared.dims!=3 || shared.maxSizes[0]!=width || shared.maxSizes[1]!=1 || shared.maxSizes[2]!=tokens)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto& dst=input->outputTensors[0].geometry;
        if(dst.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
        if(dst.dims!=3 || dst.maxSizes[0]!=width || dst.maxSizes[1]!=1 || dst.maxSizes[2]!=tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=width/128;
        output->indexSpaceGeometry[1]=tokens;
        dsv41_route_pack::map(output->inputTensorAccessPattern[0],0,0,128,0,127);
        dsv41_route_pack::map(output->inputTensorAccessPattern[0],1,0,0,0,0);
        dsv41_route_pack::map(output->inputTensorAccessPattern[0],2,1,6,0,5);
        dsv41_route_pack::map(output->inputTensorAccessPattern[1],0,1,6,0,5);
        dsv41_route_pack::map(output->inputTensorAccessPattern[1],1,0,0,0,0);
        dsv41_route_pack::map(output->inputTensorAccessPattern[2],0,0,0,0,0);
        dsv41_route_pack::map(output->inputTensorAccessPattern[2],1,1,6,0,5);
        output->inputTensorAccessPattern[3].allRequired=true;
        for(auto* pattern:{&output->inputTensorAccessPattern[4],&output->outputTensorAccessPattern[0]}) {
            dsv41_route_pack::map(*pattern,0,0,128,0,127);dsv41_route_pack::map(*pattern,1,0,0,0,0);dsv41_route_pack::map(*pattern,2,1,1,0,0);
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_scale_reduce_shared_gaudi2_o_start,
                       &_binary___deepseek_v41_scale_reduce_shared_gaudi2_o_end);
    }
    if(operation==114) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        auto inherited=*input;
        std::vector<Tensor> operands(input->inputTensors,input->inputTensors+5);
        std::vector<Tensor> results(input->outputTensors,input->outputTensors+2);
        inherited.inputTensors=operands.data();inherited.outputTensors=results.data();
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto status=next(&inherited,output);
        if(status!=GLUE_SUCCESS)return status;
        // Existing W13 products already cross the bundle in DRAM. Keep the
        // small activation unsliced; W2 weights remain eligible for slicing.
        output->inputTensorAccessPattern[0].allRequired=true;
        for(unsigned i=0;i<2;++i)output->outputTensorAccessPattern[i].allRequired=true;
        return GLUE_SUCCESS;
    }
    if(operation==113) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto rows=input->inputTensors[2].geometry.maxSizes[1];
        if(rows<1 || rows>6 || !shape(input->inputTensors[0],DATA_F32,{512,8}) ||
            !shape(input->inputTensors[1],DATA_F32,{512,8}) ||
            !shape(input->inputTensors[2],DATA_F32,{512,rows}) ||
            !shape(input->inputTensors[3],DATA_F32,{512,rows}) ||
            !shape(input->inputTensors[4],DATA_I32,{rows}) ||
            !shape(input->outputTensors[0],DATA_BF16,{512,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=4;output->indexSpaceGeometry[1]=rows;
        for(unsigned i=0;i<4;++i) {
            auto& a=output->inputTensorAccessPattern[i];
            a.mapping[0]={0,128,0,127,false};
            a.mapping[1]={1,0,0,static_cast<float>((i<2?8:rows)-1),true};
        }
        output->inputTensorAccessPattern[4].allRequired=true;
        auto& a=output->outputTensorAccessPattern[0];
        a.mapping[0]={0,128,0,127,false};a.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_compressor_sequence_bf16_gaudi2_o_start,
                             &_binary___deepseek_v41_compressor_sequence_bf16_gaudi2_o_end);
    }
    if(operation==112) {
        if(input->inputTensorNr!=5 || input->outputTensorNr!=2)return GLUE_FAILED;
        const auto& product=input->inputTensors[0].geometry;
        const auto width=product.maxSizes[0]/4,pairs=product.maxSizes[2],rows=pairs*2;
        if(product.dataType!=DATA_F32 || product.dims!=3 || product.maxSizes[1]!=1 ||
           !width || width%128 || width>2560 || !pairs || pairs>192 ||
           !shape(input->inputTensors[1],DATA_I32,{rows,1}) ||
           !shape(input->inputTensors[2],DATA_F32,{1,rows}) ||
           !shape(input->inputTensors[4],DATA_F32,{rows,1}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{width,1,rows}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& channel=input->inputTensors[3].geometry;
        if(channel.dataType!=DATA_BF16 || channel.dims!=3 || channel.maxSizes[0]!=256 ||
           channel.maxSizes[1]*256!=width*2 || !channel.maxSizes[2] || channel.maxSizes[2]>384 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4 ||
           *static_cast<const int32_t*>(input->nodeParams.nodeParams)!=static_cast<int32_t>(width))return GLUE_FAILED;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=pairs;output->indexSpaceGeometry[1]=2;
        output->preferredSplitDim=2;
        auto& p=output->inputTensorAccessPattern[0];
        p.mapping[0]={1,0,0,float(width*4-1),true};p.mapping[1]={0,0,0,0,false};p.mapping[2]={0,1,0,0,false};
        for(unsigned i:{1u,4u}) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,2,0,1,false};ap.mapping[1]={1,0,0,0,true};
        }
        auto& sx=output->inputTensorAccessPattern[2];
        sx.mapping[0]={1,0,0,0,true};sx.mapping[1]={0,2,0,1,false};
        output->inputTensorAccessPattern[3].allRequired=true;
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->outputTensorAccessPattern[i];
            ap.mapping[0]={1,0,0,float(i==0?width-1:0),true};ap.mapping[1]={0,0,0,0,false};
            ap.mapping[2]={0,2,0,1,false};
        }
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=static_cast<int32_t>(width);
        return loadElf(output,&_binary___deepseek_v41_expert_pair_silu_gaudi2_o_start,
                             &_binary___deepseek_v41_expert_pair_silu_gaudi2_o_end);
    }
    if(operation==111) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto tiles=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[2];
        const auto parts=(tiles+31)/32;
        int32_t shift=0;
        if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
        std::memcpy(&shift,input->nodeParams.nodeParams,4);
        if(!tiles || tiles>2048 || rows<1 || rows>6 || shift<0 || shift>28 || shift%4 ||
           !shape(input->inputTensors[0],DATA_I32,{tiles,2,rows}) ||
           !shape(input->inputTensors[1],DATA_I32,{1,rows}) ||
           !shape(input->outputTensors[0],DATA_I32,{parts,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=parts;output->indexSpaceGeometry[1]=rows;
        auto& b=output->inputTensorAccessPattern[0];
        b.mapping[0]={0,32,0,63,false};b.mapping[1]={0,0,0,1,false};b.mapping[2]={1,1,0,0,false};
        auto& p=output->inputTensorAccessPattern[1];p.mapping[0]={0,0,0,0,false};p.mapping[1]={1,1,0,0,false};
        auto& o=output->outputTensorAccessPattern[0];o.mapping[0]={0,1,0,0,false};o.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=shift;
        return loadElf(output,&_binary___deepseek_v41_weighted_active_blocks_gaudi2_o_start,
                             &_binary___deepseek_v41_weighted_active_blocks_gaudi2_o_end);
    }
    if(operation==109) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto columns=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(columns<64 || columns>131072 || columns%64 || rows<1 || rows>6 ||
           !shape(input->inputTensors[0],DATA_F32,{columns,rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{columns,rows}) ||
           !shape(input->outputTensors[0],DATA_I32,{columns/64,2,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=columns/64;output->indexSpaceGeometry[1]=rows;
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,64,0,63,false};ap.mapping[1]={1,1,0,0,false};
        }
        auto& ap=output->outputTensorAccessPattern[0];
        ap.mapping[0]={0,1,0,0,false};ap.mapping[1]={0,0,0,1,false};ap.mapping[2]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_weighted_score_bounds_gaudi2_o_start,
                             &_binary___deepseek_v41_weighted_score_bounds_gaudi2_o_end);
    }
    if(operation==110) {
        if(input->inputTensorNr!=4)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto columns=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(!shape(input->inputTensors[3],DATA_I32,{(columns+2047)/2048,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto inherited=*input;inherited.inputTensorNr=3;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_weighted_mass_bins_gaudi2");
        const auto capacity=output->kernel.elfSize;
        const auto status=instantiate_private(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        output->inputTensorAccessPattern[3].allRequired=true;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_weighted_sparse_bins_gaudi2_o_start,
                             &_binary___deepseek_v41_weighted_sparse_bins_gaudi2_o_end);
    }
    if(operation==106 || operation==107) {
        const bool publish=operation==106;
        const unsigned base_inputs=publish?6:5;
        if(input->inputTensorNr!=base_inputs+1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto tokens=input->inputTensors[3].geometry.maxSizes[0];
        if(!shape(input->inputTensors[0],DATA_BF16,{512,256}) ||
           !shape(input->inputTensors[base_inputs],DATA_I32,{16,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto inherited=*input;
        // The parent validates a packed U8 ring. Adapt a private descriptor
        // array; mutating GC's BF16 descriptors corrupts subsequent passes.
        std::vector<Tensor> inherited_inputs(input->inputTensors,input->inputTensors+base_inputs);
        inherited.inputTensors=inherited_inputs.data();
        inherited.inputTensorNr=base_inputs;
        inherited.inputTensors[0].geometry.dataType=DATA_U8;
        inherited.inputTensors[0].geometry.maxSizes[0]=528;
        inherited.inputTensors[0].geometry.minSizes[0]=528;
        std::strcpy(inherited.guid.name,publish?"custom_deepseek_v41_main_batch_publish_gather_gaudi2":
                                            "custom_deepseek_v41_swa_only_reuse_gather_gaudi2");
        const auto capacity=output->kernel.elfSize;
        const auto result=instantiate_private(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->inputTensorAccessPattern[0].mapping[0].end_b=511;
        output->inputTensorAccessPattern[base_inputs].allRequired=true;
        output->kernel.elfSize=capacity;
        return publish?loadElf(output,&_binary___deepseek_v41_swa_cached_publish_gaudi2_o_start,
                                    &_binary___deepseek_v41_swa_cached_publish_gaudi2_o_end):
            loadElf(output,&_binary___deepseek_v41_swa_cached_reuse_gaudi2_o_start,
                           &_binary___deepseek_v41_swa_cached_reuse_gaudi2_o_end);
    }
    if(operation==108) {
        if(input->inputTensorNr!=4 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto tokens=input->inputTensors[1].geometry.maxSizes[1];
        if(tokens<1 || tokens>6 || !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
           !shape(input->inputTensors[1],DATA_BF16,{512,tokens}) ||
           !shape(input->inputTensors[2],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[3],DATA_BF16,{512,256}) ||
           !shape(input->outputTensors[0],DATA_I32,{16,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=16;output->indexSpaceGeometry[1]=tokens;
        output->inputTensorAccessPattern[0].allRequired=true;
        output->inputTensorAccessPattern[3].allRequired=true;
        auto& value=output->inputTensorAccessPattern[1];
        value.mapping[0]={0,32,0,31,false};value.mapping[1]={1,1,0,0,false};
        output->inputTensorAccessPattern[2].mapping[0]={1,1,0,0,false};
        auto& ready=output->outputTensorAccessPattern[0];
        ready.mapping[0]={0,1,0,0,false};ready.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_swa_batch_cache_write_gaudi2_o_start,
                             &_binary___deepseek_v41_swa_batch_cache_write_gaudi2_o_end);
    }
    if (operation == 105) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=2)return GLUE_FAILED;
        const auto width=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[2],tiles=width/128;
        const auto& channel=input->inputTensors[4].geometry;
        const auto scale_rows=input->inputTensors[3].geometry.maxSizes[1];
        if(!width || width>2560 || width%128 || !rows || rows>36 ||
           !shape(input->inputTensors[0],DATA_F32,{width,1,rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{width,1,rows}) ||
           !shape(input->inputTensors[2],DATA_I32,{rows,1}) ||
           (scale_rows!=1 && scale_rows!=rows) ||
           !shape(input->inputTensors[3],DATA_F32,{1,scale_rows}) ||
           channel.dataType!=DATA_BF16 || channel.dims!=3 || channel.maxSizes[0]!=256 ||
           channel.maxSizes[1]!=tiles || !channel.maxSizes[2] || channel.maxSizes[2]>384 ||
           !shape(input->inputTensors[5],DATA_F32,{rows,1}) ||
           !shape(input->outputTensors[0],DATA_BF16,{width,1,rows}) ||
           !shape(input->outputTensors[1],DATA_F32,{tiles,1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=tiles;output->indexSpaceGeometry[1]=rows;
        for(unsigned i:{0u,1u}) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,128,0,127,false};ap.mapping[1]={0,0,0,0,false};ap.mapping[2]={1,1,0,0,false};
        }
        for(unsigned i:{2u,5u}) {
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={1,1,0,0,false};ap.mapping[1]={0,0,0,0,false};
        }
        output->inputTensorAccessPattern[3].mapping[0]={0,0,0,0,false};
        output->inputTensorAccessPattern[3].mapping[1]={1,float(scale_rows==1?0:1),0,0,false};
        output->inputTensorAccessPattern[4].allRequired=true;
        for(unsigned i=0;i<2;++i) {
            auto& ap=output->outputTensorAccessPattern[i];
            ap.mapping[0]={0,float(i==0?128:1),0,float(i==0?127:0),false};
            ap.mapping[1]={0,0,0,0,false};ap.mapping[2]={1,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_silu_split_activate_gaudi2_o_start,
                             &_binary___deepseek_v41_silu_split_activate_gaudi2_o_end);
    }
    if(operation==92) return DeepseekV41MtpDequantFP8Gaudi2().GetGcDefinitions(input,output);
    if(operation==81) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=2 || !input->nodeParams.nodeParams ||
           input->nodeParams.nodeParamsSize!=20)return GLUE_FAILED;
        const auto* p=static_cast<const int32_t*>(input->nodeParams.nodeParams);
        const auto type=input->inputTensors[0].geometry.dataType;
        if(p[0]<0 || p[0]>3 || p[1]<0 || p[1]>30 || p[2]<6 || p[3]<1 ||
           (p[4]!=1 && p[4]!=2 && p[4]!=4) ||
           !shape(input->inputTensors[0],type,{uint64_t(p[3]),uint64_t(p[2])}) ||
           !shape(input->inputTensors[1],DATA_I32,{6}) || input->inputTensors[2].geometry.dataType!=DATA_I32 ||
           input->inputTensors[2].geometry.dims!=1 || !shape(input->outputTensors[0],type,{uint64_t(p[3]),6}) ||
           !shape(input->outputTensors[1],DATA_I32,{6}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const int width=256/p[4];
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=(p[3]+width-1)/width;
        output->indexSpaceGeometry[1]=6;
        for(unsigned i=0;i<3;++i)output->inputTensorAccessPattern[i].allRequired=true;
        auto& dst=output->outputTensorAccessPattern[0];
        dst.mapping[0]={0,float(width),0,float(width-1),false};dst.mapping[1]={1,1,0,0,false};
        output->outputTensorAccessPattern[1].allRequired=true;
        output->kernel.paramsNr=5;std::memcpy(output->kernel.scalarParams,p,20);
        return loadElf(output,&_binary___deepseek_v41_journal_copy_gaudi2_o_start,
                             &_binary___deepseek_v41_journal_copy_gaudi2_o_end);
    }
    if(operation==79) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=2)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<1 || rows>6 || !shape(input->outputTensors[0],DATA_F32,{64,rows}) ||
           !shape(input->outputTensors[1],DATA_I32,{2,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<6;++i) {
            const uint64_t columns=i<2?64:i==4?4:1;
            if(!shape(input->inputTensors[i],i==1?DATA_I32:DATA_F32,{columns,rows}))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->inputTensorAccessPattern[i].allRequired=true;
        }
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        for(unsigned i=0;i<2;++i) {
            auto& p=output->outputTensorAccessPattern[i];
            p.mapping[0]={0,0,0,float(i==0?63:1),false};p.mapping[1]={0,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_bounded_nucleus_parts_gaudi2_o_start,
                             &_binary___deepseek_v41_bounded_nucleus_parts_gaudi2_o_end);
    }
    if(operation==80) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1 || !input->nodeParams.nodeParams ||
           input->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto* params=static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if(rows<1 || rows>6 || params[0]<=0 || params[0]>131072 || params[1]<0 ||
           !shape(input->inputTensors[0],DATA_F32,{64,rows}) ||
           !shape(input->inputTensors[1],DATA_I32,{64,rows}) ||
           !shape(input->outputTensors[0],DATA_F32,{uint64_t(params[0]),rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<2;++i)output->inputTensorAccessPattern[i].allRequired=true;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        auto& p=output->outputTensorAccessPattern[0];
        p.mapping[0]={0,0,0,float(params[0]-1),false};p.mapping[1]={0,1,0,0,false};
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,params,8);
        return loadElf(output,&_binary___deepseek_v41_bounded_local_distribution_gaudi2_o_start,
                             &_binary___deepseek_v41_bounded_local_distribution_gaudi2_o_end);
    }
    if(operation==78) {
        if(input->inputTensorNr!=1 || input->outputTensorNr!=3 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const int columns=*static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if(rows<8 || rows>192 || rows%8 || columns<128 || columns>4096 || columns%64 ||
           !shape(input->inputTensors[0],DATA_F32,{uint64_t(columns),rows}) ||
           !shape(input->outputTensors[0],DATA_F32,{128,rows}) ||
           !shape(input->outputTensors[1],DATA_I32,{128,rows}) ||
           !shape(input->outputTensors[2],DATA_F32,{1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->inputTensorAccessPattern[0].allRequired=true;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        for(unsigned i=0;i<3;++i) {
            auto& map=output->outputTensorAccessPattern[i];
            map.mapping[0]={0,0,0,float(i==2?0:127),false};map.mapping[1]={0,1,0,0,false};
        }
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=columns;
        return loadElf(output,&_binary___deepseek_v41_vocab_lane_candidates_gaudi2_o_start,
                             &_binary___deepseek_v41_vocab_lane_candidates_gaudi2_o_end);
    }
    if((operation>=89 && operation<=91) || operation==115) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=(operation==89?1:operation==90?2:operation==115?4:3))
            return GLUE_FAILED;
        const auto columns=input->inputTensors[0].geometry.maxSizes[0];
        const auto rows=input->inputTensors[0].geometry.maxSizes[operation==90?2:1];
        const uint64_t parts=operation==90?input->inputTensors[0].geometry.maxSizes[1]:(columns+2047)/2048;
        if(rows<1 || rows>6 || (operation!=90 && !shape(input->inputTensors[0],DATA_F32,{columns,rows})))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        int32_t shift=0;
        if(operation!=91 && operation!=115) {
            if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
            std::memcpy(&shift,input->nodeParams.nodeParams,4);
            if(shift<0 || shift>28 || shift%4)return GLUE_FAILED;
        }
        for(unsigned i=0;i<3;++i)output->inputTensorAccessPattern[i].allRequired=true;
        if(operation==90) {
            if(columns!=16 || parts<1 || parts>64 || !shape(input->inputTensors[0],DATA_F32,{16,parts,rows}) ||
               !shape(input->inputTensors[1],DATA_I32,{1,rows}) ||
               !shape(input->inputTensors[2],DATA_F32,{1,rows}) ||
               !shape(input->outputTensors[0],DATA_I32,{1,rows}) ||
               !shape(input->outputTensors[1],DATA_F32,{1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
            for(unsigned i=0;i<2;++i) {
                auto& ap=output->outputTensorAccessPattern[i];
                ap.mapping[0]={0,0,0,0,false};ap.mapping[1]={0,1,0,0,false};
            }
        } else {
            if(columns<64 || columns>131072 || columns%64 ||
               !shape(input->inputTensors[1],DATA_F32,{columns,rows}) ||
               !shape(input->inputTensors[2],DATA_I32,{1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(operation==89) {
                if(!shape(input->outputTensors[0],DATA_F32,{16,parts,rows}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
                output->indexSpaceRank=2;output->indexSpaceGeometry[0]=parts;output->indexSpaceGeometry[1]=rows;
                auto& ap=output->outputTensorAccessPattern[0];
                ap.mapping[0]={0,0,0,15,false};ap.mapping[1]={0,1,0,0,false};ap.mapping[2]={1,1,0,0,false};
            } else {
                if(!shape(input->outputTensors[0],DATA_F32,{columns,rows}) ||
                   !shape(input->outputTensors[1],DATA_I32,{parts,rows}) ||
                   !shape(input->outputTensors[2],DATA_I32,{parts,rows}) ||
                   (operation==115 && !shape(input->outputTensors[3],DATA_I32,{parts,rows})))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
                output->indexSpaceRank=2;output->indexSpaceGeometry[0]=parts;output->indexSpaceGeometry[1]=rows;
                auto& ap=output->outputTensorAccessPattern[0];
                ap.mapping[0]={0,2048,0,2047,false};ap.mapping[1]={1,1,0,0,false};
                for(unsigned i=1;i<(operation==115?4u:3u);++i) {
                    auto& p=output->outputTensorAccessPattern[i];
                    p.mapping[0]={0,1,0,0,false};p.mapping[1]={1,1,0,0,false};
                }
            }
        }
        output->kernel.paramsNr=(operation==91 || operation==115)?0:1;output->kernel.scalarParams[0]=shift;
        if(operation==89)return loadElf(output,&_binary___deepseek_v41_weighted_mass_bins_gaudi2_o_start,
                                               &_binary___deepseek_v41_weighted_mass_bins_gaudi2_o_end);
        if(operation==90)return loadElf(output,&_binary___deepseek_v41_weighted_mass_advance_gaudi2_o_start,
                                               &_binary___deepseek_v41_weighted_mass_advance_gaudi2_o_end);
        if(operation==115)return loadElf(output,&_binary___deepseek_v41_weighted_finish_mask_gaudi2_o_start,
                                                &_binary___deepseek_v41_weighted_finish_mask_gaudi2_o_end);
        return loadElf(output,&_binary___deepseek_v41_weighted_mass_finish_gaudi2_o_start,
                              &_binary___deepseek_v41_weighted_mass_finish_gaudi2_o_end);
    }
    if(operation==88) {
        if(input->inputTensorNr!=7 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto logical=input->inputTensors[6].geometry.maxSizes[1];
        if(!tokens || tokens>6 || !rows || rows>0x7ffffdffULL || logical<512 || logical>32768 ||
           !shape(input->inputTensors[0],DATA_U8,{288,rows}) ||
           !shape(input->inputTensors[1],DATA_U8,{68,rows}) ||
           !shape(input->inputTensors[2],DATA_BF16,{512,tokens}) ||
           !shape(input->inputTensors[3],DATA_BF16,{128,tokens}) ||
           !shape(input->inputTensors[4],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[5],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[6],DATA_BF16,{512,logical}) ||
           !shape(input->outputTensors[0],DATA_I32,{36,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=36;output->indexSpaceGeometry[1]=tokens;
        for(unsigned i=0;i<7;++i)output->inputTensorAccessPattern[i].allRequired=true;
        auto& access=output->outputTensorAccessPattern[0];
        access.mapping[0]={0,1,0,0,false};access.mapping[1]={1,1,0,0,false};output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_fp4_paged_decoded_rows_gaudi2_o_start,
                             &_binary___deepseek_v41_fp4_paged_decoded_rows_gaudi2_o_end);
    }
    if(operation==87) {
        if(input->inputTensorNr!=3 || input->outputTensorNr!=2)return GLUE_FAILED;
        const auto& p=input->inputTensors[0].geometry;
        const auto rows=p.maxSizes[1],groups=p.maxSizes[2];
        if(!rows || rows>6 || (groups!=2 && groups!=4) ||
           !shape(input->inputTensors[0],DATA_F32,{1024,rows,groups}) ||
           !shape(input->inputTensors[1],DATA_F32,{1024,1,groups}) ||
           !shape(input->inputTensors[2],DATA_F32,{1,rows,groups}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{groups*1024,rows}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=rows>1?1:groups*8;
        output->indexSpaceGeometry[1]=rows;
        for(unsigned i=0;i<3;++i) {
            auto& ap=output->inputTensorAccessPattern[i].mapping;
            ap[0]={0,0,0,float(i==2?0:1023),false};
            ap[1]={1,i==1?0.f:1.f,0,0,false};
            ap[2]={0,0,0,float(groups-1),false};
        }
        auto& y=output->outputTensorAccessPattern[0].mapping;
        y[0]={0,rows>1?0.f:128.f,0,float(rows>1?groups*1024-1:127),false};y[1]={1,1,0,0,false};
        auto& scale=output->outputTensorAccessPattern[1].mapping;
        scale[0]={0,0,0,0,false};scale[1]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_woa_scale_dense_quant_gaudi2_o_start,
                             &_binary___deepseek_v41_woa_scale_dense_quant_gaudi2_o_end);
    }
    if(operation==85) {
        if(input->inputTensorNr!=4 || input->outputTensorNr!=2)return GLUE_FAILED;
        const auto& scores=input->inputTensors[0].geometry;
        const auto width=scores.maxSizes[0],heads=scores.maxSizes[1],tokens=scores.maxSizes[2];
        if(width!=640 || !heads || heads>64 || !tokens || tokens>6 ||
           !shape(input->inputTensors[0],DATA_F32,{width,heads,tokens}) ||
           !shape(input->inputTensors[1],DATA_F32,{width,tokens}) ||
           !shape(input->inputTensors[2],DATA_F32,{heads}) ||
           !shape(input->inputTensors[3],DATA_F32,{1}) ||
           !shape(input->outputTensors[0],DATA_BF16,{width,heads,tokens}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,heads,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=heads;output->indexSpaceGeometry[1]=tokens;
        auto& x=output->inputTensorAccessPattern[0].mapping;
        x[0]={0,0,0,float(width-1),false};x[1]={0,1,0,0,false};x[2]={1,1,0,0,false};
        auto& mask=output->inputTensorAccessPattern[1].mapping;
        mask[0]={0,0,0,float(width-1),false};mask[1]={1,1,0,0,false};
        output->inputTensorAccessPattern[2].mapping[0]={0,1,0,0,false};
        output->inputTensorAccessPattern[3].mapping[0]={0,0,0,0,false};
        for(unsigned i=0;i<2;++i) {
            auto& y=output->outputTensorAccessPattern[i].mapping;
            y[0]={0,0,0,float(i?0:width-1),false};y[1]={0,1,0,0,false};y[2]={1,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mla_exp_bf16_gaudi2_o_start,
                             &_binary___deepseek_v41_mla_exp_bf16_gaudi2_o_end);
    }
    if(operation==86) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto& product=input->inputTensors[0].geometry;
        const auto heads=product.maxSizes[1],tokens=product.maxSizes[2];
        if(!heads || heads>64 || !tokens || tokens>6 ||
           !shape(input->inputTensors[0],DATA_F32,{512,heads,tokens}) ||
           !shape(input->inputTensors[1],DATA_F32,{1,heads,tokens}) ||
           !shape(input->outputTensors[0],DATA_BF16,{512,heads,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;output->indexSpaceGeometry[0]=4;
        output->indexSpaceGeometry[1]=heads;output->indexSpaceGeometry[2]=tokens;
        for(auto* ap:{&output->inputTensorAccessPattern[0],&output->outputTensorAccessPattern[0]}) {
            ap->mapping[0]={0,128,0,127,false};ap->mapping[1]={1,1,0,0,false};ap->mapping[2]={2,1,0,0,false};
        }
        auto& inverse=output->inputTensorAccessPattern[1].mapping;
        inverse[0]={0,0,0,0,false};inverse[1]={1,1,0,0,false};inverse[2]={2,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mla_exp_normalize_gaudi2_o_start,
                             &_binary___deepseek_v41_mla_exp_normalize_gaudi2_o_end);
    }
    if(operation==83) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<2 || rows>6 || !shape(input->inputTensors[0],DATA_BF16,{20480,rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{20480,24}) ||
           !shape(input->outputTensors[0],DATA_F32,{25,20,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;output->indexSpaceGeometry[0]=6;
        output->indexSpaceGeometry[1]=(rows+2)/3;output->indexSpaceGeometry[2]=20;
        // Keep the complete small output field dimension: each point writes
        // four projection columns, and column block zero also writes norm.
        auto& x=output->inputTensorAccessPattern[0].mapping;
        x[0]={2,1024,0,1023,false};x[1]={1,3,0,2,false};
        auto& w=output->inputTensorAccessPattern[1].mapping;
        w[0]={2,1024,0,1023,false};w[1]={0,4,0,3,false};
        auto& y=output->outputTensorAccessPattern[0].mapping;
        y[0]={0,0,0,24,true};y[1]={2,1,0,0,false};y[2]={1,3,0,2,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mhc_control_tiles_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_control_tiles_gaudi2_o_end);
    }
    if(operation==84) {
        if(input->inputTensorNr!=1 || input->outputTensorNr!=1 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[2];
        if(rows<2 || rows>6 || !shape(input->inputTensors[0],DATA_F32,{25,20,rows}) ||
           !shape(input->outputTensors[0],DATA_F32,{25,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        auto& x=output->inputTensorAccessPattern[0].mapping;
        x[0]={0,0,0,24,true};x[1]={0,0,0,19,true};x[2]={0,1,0,0,false};
        auto& y=output->outputTensorAccessPattern[0].mapping;
        y[0]={0,0,0,24,true};y[1]={0,1,0,0,false};
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,8);
        return loadElf(output,&_binary___deepseek_v41_mhc_control_tiles_finish_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_control_tiles_finish_gaudi2_o_end);
    }
    if(operation==77) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if(rows<2 || rows>6 || !shape(input->inputTensors[0],DATA_BF16,{20480,rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{20480,24}) ||
           !shape(input->outputTensors[0],DATA_F32,{25,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->inputTensorAccessPattern[0].allRequired=true;
        output->inputTensorAccessPattern[1].allRequired=true;
        output->outputTensorAccessPattern[0].allRequired=true;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=24;
        output->kernel.paramsNr=2;std::memcpy(output->kernel.scalarParams,input->nodeParams.nodeParams,8);
        return loadElf(output,&_binary___deepseek_v41_mhc_control_reuse_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_control_reuse_gaudi2_o_end);
    }
    if(operation==76) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=3 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        if(tokens<2 || tokens>6 || !shape(input->inputTensors[2],DATA_I32,{512,tokens}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        for(unsigned i=0;i<input->inputTensorNr;++i)output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i=0;i<input->outputTensorNr;++i)output->outputTensorAccessPattern[i].allRequired=true;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=16;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_merge_cache_gaudi2_o_start,
                             &_binary___deepseek_v41_logical_merge_cache_gaudi2_o_end);
    }
    if(operation==75) {
        if(input->inputTensorNr!=4 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto width=input->inputTensors[0].geometry.maxSizes[0];
        if(rows<2 || rows>6 || (width!=24 && width!=48) ||
           !shape(input->inputTensors[0],DATA_F32,{width,rows}) ||
           !shape(input->inputTensors[1],DATA_BF16,{20480,rows}) ||
           !shape(input->inputTensors[2],DATA_F32,{3}) ||
           !shape(input->inputTensors[3],DATA_F32,{24}) ||
           !shape(input->outputTensors[0],DATA_F32,{24,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<4;++i)output->inputTensorAccessPattern[i].allRequired=true;
        auto& feature=output->outputTensorAccessPattern[0].mapping[0];
        feature.indexSpaceDim=0;feature.a=0;feature.start_b=0;feature.end_b=23;
        auto& row=output->outputTensorAccessPattern[0].mapping[1];
        row.indexSpaceDim=0;row.a=1;row.start_b=0;row.end_b=0;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mhc_mme_epilogue_gaudi2_o_start,
                             &_binary___deepseek_v41_mhc_mme_epilogue_gaudi2_o_end);
    }
    if(operation==74) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        // The measured parent is sliced into six one-token producers.
        // Keep all six KV/mask planes together, without changing arithmetic
        // or computing a union. Their complete working set is below12 MB.
        for(unsigned i=0;i<input->outputTensorNr;++i)
            output->outputTensorAccessPattern[i].allRequired=true;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_scale_cache_gaudi2_o_start,
                             &_binary___deepseek_v41_logical_scale_cache_gaudi2_o_end);
    }
    if(operation==72) {
        if(input->inputTensorNr!=4 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto& scores=input->inputTensors[0].geometry;
        if(!shape(input->outputTensors[0],DATA_BF16,{scores.maxSizes[0],scores.maxSizes[1]*2,
                                                   scores.maxSizes[2]}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;auto fake=input->outputTensors[0];
        fake.geometry.dataType=DATA_F32;fake.geometry.maxSizes[1]=scores.maxSizes[1];
        inherited.outputTensors=&fake;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_selected_mla_softmax_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->outputTensorAccessPattern[0].allRequired=true;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_selected_mla_stacked_softmax_gaudi2_o_start,
                             &_binary___deepseek_v41_selected_mla_stacked_softmax_gaudi2_o_end);
    }
    if(operation==73) {
        if(input->inputTensorNr!=1 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto& v=input->outputTensors[0].geometry;
        if(v.dims!=3 || v.dataType!=DATA_BF16 || v.maxSizes[0]!=512 || !v.maxSizes[1] || v.maxSizes[1]>64 ||
           !v.maxSizes[2] || v.maxSizes[2]>6 ||
           !shape(input->inputTensors[0],DATA_F32,{512,v.maxSizes[1]*2,v.maxSizes[2]}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;output->indexSpaceGeometry[0]=4;
        output->indexSpaceGeometry[1]=v.maxSizes[1];output->indexSpaceGeometry[2]=v.maxSizes[2];
        output->inputTensorAccessPattern[0].allRequired=true;
        auto& pattern=output->outputTensorAccessPattern[0];
        pattern.mapping[0]={0,128,0,127,false};pattern.mapping[1]={1,1,0,0,false};
        pattern.mapping[2]={2,1,0,0,false};output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mla_stacked_reduce_bf16_gaudi2_o_start,
                             &_binary___deepseek_v41_mla_stacked_reduce_bf16_gaudi2_o_end);
    }
    if(operation==70 || operation==71) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,operation==70 ?
            "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2" :
            "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        // Preserve the complete SRAM/MME contract; partition only the K tile.
        if(output->indexSpaceRank!=3)return GLUE_FAILED;
        output->indexSpaceGeometry[2]*=2;
        auto& q=output->inputTensorAccessPattern[1].mapping[0];
        q.a/=2; q.end_b=(q.end_b+1)/2-1;
        auto& scales=output->inputTensorAccessPattern[2].mapping[0];
        if(scales.a) {scales.a/=2; scales.end_b=(scales.end_b+1)/2-1;}
        auto& out=output->outputTensorAccessPattern[0].mapping[1];
        out.a/=2; out.end_b=(out.end_b+1)/2-1;
        output->kernel.elfSize=capacity;
        if(operation==70)return loadElf(output,&_binary___deepseek_v41_expert_token_wide_k64_sat_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_token_wide_k64_sat_gaudi2_o_end);
        return loadElf(output,&_binary___deepseek_v41_expert_n256_k64_sat_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_n256_k64_sat_gaudi2_o_end);
    }
    if(operation==68) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_token_wide_affine_route_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_token_wide_affine_route_gaudi2_o_end);
    }
    if(operation==67) {
        if(input->inputTensorNr!=2){input->inputTensorNr=2;return GLUE_INCOMPATIBLE_INPUT_COUNT;}
        if(input->outputTensorNr!=2){input->outputTensorNr=2;return GLUE_INCOMPATIBLE_OUTPUT_COUNT;}
        const auto capacity=input->inputTensors[0].geometry.maxSizes[1];
        const auto count=input->inputTensors[1].geometry.maxSizes[0];
        const auto tokens=input->inputTensors[1].geometry.maxSizes[1];
        if(!capacity || capacity>0x7fffffffULL || !count || count>2048 || count%8 || !tokens || tokens>6 ||
            !shape(input->inputTensors[0],DATA_BF16,{128,capacity}) ||
            !shape(input->inputTensors[1],DATA_I32,{count,tokens}) ||
            !shape(input->outputTensors[0],DATA_I32,{count*8,tokens}) ||
            !shape(input->outputTensors[1],DATA_BF16,{128,count*8,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=count/8;output->indexSpaceGeometry[1]=tokens;
        auto& keys=output->inputTensorAccessPattern[0];
        std::memset(&keys,0,sizeof(keys));keys.sparseAccess=true;
        keys.mapping[0]={0,0,0,127,false};keys.mapping[1]={0,0,0,float(capacity-1),false};
        auto& blocks=output->inputTensorAccessPattern[1];
        blocks.mapping[0]={0,8,0,7,false};blocks.mapping[1]={1,1,0,0,false};
        auto& rows=output->outputTensorAccessPattern[0];
        rows.mapping[0]={0,64,0,63,false};rows.mapping[1]={1,1,0,0,false};
        auto& selected=output->outputTensorAccessPattern[1];
        selected.mapping[0]={0,0,0,127,false};selected.mapping[1]={0,64,0,63,false};
        selected.mapping[2]={1,1,0,0,false};output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_candidate_mirror_keys_gaudi2_o_start,
                       &_binary___deepseek_v41_candidate_mirror_keys_gaudi2_o_end);
    }
    if(operation==66) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        // A routed row reads one activation scale, not the entire batch.
        // The immutable channel plane is indirectly addressed by expert ID.
        auto& sx=output->inputTensorAccessPattern[2];
        std::memset(&sx,0,sizeof(sx));
        sx.mapping[0]={0,0,0,0,false};
        sx.mapping[1]={0,input->inputTensors[2].geometry.maxSizes[1]==1?0.0f:1.0f,0,0,false};
        auto& channel=output->inputTensorAccessPattern[3];
        std::memset(&channel,0,sizeof(channel));
        channel.sparseAccess=true;
        for(unsigned d=0;d<3;++d)
            channel.mapping[d]={0,0,0,float(input->inputTensors[3].geometry.maxSizes[d]-1),false};
        return result;
    }
    if(operation==65) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_scale_cache_gaudi2_o_start,
                       &_binary___deepseek_v41_logical_scale_cache_gaudi2_o_end);
    }
    if(operation==64) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=3 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        const auto main_rows=input->inputTensors[1].geometry.maxSizes[1];
        const auto pages=input->inputTensors[4].geometry.maxSizes[0];
        const int ratio=*static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if(tokens<1 || tokens>6 || !main_rows || main_rows>32768 || !pages || pages>8192 ||
           (ratio!=1 && ratio!=2) || !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
           !shape(input->inputTensors[1],DATA_BF16,{512,main_rows}) ||
           !shape(input->inputTensors[2],DATA_I32,{512,tokens}) ||
           !shape(input->inputTensors[3],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[4],DATA_I32,{pages}) ||
           !shape(input->inputTensors[5],DATA_I32,{tokens}) ||
           !shape(input->outputTensors[0],DATA_BF16,{512,640,tokens}) ||
           !shape(input->outputTensors[1],DATA_F32,{512,640,tokens}) ||
           !shape(input->outputTensors[2],DATA_F32,{640,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=640;output->indexSpaceGeometry[1]=tokens;
        for(unsigned i=0;i<6;++i)output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i=0;i<3;++i) {
            auto& a=output->outputTensorAccessPattern[i];
            if(i<2) {a.mapping[0]={0,0,0,511,false};a.mapping[1]={0,1,0,0,false};a.mapping[2]={1,1,0,0,false};}
            else {a.mapping[0]={0,1,0,0,false};a.mapping[1]={1,1,0,0,false};}
        }
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
        return loadElf(output,&_binary___deepseek_v41_logical_main_mirror_gaudi2_o_start,
                       &_binary___deepseek_v41_logical_main_mirror_gaudi2_o_end);
    }
    if(operation==63) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_token_wide_unroll_steps_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_token_wide_unroll_steps_gaudi2_o_end);
    }
    if(operation==62) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_silu_channel_vector_gaudi2_o_start,
                       &_binary___deepseek_v41_silu_channel_vector_gaudi2_o_end);
    }
    if(operation>=58 && operation<=61) {
        const unsigned inputs=operation==58?1:operation==59?3:operation==60?6:5;
        const unsigned outputs=operation==59 || operation==60?3:1;
        if(input->inputTensorNr!=inputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        if(input->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        if(operation==58) {
            const auto buckets=input->inputTensors[0].geometry.maxSizes[0];
            if((buckets!=65 && buckets!=257) || !shape(input->inputTensors[0],DATA_I32,{buckets}) ||
               !shape(input->outputTensors[0],DATA_I32,{buckets+1}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=1;
            output->inputTensorAccessPattern[0].allRequired=true;
            output->outputTensorAccessPattern[0].allRequired=true;
        } else if(operation==59) {
            const auto tokens=input->inputTensors[0].geometry.maxSizes[1];
            if(tokens<2 || tokens>6 || !shape(input->inputTensors[0],DATA_I32,{512,tokens}) ||
               !shape(input->inputTensors[1],DATA_I32,{tokens}) ||
               !shape(input->inputTensors[2],DATA_I32,{tokens}) ||
               !shape(input->outputTensors[0],DATA_I32,{1024,65}) ||
               !shape(input->outputTensors[1],DATA_I32,{1024,65}) ||
               !shape(input->outputTensors[2],DATA_I32,{65}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=65;
            for(unsigned i=0;i<inputs;++i)output->inputTensorAccessPattern[i].allRequired=true;
            for(unsigned i=0;i<2;++i) {
                auto& ap=output->outputTensorAccessPattern[i];
                ap.mapping[0]={0,0,0,1023,false};ap.mapping[1]={0,1,0,0,false};
            }
            output->outputTensorAccessPattern[2].mapping[0]={0,1,0,0,false};
        } else if(operation==60) {
            const auto width=input->outputTensors[2].geometry.maxSizes[0];
            if(width<1280 || width>3328 || (width-256)%512 ||
               !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
               !shape(input->inputTensors[2],DATA_I32,{1024,65}) ||
               !shape(input->inputTensors[3],DATA_I32,{1024,65}) ||
               !shape(input->inputTensors[4],DATA_I32,{66}) ||
               !shape(input->outputTensors[0],DATA_BF16,{512,width}) ||
               !shape(input->outputTensors[1],DATA_F32,{512,width}) ||
               !shape(input->outputTensors[2],DATA_I32,{width}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(int32_t))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            const int32_t ratio=*static_cast<const int32_t*>(input->nodeParams.nodeParams);
            if(ratio!=1 && ratio!=2)return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
            output->indexSpaceRank=2;output->indexSpaceGeometry[0]=128;output->indexSpaceGeometry[1]=65;
            for(unsigned i=0;i<inputs;++i)output->inputTensorAccessPattern[i].allRequired=true;
            for(unsigned i=0;i<outputs;++i) {
                auto& ap=output->outputTensorAccessPattern[i];
                ap.allRequired=true;ap.sparseAccess=true;ap.memsetBeforeExecution=true;
            }
        } else {
            const auto width=input->inputTensors[0].geometry.maxSizes[0];
            const auto heads=input->inputTensors[2].geometry.maxSizes[0];
            const auto tokens=(width-256)/512;
            if(tokens<2 || tokens>6 || width!=256+tokens*512 || !heads || heads>64 ||
               !shape(input->inputTensors[0],DATA_F32,{width,tokens*heads}) ||
               !shape(input->inputTensors[1],DATA_I32,{width}) ||
               !shape(input->inputTensors[2],DATA_F32,{heads}) ||
               !shape(input->inputTensors[3],DATA_F32,{1}) ||
               !shape(input->inputTensors[4],DATA_I32,{66}) ||
               !shape(input->outputTensors[0],DATA_F32,{width,tokens*heads}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=tokens*heads;
            for(unsigned i=1;i<inputs;++i)output->inputTensorAccessPattern[i].allRequired=true;
            auto& ap=output->inputTensorAccessPattern[0];
            ap.mapping[0]={0,0,0,float(width-1),false};ap.mapping[1]={0,1,0,0,false};
            auto& result=output->outputTensorAccessPattern[0];
            result.allRequired=true;result.sparseAccess=true;result.memsetBeforeExecution=true;
        }
        const unsigned char *first,*last;
        if(operation==58){first=&_binary___deepseek_v41_mla_compact_prefix_gaudi2_o_start;last=&_binary___deepseek_v41_mla_compact_prefix_gaudi2_o_end;}
        else if(operation==59){first=&_binary___deepseek_v41_mla_compact_metadata_gaudi2_o_start;last=&_binary___deepseek_v41_mla_compact_metadata_gaudi2_o_end;}
        else if(operation==60){first=&_binary___deepseek_v41_mla_compact_decode_gaudi2_o_start;last=&_binary___deepseek_v41_mla_compact_decode_gaudi2_o_end;}
        else {first=&_binary___deepseek_v41_mla_compact_softmax_gaudi2_o_start;last=&_binary___deepseek_v41_mla_compact_softmax_gaudi2_o_end;}
        return loadElf(output,first,last);
    }
    if(operation==56 || operation==57) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,operation==56 ? "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2" :
                    "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        const auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return operation==56 ? loadElf(output,&_binary___deepseek_v41_expert_token_wide_group_pipe_gaudi2_o_start,
                                      &_binary___deepseek_v41_expert_token_wide_group_pipe_gaudi2_o_end) :
            loadElf(output,&_binary___deepseek_v41_expert_down_group_pipe_gaudi2_o_start,
                    &_binary___deepseek_v41_expert_down_group_pipe_gaudi2_o_end);
    }
    if(operation>=53 && operation<=55) {
        const unsigned inputs=operation==53?3:operation==54?5:4;
        const unsigned outputs=operation==55?1:2;
        if(input->inputTensorNr!=inputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        if(input->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        uint64_t width=0,tokens=0,heads=0;
        if(operation==53) {
            const auto& selected=input->inputTensors[0].geometry;
            tokens=selected.maxSizes[1];width=256+tokens*512;
            if(!shape(input->inputTensors[0],DATA_I32,{512,tokens}) || tokens<2 || tokens>6 ||
               !shape(input->inputTensors[1],DATA_I32,{tokens}) ||
               !shape(input->inputTensors[2],DATA_I32,{tokens}) ||
               !shape(input->outputTensors[0],DATA_I32,{width}) ||
               !shape(input->outputTensors[1],DATA_I32,{width}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=width/16;
            for(unsigned i=0;i<inputs;++i)output->inputTensorAccessPattern[i].allRequired=true;
            for(unsigned i=0;i<outputs;++i)
                output->outputTensorAccessPattern[i].mapping[0]={0,16,0,15,false};
        } else if(operation==54) {
            width=input->inputTensors[2].geometry.maxSizes[0];
            if(width<1280 || width>3328 || (width-256)%512 ||
               !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
               !shape(input->inputTensors[2],DATA_I32,{width}) ||
               !shape(input->inputTensors[3],DATA_I32,{width}) ||
               !shape(input->outputTensors[0],DATA_BF16,{512,width}) ||
               !shape(input->outputTensors[1],DATA_F32,{512,width}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(int32_t))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            const int32_t ratio=*static_cast<const int32_t*>(input->nodeParams.nodeParams);
            if(ratio!=1 && ratio!=2)return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=width/8;
            for(unsigned i=0;i<inputs;++i)output->inputTensorAccessPattern[i].allRequired=true;
            for(unsigned i=0;i<outputs;++i) {
                auto& ap=output->outputTensorAccessPattern[i];
                ap.mapping[0]={0,0,0,511,false};ap.mapping[1]={0,8,0,7,false};
            }
        } else {
            const auto& scores=input->inputTensors[0].geometry;
            width=scores.maxSizes[0];heads=input->inputTensors[2].geometry.maxSizes[0];
            tokens=(width-256)/512;
            if(tokens<2 || tokens>6 || width!=256+tokens*512 || !heads || heads>64 ||
               !shape(input->inputTensors[0],DATA_F32,{width,tokens*heads}) ||
               !shape(input->inputTensors[1],DATA_I32,{width}) ||
               !shape(input->inputTensors[2],DATA_F32,{heads}) ||
               !shape(input->inputTensors[3],DATA_F32,{1}) ||
               !shape(input->outputTensors[0],DATA_F32,{width,tokens*heads}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank=1;output->indexSpaceGeometry[0]=tokens*heads;
            for(unsigned i=1;i<inputs;++i)output->inputTensorAccessPattern[i].allRequired=true;
            for(auto* ap:{&output->inputTensorAccessPattern[0],&output->outputTensorAccessPattern[0]}) {
                ap->mapping[0]={0,0,0,float(width-1),false};ap->mapping[1]={0,1,0,0,false};
            }
        }
        const unsigned char *first,*last;
        if(operation==53){first=&_binary___deepseek_v41_mla_shared_owners_gaudi2_o_start;last=&_binary___deepseek_v41_mla_shared_owners_gaudi2_o_end;}
        else if(operation==54){first=&_binary___deepseek_v41_mla_shared_decode_gaudi2_o_start;last=&_binary___deepseek_v41_mla_shared_decode_gaudi2_o_end;}
        else {first=&_binary___deepseek_v41_mla_shared_softmax_gaudi2_o_start;last=&_binary___deepseek_v41_mla_shared_softmax_gaudi2_o_end;}
        return loadElf(output,first,last);
    }
    if(operation==52) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_logical_mla_coord_cached_gaudi2_o_start,
                       &_binary___deepseek_v41_logical_mla_coord_cached_gaudi2_o_end);
    }
    if(operation==82) {
        if(input->inputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        if(input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        const auto& product=input->inputTensors[0].geometry;
        if(product.dataType!=DATA_F32 || product.dims!=3 || product.maxSizes[1]!=1 ||
           !product.maxSizes[0] || product.maxSizes[0]%512 || product.maxSizes[0]>10240 ||
           !product.maxSizes[2] || product.maxSizes[2]>192)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto width=product.maxSizes[0]/4,pairs=product.maxSizes[2];
        if(!shape(input->inputTensors[1],DATA_I32,{2,pairs}) ||
           !shape(input->inputTensors[2],DATA_F32,{2,pairs}) ||
           !shape(input->inputTensors[4],DATA_F32,{2,pairs}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{width*2,1,pairs}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,2,pairs}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const auto& channel=input->inputTensors[3].geometry;
        if(channel.dataType!=DATA_BF16 || channel.dims!=3 || channel.maxSizes[0]!=256 ||
           channel.maxSizes[1]*256!=width*2 || !channel.maxSizes[2] || channel.maxSizes[2]>384)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(int32_t) ||
           *static_cast<const int32_t*>(input->nodeParams.nodeParams)!=static_cast<int32_t>(width))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=pairs;output->indexSpaceGeometry[1]=2;
        output->preferredSplitDim=2;
        auto& ap=output->inputTensorAccessPattern[0];
        ap.mapping[0]={1,0,0,float(width*4-1),true};
        ap.mapping[1]={0,0,0,0,false};ap.mapping[2]={0,1,0,0,false};
        for(unsigned i:{1u,2u,4u}) {
            auto& p=output->inputTensorAccessPattern[i];
            p.mapping[0]={1,0,0,1,true};p.mapping[1]={0,1,0,0,false};
        }
        output->inputTensorAccessPattern[3].allRequired=true;
        auto& q=output->outputTensorAccessPattern[0];
        q.mapping[0]={1,0,0,float(width*2-1),true};q.mapping[1]={0,0,0,0,false};q.mapping[2]={0,1,0,0,false};
        auto& scale=output->outputTensorAccessPattern[1];
        scale.mapping[0]={0,0,0,0,false};scale.mapping[1]={1,0,0,1,true};scale.mapping[2]={0,1,0,0,false};
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=static_cast<int32_t>(width);
        return loadElf(output,&_binary___deepseek_v41_expert_physical_role_silu_gaudi2_o_start,&_binary___deepseek_v41_expert_physical_role_silu_gaudi2_o_end);
    }
    if(operation==51) {
        if(input->inputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        if(input->outputTensorNr!=2)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        const auto& product=input->inputTensors[0].geometry;
        if(product.dataType!=DATA_F32 || product.dims!=3 || product.maxSizes[1]!=1 ||
           !product.maxSizes[0] || product.maxSizes[0]%512 || product.maxSizes[0]>10240 ||
           !product.maxSizes[2] || product.maxSizes[2]>192)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto width=product.maxSizes[0]/4,pairs=product.maxSizes[2];
        if(!shape(input->inputTensors[1],DATA_I32,{2,pairs}) ||
           !shape(input->inputTensors[2],DATA_F32,{2,pairs}) ||
           !shape(input->inputTensors[4],DATA_F32,{2,pairs}) ||
           !shape(input->outputTensors[0],DATA_F8_143,{width*2,1,pairs}) ||
           !shape(input->outputTensors[1],DATA_F32,{1,2,pairs}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const auto& channel=input->inputTensors[3].geometry;
        if(channel.dataType!=DATA_BF16 || channel.dims!=3 || channel.maxSizes[0]!=256 ||
           channel.maxSizes[1]*256!=width*2 || !channel.maxSizes[2] || channel.maxSizes[2]>384)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(int32_t) ||
           *static_cast<const int32_t*>(input->nodeParams.nodeParams)!=static_cast<int32_t>(width))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=pairs;
        auto& ap=output->inputTensorAccessPattern[0];
        ap.mapping[0]={0,0,0,float(width*4-1),false};
        ap.mapping[1]={0,0,0,0,false};ap.mapping[2]={0,1,0,0,false};
        for(unsigned i:{1u,2u,4u}) {
            auto& p=output->inputTensorAccessPattern[i];
            p.mapping[0]={0,0,0,1,false};p.mapping[1]={0,1,0,0,false};
        }
        output->inputTensorAccessPattern[3].allRequired=true;
        auto& q=output->outputTensorAccessPattern[0];
        q.mapping[0]={0,0,0,float(width*2-1),false};q.mapping[1]={0,0,0,0,false};q.mapping[2]={0,1,0,0,false};
        auto& scale=output->outputTensorAccessPattern[1];
        scale.mapping[0]={0,0,0,0,false};scale.mapping[1]={0,0,0,1,false};scale.mapping[2]={0,1,0,0,false};
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=static_cast<int32_t>(width);
        return loadElf(output,&_binary___deepseek_v41_expert_physical_silu_gaudi2_o_start,&_binary___deepseek_v41_expert_physical_silu_gaudi2_o_end);
    }
    if (operation>=48 && operation<=50) {
        const bool owner=operation==48,decode=operation==49;
        const unsigned inputs=owner?3:decode?5:8,outputs=owner||decode?1:3;
        if(input->inputTensorNr!=inputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
        if(input->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        const auto& selected=input->inputTensors[owner?0:2].geometry;
        if(selected.dataType!=DATA_I32 || selected.dims!=2 || selected.maxSizes[0]!=512 ||
            !selected.maxSizes[1] || selected.maxSizes[1]>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto tokens=selected.maxSizes[1];
        for(unsigned i=0;i<inputs;++i) {
            const auto& g=input->inputTensors[i].geometry;
            const auto type=owner?DATA_I32:i<2?DATA_U8:!decode && i==7?DATA_BF16:DATA_I32;
            if(g.dataType!=type)return GLUE_INCOMPATIBLE_DATA_TYPE;
            output->inputTensorAccessPattern[i].allRequired=true;
        }
        if(owner) {
            if(!shape(input->inputTensors[1],DATA_I32,{tokens}) ||
               !shape(input->inputTensors[2],DATA_I32,{tokens}) ||
               !shape(input->outputTensors[0],DATA_I32,{4352}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            auto& ap=output->outputTensorAccessPattern[0];
            ap.allRequired=true;ap.memsetBeforeExecution=true;ap.sparseAccess=true;
            output->indexSpaceRank=2;output->indexSpaceGeometry[0]=640;output->indexSpaceGeometry[1]=tokens;
        } else {
            if(!shape(input->inputTensors[0],DATA_U8,{528,256}) ||
               input->inputTensors[1].geometry.dims!=2 || input->inputTensors[1].geometry.maxSizes[0]!=288 ||
               !input->inputTensors[1].geometry.maxSizes[1] ||
               !shape(input->inputTensors[decode?3:6],DATA_I32,{4352}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            const auto& pages=input->inputTensors[decode?4:5].geometry;
            if(pages.dims!=1 || !pages.maxSizes[0] || pages.maxSizes[0]>8192)return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=sizeof(int))return GLUE_FAILED;
            int ratio;std::memcpy(&ratio,input->nodeParams.nodeParams,sizeof(ratio));
            if(ratio!=1 && ratio!=2)return GLUE_FAILED;
            output->kernel.paramsNr=1;std::memcpy(output->kernel.scalarParams,&ratio,sizeof(ratio));
            if(decode) {
                if(!shape(input->outputTensors[0],DATA_BF16,{512,4352}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
                auto& ap=output->outputTensorAccessPattern[0];
                ap.mapping[0]={0,0,0,511,false};ap.mapping[1]={0,1,0,0,false};ap.sparseAccess=true;
                output->indexSpaceRank=1;output->indexSpaceGeometry[0]=4352;
            } else {
                if(!shape(input->inputTensors[3],DATA_I32,{tokens}) ||
                   !shape(input->inputTensors[4],DATA_I32,{tokens}) ||
                   !shape(input->inputTensors[7],DATA_BF16,{512,4352}) ||
                   !shape(input->outputTensors[0],DATA_BF16,{512,640,tokens}) ||
                   !shape(input->outputTensors[1],DATA_F32,{512,640,tokens}) ||
                   !shape(input->outputTensors[2],DATA_F32,{640,tokens}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
                for(unsigned i=0;i<3;++i) {
                    auto& ap=output->outputTensorAccessPattern[i];
                    if(i<2){ap.mapping[0]={0,0,0,511,false};ap.mapping[1]={0,1,0,0,false};ap.mapping[2]={1,1,0,0,false};}
                    else{ap.mapping[0]={0,1,0,0,false};ap.mapping[1]={1,1,0,0,false};}
                }
                output->indexSpaceRank=2;output->indexSpaceGeometry[0]=640;output->indexSpaceGeometry[1]=tokens;
            }
        }
        if(owner)return loadElf(output,&_binary___deepseek_v41_mla_hash_owners_gaudi2_o_start,&_binary___deepseek_v41_mla_hash_owners_gaudi2_o_end);
        if(decode)return loadElf(output,&_binary___deepseek_v41_mla_hash_decode_gaudi2_o_start,&_binary___deepseek_v41_mla_hash_decode_gaudi2_o_end);
        return loadElf(output,&_binary___deepseek_v41_mla_hash_gather_gaudi2_o_start,&_binary___deepseek_v41_mla_hash_gather_gaudi2_o_end);
    }
    if (operation == 47) return DeepseekV41QkvNormPublishGaudi2().GetGcDefinitions(input, output);
    if (operation == 42) return DeepseekV41KVNormRopePublishGaudi2().GetGcDefinitions(input, output);
    if (operation == 40) return DeepseekV41PeerPostCollapseGaudi2().GetGcDefinitions(input, output);
    if (operation == 39) return DeepseekV41PeerPostNormQuantGaudi2().GetGcDefinitions(input, output);
    if (operation == 37) return dsv41_route_pack::instantiate(input, output, 3,
        &_binary___deepseek_v41_expert_three_route_sat_fp8_gaudi2_o_start,
        &_binary___deepseek_v41_expert_three_route_sat_fp8_gaudi2_o_end);
    if (operation == 38) {
        if (input->inputTensorNr != 4 || input->outputTensorNr != 1) return GLUE_FAILED;
        const auto tokens = input->outputTensors[0].geometry.maxSizes[2];
        const auto hidden = input->outputTensors[0].geometry.maxSizes[0];
        const auto experts = input->inputTensors[3].geometry.maxSizes[2];
        if (tokens < 2 || tokens > 6 || hidden != 5120 || !experts || experts > 384 ||
            !shape(input->inputTensors[0], DATA_F32, {hidden * 3, 3, tokens * 2}) ||
            !shape(input->inputTensors[1], DATA_I32, {tokens * 6, 1}) ||
            !shape(input->inputTensors[2], DATA_F32, {1, tokens * 6}) ||
            !shape(input->inputTensors[3], DATA_BF16, {256, hidden / 256, experts}) ||
            !shape(input->outputTensors[0], DATA_BF16, {hidden, 1, tokens}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 2;
        output->indexSpaceGeometry[0] = hidden / 128;
        output->indexSpaceGeometry[1] = tokens;
        auto& product = output->inputTensorAccessPattern[0];
        product.mapping[0] = {0, 128, 0, float(2 * hidden + 127), false};
        product.mapping[1] = {1, 0, 0, 2, false};
        product.mapping[2] = {1, 2, 0, 1, false};
        output->inputTensorAccessPattern[1].allRequired = true;
        output->inputTensorAccessPattern[2].allRequired = true;
        output->inputTensorAccessPattern[3].allRequired = true;
        auto& result = output->outputTensorAccessPattern[0];
        result.mapping[0] = {0, 128, 0, 127, false};
        result.mapping[1] = {1, 0, 0, 0, false};
        result.mapping[2] = {1, 1, 0, 0, false};
        output->kernel.paramsNr = 0;
        return loadElf(output, &_binary___deepseek_v41_expert_three_route_scale_reduce_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_three_route_scale_reduce_gaudi2_o_end);
    }
    if (operation == 35) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1 || !input->nodeParams.nodeParams ||
           input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const int columns=*static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if(rows<8 || rows>48 || rows%8 || columns<128 || columns>4096 || columns%128 ||
           !shape(input->inputTensors[0],DATA_F32,{uint64_t(columns),rows}) ||
           !shape(input->inputTensors[1],DATA_F32,{1,rows/8}) ||
           !shape(input->outputTensors[0],DATA_I32,{uint64_t(columns/32),rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=columns/128;output->indexSpaceGeometry[1]=rows;
        auto& inputMap=output->inputTensorAccessPattern[0];
        inputMap.mapping[0]={0,128,0,127,false};inputMap.mapping[1]={1,1,0,0,false};
        output->inputTensorAccessPattern[1].allRequired=true;
        auto& outputMap=output->outputTensorAccessPattern[0];
        outputMap.mapping[0]={0,4,0,3,false};outputMap.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=columns;
        return loadElf(output,&_binary___deepseek_v41_vocab_filter_gaudi2_o_start,
                       &_binary___deepseek_v41_vocab_filter_gaudi2_o_end);
    }
    if (operation == 36) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=3 || !input->nodeParams.nodeParams ||
           input->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto* params=static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if(rows<8 || rows>48 || rows%8 || params[0]<128 || params[0]>4096 || params[0]%128 ||
           params[1]<1 || params[1]>256 || !shape(input->inputTensors[0],DATA_F32,{uint64_t(params[0]),rows}) ||
           !shape(input->inputTensors[1],DATA_I32,{uint64_t(params[0]/32),rows}) ||
           !shape(input->outputTensors[0],DATA_F32,{uint64_t(params[1]),rows}) ||
           !shape(input->outputTensors[1],DATA_I32,{uint64_t(params[1]),rows}) ||
           !shape(input->outputTensors[2],DATA_I32,{1,rows}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;
        for(unsigned i=0;i<2;++i)output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i=0;i<3;++i) {
            auto& pattern=output->outputTensorAccessPattern[i];
            pattern.mapping[0]={0,0,0,float(i==2?0:params[1]-1),false};pattern.mapping[1]={0,1,0,0,false};
        }
        output->kernel.paramsNr=2;for(unsigned i=0;i<2;++i)output->kernel.scalarParams[i]=params[i];
        return loadElf(output,&_binary___deepseek_v41_vocab_filter_emit_gaudi2_o_start,
                       &_binary___deepseek_v41_vocab_filter_emit_gaudi2_o_end);
    }
    if (operation == 34) {
        if(input->inputTensorNr!=6 || input->outputTensorNr!=3 || !input->nodeParams.nodeParams ||
           input->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        const auto main_rows=input->inputTensors[1].geometry.maxSizes[1];
        const auto page_rows=input->inputTensors[4].geometry.maxSizes[0];
        const auto* params=static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if(tokens<2 || tokens>6 || (params[0]!=1 && params[0]!=2) ||
           params[1]<512 || params[1]>65536 || params[1]%128 || !main_rows || !page_rows || page_rows>8192 ||
           !shape(input->inputTensors[0],DATA_U8,{528,256}) ||
           !shape(input->inputTensors[1],DATA_U8,{288,main_rows}) ||
           !shape(input->inputTensors[2],DATA_I32,{512,tokens}) ||
           !shape(input->inputTensors[3],DATA_I32,{tokens}) ||
           !shape(input->inputTensors[4],DATA_I32,{page_rows}) ||
           !shape(input->inputTensors[5],DATA_I32,{tokens}) ||
           !shape(input->outputTensors[0],DATA_BF16,{512,640,tokens}) ||
           !shape(input->outputTensors[1],DATA_F32,{512,640,tokens}) ||
           !shape(input->outputTensors[2],DATA_F32,{640,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=128;
        // Shared owners scatter to disjoint query slots. Preserve the full
        // C extent and never substitute affine one-query slices.
        for(unsigned i=0;i<6;++i)output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i=0;i<2;++i) {
            auto& access=output->inputTensorAccessPattern[i];
            access.allRequired=false;access.sparseAccess=true;
            for(unsigned d=0;d<input->inputTensors[i].geometry.dims;++d)
                access.mapping[d]={0,0,0,float(input->inputTensors[i].geometry.maxSizes[d]-1),false};
        }
        for(unsigned i=0;i<3;++i) {
            auto& access=output->outputTensorAccessPattern[i];access.sparseAccess=true;
            for(unsigned d=0;d<input->outputTensors[i].geometry.dims;++d)
                access.mapping[d]={0,0,0,float(input->outputTensors[i].geometry.maxSizes[d]-1),false};
        }
        output->kernel.paramsNr=2;
        std::memcpy(output->kernel.scalarParams,params,8);
        return loadElf(output,&_binary___deepseek_v41_mla_merged_gaudi2_o_start,
                       &_binary___deepseek_v41_mla_merged_gaudi2_o_end);
    }
    if (operation == 32) return DeepseekV41SiluActivateTileGaudi2().GetGcDefinitions(input, output);
    if (operation == 33) return DeepseekV41SiluQuantTileGaudi2().GetGcDefinitions(input, output);
    if (operation == 31) {
        // The source already emits [C,640,512]. Declare a conservative full-C
        // extent on its row axis so a batch-GEMM consumer cannot slice the
        // producer into C independent one-row invocations. The ELF, logical
        // addresses and index space [640,C] remain exactly the parent ones.
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next || input->inputTensorNr != 6 || input->outputTensorNr != 3) return GLUE_FAILED;
        auto inherited = *input;
        std::strcpy(inherited.guid.name, "custom_deepseek_v41_logical_mla_vector_gaudi2");
        const auto result = next(&inherited, output);
        if (result != GLUE_SUCCESS && result != GLUE_INSUFFICIENT_ELF_BUFFER) return result;
        const auto count = input->inputTensors[2].geometry.maxSizes[1];
        auto full = [count](TensorAccessPattern& pattern, unsigned dim) {
            pattern.mapping[dim] = {1, 0, 0, float(count - 1), false};
        };
        full(output->inputTensorAccessPattern[2], 1);
        full(output->inputTensorAccessPattern[3], 0);
        full(output->inputTensorAccessPattern[5], 0);
        full(output->outputTensorAccessPattern[0], 2);
        full(output->outputTensorAccessPattern[1], 2);
        full(output->outputTensorAccessPattern[2], 1);
        return result;
    }
    if (operation == 29 || operation == 30) {
        const bool emit = operation == 30;
        if (input->inputTensorNr != (emit ? 2u : 1u) || input->outputTensorNr != (emit ? 2u : 1u) ||
            !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != 2 * sizeof(int))
            return GLUE_FAILED;
        const auto* params = static_cast<const int*>(input->nodeParams.nodeParams);
        const int columns = params[0], width = params[1];
        const auto& score = input->inputTensors[0].geometry;
        if (columns < 64 || columns > 131072 || columns % 64 || width < 1 || width > 256 || width > columns ||
            score.dims != 2 || score.maxSizes[0] != static_cast<unsigned>(columns) ||
            score.maxSizes[1] < 1 || score.maxSizes[1] > 48 || score.dataType != DATA_F32)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        const auto capacity = output->kernel.elfSize;
        auto inherited = *input;
        std::vector<Tensor> tensors(input->inputTensors, input->inputTensors + input->inputTensorNr);
        std::vector<Tensor> results(input->outputTensors, input->outputTensors + input->outputTensorNr);
        const unsigned metadata_width = 18 + (columns+127)/128*8;
        const auto& metadata = emit ? tensors[1].geometry : results[0].geometry;
        if (metadata.dims != 2 || metadata.maxSizes[0] != metadata_width ||
            metadata.maxSizes[1] != score.maxSizes[1] || metadata.dataType != DATA_I32)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto& inherited_metadata = emit ? tensors[1].geometry : results[0].geometry;
        inherited_metadata.maxSizes[0] = inherited_metadata.minSizes[0] = 18;
        int bounded_params[] = {columns > 4096 ? 4096 : columns, width};
        tensors[0].geometry.maxSizes[0] = bounded_params[0];
        tensors[0].geometry.minSizes[0] = bounded_params[0];
        inherited.inputTensors = tensors.data();
        inherited.outputTensors = results.data();
        inherited.nodeParams.nodeParams = bounded_params;
        std::strcpy(inherited.guid.name, emit ? "custom_deepseek_v41_prefill_topk_emit_gaudi2" :
                                              "custom_deepseek_v41_prefill_topk_threshold_gaudi2");
        const auto status = next(&inherited, output);
        if (status != GLUE_SUCCESS && status != GLUE_INSUFFICIENT_ELF_BUFFER) return status;
        // Restore the real access span and scalar columns; no truncated mapping
        // may reach the compiler or runtime binding.
        output->inputTensorAccessPattern[0].mapping[0].end_b = columns - 1;
        if (emit) output->inputTensorAccessPattern[1].mapping[0].end_b = metadata_width - 1;
        else output->outputTensorAccessPattern[0].mapping[0].end_b = metadata_width - 1;
        std::memcpy(output->kernel.scalarParams, params, 2 * sizeof(int));
        output->kernel.elfSize = capacity;
        return emit ? loadElf(output, &_binary___deepseek_v41_vocab_radix_emit_gaudi2_o_start,
                                      &_binary___deepseek_v41_vocab_radix_emit_gaudi2_o_end) :
                      loadElf(output, &_binary___deepseek_v41_vocab_radix_threshold_gaudi2_o_start,
                                      &_binary___deepseek_v41_vocab_radix_threshold_gaudi2_o_end);
    }
    if (operation>=44 && operation<=46) {
        const bool mass=operation==44,draw=operation==45;
        const unsigned inputs=mass?2:draw?4:2,outputs=mass?3:draw?2:2;
        if(input->inputTensorNr!=inputs || input->outputTensorNr!=outputs)return GLUE_FAILED;
        const auto& g=input->inputTensors[0].geometry;
        const auto rows=g.maxSizes[1];
        if(g.dims!=2 || !rows || rows>6 || (!mass && !draw && g.maxSizes[0]!=2))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto columns=mass||draw ? g.maxSizes[0] : input->inputTensors[1].geometry.maxSizes[0]*64;
        if(columns<128 || columns>129280 || columns%128)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const TensorDataType inTypes[4]={mass||draw?DATA_F32:DATA_I32,mass?DATA_F32:DATA_I32,DATA_I32,DATA_F32};
        for(unsigned i=0;i<inputs;++i) {
            const auto& x=input->inputTensors[i].geometry;
            const uint64_t width=mass ? (i==0?columns:4) : draw ? (i==0?columns:i==1?columns/32:i==2?2:4) : (i==0?2:columns/64);
            if(x.dims!=2 || x.dataType!=inTypes[i] || x.maxSizes[0]!=width || x.maxSizes[1]!=rows)return GLUE_INCOMPATIBLE_INPUT_SIZE;
            auto& ap=output->inputTensorAccessPattern[i];
            ap.mapping[0]={0,0,0,float(width-1),false};ap.mapping[1]={0,1,0,0,false};
        }
        for(unsigned i=0;i<outputs;++i) {
            const auto& x=input->outputTensors[i].geometry;
            const auto type=mass && i==0?DATA_F32:DATA_I32;
            const uint64_t width=mass ? (i==0?columns:i==1?columns/32:2) : draw ? (i==0?2:columns/64) : rows;
            if(x.dataType!=type || x.dims!=(mass||draw?2u:1u) || x.maxSizes[0]!=width ||
               ((mass||draw) && x.maxSizes[1]!=rows))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            auto& ap=output->outputTensorAccessPattern[i];
            if(mass||draw){ap.mapping[0]={0,0,0,float(width-1),false};ap.mapping[1]={0,1,0,0,false};}
            else ap.mapping[0]={0,1,0,0,false};
        }
        output->indexSpaceRank=1;output->indexSpaceGeometry[0]=rows;output->kernel.paramsNr=0;
        if(mass)return loadElf(output,&_binary___deepseek_v41_nucleus_mass_gaudi2_o_start,&_binary___deepseek_v41_nucleus_mass_gaudi2_o_end);
        if(draw)return loadElf(output,&_binary___deepseek_v41_nucleus_draw_gaudi2_o_start,&_binary___deepseek_v41_nucleus_draw_gaudi2_o_end);
        return loadElf(output,&_binary___deepseek_v41_nucleus_finish_gaudi2_o_start,&_binary___deepseek_v41_nucleus_finish_gaudi2_o_end);
    }
    if (operation == 43) {
        if (input->inputTensorNr != 3 || input->outputTensorNr != 1 ||
            input->inputTensors[0].geometry.dataType != DATA_F32) return GLUE_FAILED;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        Tensor fake[3]={input->inputTensors[0],input->inputTensors[1],input->inputTensors[2]};
        fake[0].geometry.dataType=DATA_BF16;inherited.inputTensors=fake;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_rope_inverse_bf16_gaudi2");
        const auto status=next(&inherited,output);
        if(status!=GLUE_SUCCESS && status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_pv_rope_f32_gaudi2_o_start,
                              &_binary___deepseek_v41_pv_rope_f32_gaudi2_o_end);
    }
    if (operation == 41) {
        if (input->inputTensorNr != 5 || input->outputTensorNr != 2 ||
            input->inputTensors[0].geometry.maxSizes[0] != 1280 ||
            input->outputTensors[0].geometry.maxSizes[0] != 640)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        const auto capacity = output->kernel.elfSize;
        auto inherited = *input;
        std::strcpy(inherited.guid.name, "custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto status = next(&inherited, output);
        if (status != GLUE_SUCCESS && status != GLUE_INSUFFICIENT_ELF_BUFFER) return status;
        output->kernel.elfSize = capacity;
        return loadElf(output, &_binary___deepseek_v41_silu_static_gaudi2_o_start,
                       &_binary___deepseek_v41_silu_static_gaudi2_o_end);
    }
    if (operation == 28) {
        if (input->inputTensorNr != 5 || input->outputTensorNr != 2 ||
            input->inputTensors[0].geometry.maxSizes[0] != 1280)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        const auto capacity = output->kernel.elfSize;
        auto inherited = *input;
        std::strcpy(inherited.guid.name, "custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
        const auto status = next(&inherited, output);
        if (status != GLUE_SUCCESS && status != GLUE_INSUFFICIENT_ELF_BUFFER) return status;
        output->kernel.elfSize = capacity;
        return loadElf(output, &_binary___deepseek_v41_silu640_gaudi2_o_start,
                       &_binary___deepseek_v41_silu640_gaudi2_o_end);
    }
    if (operation == 26 || operation == 27) {
        if (input->inputTensorNr != 4 || input->outputTensorNr != 1) return GLUE_FAILED;
        const auto& q = input->inputTensors[1].geometry;
        if (q.dataType != DATA_U8 || q.dims != 3 || q.maxSizes[0] % 16384)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        const auto capacity = output->kernel.elfSize;
        auto inherited = *input;
        std::vector<Tensor> tensors(input->inputTensors, input->inputTensors + input->inputTensorNr);
        tensors[1].geometry.dataType = DATA_I16;
        tensors[1].geometry.maxSizes[0] /= 2;
        tensors[1].geometry.minSizes[0] /= 2;
        inherited.inputTensors = tensors.data();
        std::strcpy(inherited.guid.name, operation == 26 ?
            "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2" :
            "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        const auto status = next(&inherited, output);
        if (status != GLUE_SUCCESS && status != GLUE_INSUFFICIENT_ELF_BUFFER) return status;
        auto& mapping = output->inputTensorAccessPattern[1].mapping[0];
        mapping.a *= 2;
        mapping.start_b *= 2;
        mapping.end_b = 2 * (mapping.end_b + 1) - 1;
        output->kernel.elfSize = capacity;
        return operation == 26 ?
            loadElf(output, &_binary___deepseek_v41_expert_token_wide_bytes_gaudi2_o_start,
                    &_binary___deepseek_v41_expert_token_wide_bytes_gaudi2_o_end) :
            loadElf(output, &_binary___deepseek_v41_expert_n256_bytes_gaudi2_o_start,
                    &_binary___deepseek_v41_expert_n256_bytes_gaudi2_o_end);
    }
    if (operation == 25) {
        // Diagnostic alias isolates compiler graphs; arithmetic, geometry and
        // ELF are inherited unchanged. No production dispatch uses this GUID.
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        auto inherited = *input;
        std::strcpy(inherited.guid.name, "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2");
        return next(&inherited, output);
    }
    if (operation == 24) {
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        const auto capacity = output->kernel.elfSize;
        auto inherited = *input;
        std::strcpy(inherited.guid.name, "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2");
        const auto status = next(&inherited, output);
        if (status != GLUE_SUCCESS && status != GLUE_INSUFFICIENT_ELF_BUFFER) return status;
        const uint64_t dimensions[] = {output->indexSpaceGeometry[2], output->indexSpaceGeometry[0],
                                       output->indexSpaceGeometry[1]};
        for (unsigned i = 0; i < 3; ++i) output->indexSpaceGeometry[i] = dimensions[i];
        const unsigned axes[] = {1, 2, 0};
        for (unsigned i = 0; i < input->inputTensorNr + input->outputTensorNr; ++i) {
            auto& pattern = i < input->inputTensorNr ? output->inputTensorAccessPattern[i] :
                            output->outputTensorAccessPattern[i - input->inputTensorNr];
            for (auto& map : pattern.mapping)
                if (map.indexSpaceDim < 3) map.indexSpaceDim = axes[map.indexSpaceDim];
        }
        output->kernel.elfSize = capacity;
        return loadElf(output, &_binary___deepseek_v41_expert_token_wide_k_first_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_token_wide_k_first_gaudi2_o_end);
    }
    if (operation >= 21 && operation <= 23) {
        if (input->inputTensorNr != 4 || input->outputTensorNr != 1) return GLUE_FAILED;
        const auto& q = input->inputTensors[1].geometry;
        const auto& s = input->inputTensors[2].geometry;
        if (q.dims != 3 || q.maxSizes[0] % 8192 ||
            s.maxSizes[0] != q.maxSizes[0] / 16 + 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto next = base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if (!next) return GLUE_FAILED;
        const auto capacity = output->kernel.elfSize;
        auto inherited = *input;
        auto fake = input->outputTensors[0];
        fake.geometry.dataType = DATA_F8_143;
        fake.geometry.maxSizes[1] = q.maxSizes[0] / 64;
        inherited.outputTensors = &fake;
        if (input->nodeParams.nodeParamsSize != sizeof(int32_t) || !input->nodeParams.nodeParams) return GLUE_FAILED;
        const auto pack = *static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if (pack != 1 && pack != 2) return GLUE_FAILED;
        std::strcpy(inherited.guid.name, pack == 2 ? "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2" :
                                                  "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        auto result = next(&inherited, output);
        if (result != GLUE_SUCCESS && result != GLUE_INSUFFICIENT_ELF_BUFFER) return result;
        if (operation != 23) {
            if (input->outputTensors[0].geometry.dataType != DATA_U8 ||
                input->outputTensors[0].geometry.maxSizes[1] != q.maxSizes[0] / 8192)
                return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            output->outputTensorAccessPattern[0].mapping[1] = {2, 1, 0, 0, false};
        }
        output->kernel.elfSize = capacity;
        output->kernel.paramsNr = 1;
        output->kernel.scalarParams[0] = pack;
        if (operation == 21)
            return loadElf(output, &_binary___deepseek_v41_expert_probe_read_gaudi2_o_start,
                           &_binary___deepseek_v41_expert_probe_read_gaudi2_o_end);
        if (operation == 22)
            return loadElf(output, &_binary___deepseek_v41_expert_probe_decode_gaudi2_o_start,
                           &_binary___deepseek_v41_expert_probe_decode_gaudi2_o_end);
        return loadElf(output, &_binary___deepseek_v41_expert_probe_store_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_probe_store_gaudi2_o_end);
    }
    if(operation==18 || operation==19 || operation==69) {
        const bool bf16_kv=operation==18 || operation==69;
        const auto expected=bf16_kv?3u:2u;
        if(input->outputTensorNr!=expected)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        Tensor fake[3];
        for(unsigned i=0;i<expected;++i)fake[i]=input->outputTensors[i];
        inherited.outputTensors=fake;
        if(bf16_kv) {
            fake[1]=fake[0];fake[1].geometry.dataType=DATA_F32;
            std::strcpy(inherited.guid.name,"custom_deepseek_v41_logical_mla_vector_gaudi2");
        } else {
            inherited.outputTensorNr=1;fake[0].geometry.dataType=DATA_F32;
            std::strcpy(inherited.guid.name,"custom_deepseek_v41_selected_mla_softmax_gaudi2");
        }
        auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        if(bf16_kv) {
            output->outputTensorAccessPattern[1]={};
            output->outputTensorAccessPattern[1].allRequired=true;
        } else output->outputTensorAccessPattern[1]=output->outputTensorAccessPattern[0];
        output->kernel.elfSize=capacity;
        if(operation==69)
            return loadElf(output,&_binary___deepseek_v41_logical_scale_bf16_gaudi2_o_start,
                            &_binary___deepseek_v41_logical_scale_bf16_gaudi2_o_end);
        return operation==18
            ? loadElf(output,&_binary___deepseek_v41_logical_mla_bf16_only_gaudi2_o_start,
                            &_binary___deepseek_v41_logical_mla_bf16_only_gaudi2_o_end)
            : loadElf(output,&_binary___deepseek_v41_selected_mla_pair_softmax_gaudi2_o_start,
                            &_binary___deepseek_v41_selected_mla_pair_softmax_gaudi2_o_end);
    }
    if(operation==20) {
        if(input->inputTensorNr!=2 || input->outputTensorNr!=1)return GLUE_FAILED;
        const auto& s=input->inputTensors[0].geometry;
        if(s.dims!=3 || s.dataType!=DATA_F32 || s.maxSizes[0]!=512 || !s.maxSizes[1] || s.maxSizes[1]>64 ||
           !s.maxSizes[2] || s.maxSizes[2]>6 ||
           !shape(input->inputTensors[1],DATA_F32,{512,s.maxSizes[1],s.maxSizes[2]}) ||
           !shape(input->outputTensors[0],DATA_BF16,{512,s.maxSizes[1],s.maxSizes[2]}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=3;
        output->indexSpaceGeometry[0]=4;
        output->indexSpaceGeometry[1]=s.maxSizes[1];
        output->indexSpaceGeometry[2]=s.maxSizes[2];
        for(unsigned i=0;i<3;++i) {
            auto& pattern=i<2?output->inputTensorAccessPattern[i]:output->outputTensorAccessPattern[0];
            pattern.mapping[0]={0,128,0,127,false};
            pattern.mapping[1]={1,1,0,0,false};
            pattern.mapping[2]={2,1,0,0,false};
        }
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_mla_pair_reduce_bf16_gaudi2_o_start,
                       &_binary___deepseek_v41_mla_pair_reduce_bf16_gaudi2_o_end);
    }
    if(operation==17) {
        if(input->outputTensorNr!=1 || input->inputTensorNr!=4)return GLUE_FAILED;
        const auto physical_k=input->inputTensors[1].geometry.maxSizes[0]/64;
        const auto active_k=input->outputTensors[0].geometry.maxSizes[1];
        if(!active_k || active_k>physical_k || active_k%32)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        auto physical_output=input->outputTensors[0];
        physical_output.geometry.maxSizes[1]=physical_k;
        inherited.outputTensors=&physical_output;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_n256_sat_fp8_gaudi2");
        auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.paramsNr=1;
        output->kernel.scalarParams[0]=active_k;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_n256_sat_active_k_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_n256_sat_active_k_gaudi2_o_end);
    }
    if(operation==16) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2");
        auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_expert_token_wide_sat16_fp8_gaudi2_o_start,
                       &_binary___deepseek_v41_expert_token_wide_sat16_fp8_gaudi2_o_end);
    }
    if(operation==15) {
        auto next=base<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
        if(!next)return GLUE_FAILED;
        const auto capacity=output->kernel.elfSize;
        auto inherited=*input;
        std::strcpy(inherited.guid.name,"custom_deepseek_v41_q_scale_rope_gaudi2");
        auto result=next(&inherited,output);
        if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
        output->indexSpaceGeometry[0]=input->outputTensors[0].geometry.maxSizes[0]/128;
        for(unsigned operand=0;operand<2;++operand) {
            auto& mapping=output->inputTensorAccessPattern[operand].mapping[0];
            mapping.a=128;mapping.start_b=0;mapping.end_b=127;
        }
        auto& mapping=output->outputTensorAccessPattern[0].mapping[0];
        mapping.a=128;mapping.start_b=0;mapping.end_b=127;
        output->kernel.elfSize=capacity;
        return loadElf(output,&_binary___deepseek_v41_q_scale_rope_tiled_gaudi2_o_start,
                       &_binary___deepseek_v41_q_scale_rope_tiled_gaudi2_o_end);
    }
    if(operation==13 || operation==14) {
        const bool mark=operation==13;
        if(input->inputTensorNr!=(mark?4u:7u)||input->outputTensorNr!=(mark?1u:3u))return GLUE_FAILED;
        const auto tokens=input->inputTensors[mark?0:2].geometry.maxSizes[1];
        if(!tokens||tokens>6||!shape(input->inputTensors[mark?0:2],DATA_I32,{512,tokens})||
            !shape(input->inputTensors[mark?1:3],DATA_I32,{tokens})||
            !shape(input->inputTensors[mark?3:5],DATA_I32,{33024,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(mark) {
            if(!shape(input->inputTensors[2],DATA_I32,{tokens})||
               !shape(input->outputTensors[0],DATA_I32,{640,tokens}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        } else {
            const auto main_rows=input->inputTensors[1].geometry.maxSizes[1];
            const auto pages=input->inputTensors[4].geometry.maxSizes[0];
            if(!input->nodeParams.nodeParams||input->nodeParams.nodeParamsSize!=4)return GLUE_FAILED;
            const int ratio=*static_cast<int32_t*>(input->nodeParams.nodeParams);
            if((ratio!=1&&ratio!=2)||!main_rows||!pages||pages>8192||
               !shape(input->inputTensors[0],DATA_U8,{528,256})||
               !shape(input->inputTensors[1],DATA_U8,{288,main_rows})||
               !shape(input->inputTensors[4],DATA_I32,{pages})||
               !shape(input->inputTensors[6],DATA_I32,{640,tokens})||
               !shape(input->outputTensors[0],DATA_BF16,{512,640,tokens})||
               !shape(input->outputTensors[1],DATA_F32,{512,640,tokens})||
               !shape(input->outputTensors[2],DATA_F32,{640,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
        }
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=640;output->indexSpaceGeometry[1]=tokens;
        for(unsigned i=0;i<input->inputTensorNr;++i)output->inputTensorAccessPattern[i].allRequired=true;
        if(mark) {
            auto& a=output->outputTensorAccessPattern[0];a.mapping[0]={0,1,0,0,false};a.mapping[1]={1,1,0,0,false};
            output->kernel.paramsNr=0;
            return loadElf(output,&_binary___deepseek_v41_mla_slotmap_gaudi2_o_start,
                           &_binary___deepseek_v41_mla_slotmap_gaudi2_o_end);
        }
        for(unsigned i=0;i<3;++i)output->outputTensorAccessPattern[i].allRequired=true;
        return loadElf(output,&_binary___deepseek_v41_mla_slotmap_scatter_gaudi2_o_start,
                       &_binary___deepseek_v41_mla_slotmap_scatter_gaudi2_o_end);
    }
    if (operation == 10) {
        if (input->inputTensorNr!=5 || input->outputTensorNr!=1) return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        if (tokens<1 || tokens>6 || rows<1 || rows>1048576 ||
            !shape(input->inputTensors[0],DATA_U8,{288,rows}) ||
            !shape(input->inputTensors[1],DATA_U8,{68,rows}) ||
            !shape(input->inputTensors[2],DATA_BF16,{512,tokens}) ||
            !shape(input->inputTensors[3],DATA_BF16,{128,tokens}) ||
            !shape(input->inputTensors[4],DATA_I32,{tokens}) ||
            !shape(input->outputTensors[0],DATA_I32,{36,tokens})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=36;output->indexSpaceGeometry[1]=tokens;
        for(unsigned i=0;i<5;++i) output->inputTensorAccessPattern[i].allRequired=true;
        auto& access=output->outputTensorAccessPattern[0];
        access.mapping[0]={0,1,0,0,false};access.mapping[1]={1,1,0,0,false};
        output->kernel.paramsNr=0;
        return loadElf(output,&_binary___deepseek_v41_fp4_cache_write_rows_gaudi2_o_start,
                       &_binary___deepseek_v41_fp4_cache_write_rows_gaudi2_o_end);
    }
    if (operation == 12) {
        if(input->inputTensorNr!=4 || input->outputTensorNr!=1 ||
           !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize!=4) return GLUE_FAILED;
        const auto rows=input->inputTensors[0].geometry.maxSizes[1];
        const auto pages=input->inputTensors[1].geometry.maxSizes[0];
        const auto cols=input->inputTensors[2].geometry.maxSizes[0],batch=input->inputTensors[2].geometry.maxSizes[1];
        const auto tokens=input->inputTensors[3].geometry.maxSizes[1];
        const int ratio=*static_cast<int32_t*>(input->nodeParams.nodeParams);
        if(!rows||!pages||pages>8192||!cols||cols>2048||!batch||batch>128||!tokens||tokens>6||
           (ratio!=1&&ratio!=2)||!shape(input->inputTensors[0],DATA_U8,{68,rows})||
           !shape(input->inputTensors[1],DATA_I32,{pages})||!shape(input->inputTensors[2],DATA_I32,{cols,batch})||
           !shape(input->inputTensors[3],DATA_I32,{36,tokens})||
           !shape(input->outputTensors[0],DATA_BF16,{128,cols,batch}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=cols;output->indexSpaceGeometry[1]=batch;
        for(unsigned i=0;i<4;++i)output->inputTensorAccessPattern[i].allRequired=true;
        auto& a=output->outputTensorAccessPattern[0];a.mapping[0]={0,0,0,127,false};
        a.mapping[1]={0,1,0,0,false};a.mapping[2]={1,1,0,0,false};
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
        return loadElf(output,&_binary___deepseek_v41_index_keys_write_ordered_gaudi2_o_start,
                       &_binary___deepseek_v41_index_keys_write_ordered_gaudi2_o_end);
    }
    if (operation == 9 || operation == 11) {
        if (input->inputTensorNr != 7 || input->outputTensorNr != 3 ||
            !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != 4) return GLUE_FAILED;
        const auto tokens=input->inputTensors[2].geometry.maxSizes[1];
        const auto main_rows=input->inputTensors[1].geometry.maxSizes[1];
        const auto pages=input->inputTensors[4].geometry.maxSizes[0];
        const int ratio=*static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if (tokens<1 || tokens>6 || !main_rows || main_rows>0x7ffffdffULL || !pages || pages>8192 ||
            (ratio!=1 && ratio!=2) || !shape(input->inputTensors[0],operation==9?DATA_BF16:DATA_U8,{operation==9?512u:528u,256}) ||
            !shape(input->inputTensors[1],operation==9?DATA_BF16:DATA_U8,{operation==9?512u:288u,main_rows}) ||
            !shape(input->inputTensors[2],DATA_I32,{512,tokens}) ||
            !shape(input->inputTensors[3],DATA_I32,{tokens}) ||
            !shape(input->inputTensors[4],DATA_I32,{pages}) ||
            !shape(input->inputTensors[5],DATA_I32,{tokens}) ||
            !(operation==9 ? shape(input->inputTensors[6],DATA_I32,{tokens})
                           : shape(input->inputTensors[6],DATA_I32,{36,tokens})) ||
            !shape(input->outputTensors[0],DATA_BF16,{512,640,tokens}) ||
            !shape(input->outputTensors[1],DATA_F32,{512,640,tokens}) ||
            !shape(input->outputTensors[2],DATA_F32,{640,tokens})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank=2;output->indexSpaceGeometry[0]=640;output->indexSpaceGeometry[1]=tokens;
        for(unsigned i=0;i<7;++i) output->inputTensorAccessPattern[i].allRequired=true;
        for(unsigned i=0;i<3;++i) {
            auto& a=output->outputTensorAccessPattern[i];
            if(i<2) {
                a.mapping[0]={0,0,0,511,false};
                a.mapping[1]={0,1,0,0,false};a.mapping[2]={1,1,0,0,false};
            } else {
                a.mapping[0]={0,1,0,0,false};a.mapping[1]={1,1,0,0,false};
            }
        }
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
        if(operation==11) return loadElf(output,&_binary___deepseek_v41_logical_mla_write_ordered_gaudi2_o_start,
                                       &_binary___deepseek_v41_logical_mla_write_ordered_gaudi2_o_end);
        return loadElf(output,&_binary___deepseek_v41_logical_mla_decoded_gaudi2_o_start,
                       &_binary___deepseek_v41_logical_mla_decoded_gaudi2_o_end);
    }
    if (operation == 7) {
        if (input->inputTensorNr != 3 || input->outputTensorNr != 4) return GLUE_FAILED;
        const auto tokens = input->inputTensors[0].geometry.maxSizes[1];
        if (tokens < 1 || tokens > 6 || !shape(input->inputTensors[0], DATA_I32, {512,tokens}) ||
            !shape(input->inputTensors[1], DATA_I32, {tokens}) ||
            !shape(input->inputTensors[2], DATA_I32, {tokens}) ||
            !shape(input->outputTensors[0], DATA_I32, {512*tokens}) ||
            !shape(input->outputTensors[1], DATA_I32, {257+512*tokens}) ||
            !shape(input->outputTensors[2], DATA_I32, {640,tokens}) ||
            !shape(input->outputTensors[3], DATA_I32, {4})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 1; output->indexSpaceGeometry[0] = 1;
        for (unsigned i=0;i<3;++i) output->inputTensorAccessPattern[i].allRequired = true;
        for (unsigned i=0;i<4;++i) output->outputTensorAccessPattern[i].allRequired = true;
        output->kernel.paramsNr = 0;
        return loadElf(output, &_binary___deepseek_v41_mla_union_metadata_gaudi2_o_start,
                       &_binary___deepseek_v41_mla_union_metadata_gaudi2_o_end);
    }
    if (operation == 8) {
        if (input->inputTensorNr != 7 || input->outputTensorNr != 3 ||
            !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != 4) return GLUE_FAILED;
        const auto tokens = input->inputTensors[4].geometry.maxSizes[1];
        const auto main_rows = input->inputTensors[1].geometry.maxSizes[1];
        const auto page_rows = input->inputTensors[6].geometry.maxSizes[0];
        const int ratio = *static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if (tokens < 1 || tokens > 6 || (ratio != 1 && ratio != 2) ||
            !main_rows || main_rows > 0x7ffffdffULL || !page_rows || page_rows > 8192 ||
            !shape(input->inputTensors[0], DATA_U8, {528,256}) ||
            !shape(input->inputTensors[1], DATA_U8, {288,main_rows}) ||
            !shape(input->inputTensors[2], DATA_I32, {512*tokens}) ||
            !shape(input->inputTensors[3], DATA_I32, {257+512*tokens}) ||
            !shape(input->inputTensors[4], DATA_I32, {640,tokens}) ||
            !shape(input->inputTensors[5], DATA_I32, {4}) ||
            !shape(input->inputTensors[6], DATA_I32, {page_rows}) ||
            !shape(input->outputTensors[0], DATA_BF16, {512,640,tokens}) ||
            !shape(input->outputTensors[1], DATA_F32, {512,640,tokens}) ||
            !shape(input->outputTensors[2], DATA_F32, {640,tokens})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 1; output->indexSpaceGeometry[0] = 128;
        // CSR destinations are disjoint by construction but are not affine.
        // Keep conservative full tensor bounds and explicit sparse access;
        // compiler slicing must not substitute a narrower affine footprint.
        for (unsigned i=0;i<7;++i) {
            auto& access=output->inputTensorAccessPattern[i];access.sparseAccess=true;
            for (unsigned d=0;d<input->inputTensors[i].geometry.dims;++d)
                access.mapping[d]={0,0,0,float(input->inputTensors[i].geometry.maxSizes[d]-1),false};
        }
        for (unsigned i=0;i<3;++i) {
            auto& access=output->outputTensorAccessPattern[i];access.sparseAccess=true;
            for (unsigned d=0;d<input->outputTensors[i].geometry.dims;++d)
                access.mapping[d]={0,0,0,float(input->outputTensors[i].geometry.maxSizes[d]-1),false};
        }
        output->kernel.paramsNr=1;output->kernel.scalarParams[0]=ratio;
        return loadElf(output, &_binary___deepseek_v41_mla_union_scatter_gaudi2_o_start,
                       &_binary___deepseek_v41_mla_union_scatter_gaudi2_o_end);
    }
    if (operation >= 3) {
        for (unsigned i = 0; i < input->inputTensorNr; ++i)
            output->inputTensorAccessPattern[i].allRequired = true;
        for (unsigned i = 0; i < input->outputTensorNr; ++i)
            output->outputTensorAccessPattern[i].allRequired = true;
        output->kernel.paramsNr = 0;
        if (operation == 3) {
            if (input->inputTensorNr != 4 || input->outputTensorNr != 3) return GLUE_FAILED;
            const auto h = input->inputTensors[0].geometry.maxSizes[0];
            const auto tokens = input->inputTensors[0].geometry.maxSizes[1];
            if (h != 5120 || tokens < 1 || tokens > 6 ||
                !shape(input->inputTensors[0], DATA_F8_143, {h, tokens}) ||
                !shape(input->inputTensors[1], DATA_F32, {1, tokens}) ||
                !shape(input->inputTensors[2], DATA_F32, {6, tokens}) ||
                !shape(input->inputTensors[3], DATA_I32, {64, 10}) ||
                !shape(input->outputTensors[0], DATA_F8_143, {h, 6, 36}) ||
                !shape(input->outputTensors[1], DATA_F32, {6, 36}) ||
                !shape(input->outputTensors[2], DATA_F32, {6, 36})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank = 1;
            output->indexSpaceGeometry[0] = tokens * 6;
            return loadElf(output, &_binary___deepseek_v41_unique_owner_pack_fp8_gaudi2_o_start,
                           &_binary___deepseek_v41_unique_owner_pack_fp8_gaudi2_o_end);
        }
        if (operation == 6) {
            if (input->inputTensorNr != 2 || input->outputTensorNr != 1) return GLUE_FAILED;
            const auto h = input->outputTensors[0].geometry.maxSizes[0];
            const auto tokens = input->outputTensors[0].geometry.maxSizes[1];
            if (h != 5120 || tokens < 1 || tokens > 6 ||
                !shape(input->inputTensors[0], DATA_BF16, {h, 6, 36}) ||
                !shape(input->inputTensors[1], DATA_I32, {64, 10}) ||
                !shape(input->outputTensors[0], DATA_BF16, {h, tokens})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank = 2;
            output->indexSpaceGeometry[0] = h / 128;
            output->indexSpaceGeometry[1] = tokens;
            return loadElf(output, &_binary___deepseek_v41_unique_restore_simple_bf16_gaudi2_o_start,
                           &_binary___deepseek_v41_unique_restore_simple_bf16_gaudi2_o_end);
        }
        if (!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != 4) return GLUE_FAILED;
        const int owner = *static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if (owner < 0 || owner >= 36) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->kernel.paramsNr = 1;
        output->kernel.scalarParams[0] = owner;
        if (operation == 4) {
            if (input->inputTensorNr != 5 || input->outputTensorNr != 2) return GLUE_FAILED;
            const auto i = input->outputTensors[0].geometry.maxSizes[0];
            const auto experts = input->inputTensors[3].geometry.maxSizes[2];
            if (!i || i % 128 || i > 1280 || !experts || experts > 384 ||
                !shape(input->inputTensors[0], DATA_F32, {2*i, 6}) ||
                !shape(input->inputTensors[1], DATA_I32, {64, 10}) ||
                !shape(input->inputTensors[2], DATA_F32, {6, 36}) ||
                !shape(input->inputTensors[3], DATA_BF16, {256, 2*i/256, experts}) ||
                !shape(input->inputTensors[4], DATA_F32, {6, 36}) ||
                !shape(input->outputTensors[0], DATA_F8_143, {i, 6}) ||
                !shape(input->outputTensors[1], DATA_F32, {1, 6})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            output->indexSpaceRank = 1;
            output->indexSpaceGeometry[0] = 6;
            return loadElf(output, &_binary___deepseek_v41_unique_silu_quant_fp8_gaudi2_o_start,
                           &_binary___deepseek_v41_unique_silu_quant_fp8_gaudi2_o_end);
        }
        if (input->inputTensorNr != 4 || input->outputTensorNr != 1) return GLUE_FAILED;
        const auto h = input->outputTensors[0].geometry.maxSizes[0];
        const auto experts = input->inputTensors[3].geometry.maxSizes[2];
        if (h != 5120 || !experts || experts > 384 ||
            !shape(input->inputTensors[0], DATA_F32, {h, 6}) ||
            !shape(input->inputTensors[1], DATA_I32, {64, 10}) ||
            !shape(input->inputTensors[2], DATA_F32, {1, 6}) ||
            !shape(input->inputTensors[3], DATA_BF16, {256, h/256, experts}) ||
            !shape(input->outputTensors[0], DATA_BF16, {h, 6})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 2;
        output->indexSpaceGeometry[0] = h / 64;
        output->indexSpaceGeometry[1] = 1;
        return loadElf(output, &_binary___deepseek_v41_unique_scale_fp8_gaudi2_o_start,
                       &_binary___deepseek_v41_unique_scale_fp8_gaudi2_o_end);
    }
    if (operation == 0) {
        if (input->inputTensorNr != 2 || input->outputTensorNr != 2 ||
            !input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != 4) return GLUE_FAILED;
        const auto tokens = input->inputTensors[0].geometry.maxSizes[1];
        const auto experts = *static_cast<const int32_t*>(input->nodeParams.nodeParams);
        if (tokens < 1 || tokens > 6 || experts < 1 || experts >= 65535 ||
            !shape(input->inputTensors[0], DATA_I32, {6, tokens}) ||
            !shape(input->inputTensors[1], DATA_I32, {2}) ||
            !shape(input->outputTensors[0], DATA_I32, {64, 10}) ||
            !shape(input->outputTensors[1], DATA_I16, {64, 11})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 1;
        output->indexSpaceGeometry[0] = 1;
        for (unsigned i = 0; i < 2; ++i) {
            output->inputTensorAccessPattern[i].allRequired = true;
            output->outputTensorAccessPattern[i].allRequired = true;
        }
        output->kernel.paramsNr = 1;
        output->kernel.scalarParams[0] = experts;
        return loadElf(output, &_binary___deepseek_v41_unique_control_gaudi2_o_start,
                       &_binary___deepseek_v41_unique_control_gaudi2_o_end);
    }
    if (input->inputTensorNr != 3 || input->outputTensorNr != 1) return GLUE_FAILED;
    if (operation == 1) {
        if (!shape(input->inputTensors[0], DATA_I32, {65536, 6}) ||
            !shape(input->inputTensors[1], DATA_I32, {4, 2048}) ||
            !shape(input->inputTensors[2], DATA_I32, {64, 10}) ||
            !shape(input->outputTensors[0], DATA_I32, {65536})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 1;
        output->indexSpaceGeometry[0] = 2048;
    } else {
        if (!shape(input->inputTensors[0], DATA_I32, {64, 10}) ||
            !shape(input->inputTensors[1], DATA_I16, {64, 11}) ||
            !shape(input->inputTensors[2], DATA_I32, {65536}) ||
            !shape(input->outputTensors[0], DATA_I32, {36, 1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        output->indexSpaceRank = 1;
        output->indexSpaceGeometry[0] = 1;
    }
    for (unsigned i = 0; i < 3; ++i) output->inputTensorAccessPattern[i].allRequired = true;
    output->outputTensorAccessPattern[0].allRequired = true;
    output->kernel.paramsNr = 0;
    if (operation == 1)
        return loadElf(output, &_binary___deepseek_v41_unique_program_gaudi2_o_start,
                       &_binary___deepseek_v41_unique_program_gaudi2_o_end);
    return loadElf(output, &_binary___deepseek_v41_unique_ids_gaudi2_o_start,
                   &_binary___deepseek_v41_unique_ids_gaudi2_o_end);
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* parameters, tpc_lib_api::ShapeInferenceOutput* output) {
    const auto* guid = parameters->pGuid ? parameters->pGuid : &parameters->guid;
    if (kind(guid->name) >= 0) return tpc_lib_api::GLUE_SUCCESS;
    auto next = base<decltype(&GetShapeInference)>("GetShapeInference");
    return next ? next(device, parameters, output) : tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
                                                    tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (kind(p->guid.name) >= 0) return tpc_lib_api::GLUE_FAILED;
    auto next = base<decltype(&GetSuggestedManipulation)>("GetSuggestedManipulation");
    return next ? next(p, suggestion) : tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count) {
    if (kind(p->guid.name) < 0) {
        auto next = base<decltype(&GetSupportedDataLayouts)>("GetSupportedDataLayouts");
        return next ? next(p, layouts, count) : tpc_lib_api::GLUE_FAILED;
    }
    if (!count) return tpc_lib_api::GLUE_FAILED;
    *count = 1;
    if (layouts) {
        for (unsigned i = 0; i < layouts->inputTensorNr; ++i)
            std::memset(layouts->inputs[i].layout, 'x', sizeof(layouts->inputs[i].layout));
        for (unsigned i = 0; i < layouts->outputTensorNr; ++i)
            std::memset(layouts->outputs[i].layout, 'x', sizeof(layouts->outputs[i].layout));
    }
    return tpc_lib_api::GLUE_SUCCESS;
}
}
