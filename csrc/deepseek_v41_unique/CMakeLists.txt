# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(deepseek_v41_unique LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC headers")
set(EMBEDDED_OBJECTS)
foreach(NAME deepseek_v41_expert_w13_k512_gaudi2 deepseek_v41_expert_w2_k512_gaudi2 deepseek_v41_mhc_gates_post_gaudi2 deepseek_v41_expert_w13_split_scale_gaudi2 deepseek_v41_expert_w2_split_scale_gaudi2 deepseek_v41_weighted_row_parameters_gaudi2 deepseek_v41_norm_prepared_weighted_stats_gaudi2 deepseek_v41_mhc_post_weighted_stats_gaudi2 deepseek_v41_norm_from_weighted_stats_gaudi2 deepseek_v41_main_single_bank_gather_gaudi2 deepseek_v41_control_pair_quant_rrms_gaudi2 deepseek_v41_control_pair_finish_gaudi2 deepseek_v41_coherent_swa_keys_gaudi2 deepseek_v41_coherent_swa_pair_softmax_gaudi2 deepseek_v41_coherent_swa_finish_gaudi2 deepseek_v41_coherent_swa_decode_gaudi2 deepseek_v41_coherent_swa_softmax_gaudi2 deepseek_v41_expert_pair_channels_gaudi2 deepseek_v41_expert_scaled_silu_quant_gaudi2 deepseek_v41_attention_norm_quant_remat_gaudi2 deepseek_v41_control_statistics_gaudi2 deepseek_v41_control_quant_rrms_gaudi2 deepseek_v41_expert_token_wide_explicit_steps_gaudi2 deepseek_v41_mhc_deferred_post_gaudi2 deepseek_v41_silu_decode_c6_gaudi2 deepseek_v41_norm_roundtrip_tiled_gaudi2 deepseek_v41_norm_statistics_gaudi2 deepseek_v41_norm_roundtrip_gaudi2 deepseek_v41_probability_parts_gaudi2 deepseek_v41_probability_draw_gaudi2 deepseek_v41_expert_cached_gather_bf16_gaudi2 deepseek_v41_fixed_dense_quant_gaudi2 deepseek_v41_router_shared_scaled_gaudi2 deepseek_v41_rope_all_rows_gaudi2 deepseek_v41_rope_inverse_all_rows_gaudi2 deepseek_v41_expert_token_wide_n512_gaudi2 deepseek_v41_expert_down_n512_gaudi2 deepseek_v41_mhc_statistics_gaudi2 deepseek_v41_mhc_statistics_finish_gaudi2 deepseek_v41_input_norm_bf16_gaudi2 deepseek_v41_engram_batch_gaudi2 deepseek_v41_mla_adjacent_softmax_gaudi2 deepseek_v41_mla_adjacent_finish_gaudi2 deepseek_v41_swa_keys_reuse_gaudi2 deepseek_v41_swa_source_reuse_gaudi2 deepseek_v41_weighted_finish_mask_gaudi2 deepseek_v41_compressor_sequence_bf16_gaudi2 deepseek_v41_expert_pair_silu_gaudi2 deepseek_v41_weighted_active_blocks_gaudi2 deepseek_v41_weighted_score_bounds_gaudi2 deepseek_v41_weighted_sparse_bins_gaudi2 deepseek_v41_swa_cached_publish_gaudi2 deepseek_v41_swa_cached_reuse_gaudi2 deepseek_v41_swa_batch_cache_write_gaudi2 deepseek_v41_silu_split_activate_gaudi2 deepseek_v41_main_batch_publish_gather_gaudi2 deepseek_v41_main_batch_reuse_gather_gaudi2 deepseek_v41_mla_stream_metadata_gaudi2 deepseek_v41_mla_stream_decode_gaudi2 deepseek_v41_mla_stream_softmax_gaudi2 deepseek_v41_q_bf16_rope_tiled_gaudi2 deepseek_v41_logical_full_row_gaudi2 deepseek_v41_silu_scalar_cache_gaudi2 deepseek_v41_logical_fp4_table_gaudi2 deepseek_v41_logical_row_cache_gaudi2 deepseek_v41_scale_reduce_shared_gaudi2 deepseek_v41_mtp_dequant_fp8_gaudi2 deepseek_v41_weighted_mass_bins_gaudi2 deepseek_v41_weighted_mass_advance_gaudi2 deepseek_v41_weighted_mass_finish_gaudi2
    deepseek_v41_fp4_paged_decoded_rows_gaudi2 deepseek_v41_woa_scale_dense_quant_gaudi2
    deepseek_v41_mla_exp_bf16_gaudi2 deepseek_v41_mla_exp_normalize_gaudi2
    deepseek_v41_mhc_control_tiles_gaudi2 deepseek_v41_mhc_control_tiles_finish_gaudi2
    deepseek_v41_expert_physical_role_silu_gaudi2
    deepseek_v41_journal_copy_gaudi2
    deepseek_v41_bounded_nucleus_parts_gaudi2
    deepseek_v41_bounded_local_distribution_gaudi2
    deepseek_v41_vocab_lane_candidates_gaudi2
    deepseek_v41_mhc_control_reuse_gaudi2 deepseek_v41_logical_merge_cache_gaudi2 deepseek_v41_mhc_mme_epilogue_gaudi2 deepseek_v41_unique_control_gaudi2 deepseek_v41_unique_program_gaudi2 deepseek_v41_unique_ids_gaudi2
        deepseek_v41_unique_owner_pack_fp8_gaudi2 deepseek_v41_unique_silu_quant_fp8_gaudi2
        deepseek_v41_unique_scale_fp8_gaudi2 deepseek_v41_unique_restore_simple_bf16_gaudi2
        deepseek_v41_mla_union_metadata_gaudi2 deepseek_v41_mla_union_scatter_gaudi2 deepseek_v41_mla_merged_gaudi2
        deepseek_v41_logical_mla_decoded_gaudi2 deepseek_v41_fp4_cache_write_rows_gaudi2
        deepseek_v41_logical_mla_write_ordered_gaudi2 deepseek_v41_index_keys_write_ordered_gaudi2
        deepseek_v41_mla_slotmap_gaudi2 deepseek_v41_mla_slotmap_scatter_gaudi2 deepseek_v41_q_scale_rope_tiled_gaudi2
        deepseek_v41_expert_token_wide_sat16_fp8_gaudi2 deepseek_v41_expert_n256_sat_active_k_gaudi2
        deepseek_v41_logical_mla_bf16_only_gaudi2 deepseek_v41_selected_mla_pair_softmax_gaudi2
        deepseek_v41_mla_pair_reduce_bf16_gaudi2
        deepseek_v41_expert_probe_read_gaudi2 deepseek_v41_expert_probe_decode_gaudi2
        deepseek_v41_expert_probe_store_gaudi2 deepseek_v41_expert_token_wide_k_first_gaudi2
        deepseek_v41_expert_token_wide_bytes_gaudi2 deepseek_v41_expert_n256_bytes_gaudi2
        deepseek_v41_silu640_gaudi2 deepseek_v41_silu_activate_tile_gaudi2 deepseek_v41_silu_quant_tile_gaudi2
        deepseek_v41_vocab_radix_threshold_gaudi2 deepseek_v41_vocab_radix_emit_gaudi2 deepseek_v41_vocab_filter_gaudi2 deepseek_v41_vocab_filter_emit_gaudi2 deepseek_v41_expert_three_route_sat_fp8_gaudi2
        deepseek_v41_expert_three_route_scale_reduce_gaudi2 deepseek_v41_peer_post_norm_quant_gaudi2 deepseek_v41_peer_post_collapse_gaudi2 deepseek_v41_silu_static_gaudi2 deepseek_v41_kv_norm_rope_publish_gaudi2 deepseek_v41_pv_rope_f32_gaudi2
        deepseek_v41_nucleus_mass_gaudi2 deepseek_v41_nucleus_draw_gaudi2 deepseek_v41_nucleus_finish_gaudi2 deepseek_v41_qkv_norm_publish_gaudi2 deepseek_v41_mla_hash_owners_gaudi2
        deepseek_v41_mla_hash_decode_gaudi2 deepseek_v41_mla_hash_gather_gaudi2
        deepseek_v41_expert_physical_silu_gaudi2 deepseek_v41_logical_mla_coord_cached_gaudi2
        deepseek_v41_mla_shared_owners_gaudi2 deepseek_v41_mla_shared_decode_gaudi2 deepseek_v41_mla_shared_softmax_gaudi2
        deepseek_v41_expert_token_wide_group_pipe_gaudi2 deepseek_v41_expert_down_group_pipe_gaudi2
        deepseek_v41_mla_compact_metadata_gaudi2 deepseek_v41_mla_compact_prefix_gaudi2
        deepseek_v41_mla_compact_decode_gaudi2 deepseek_v41_mla_compact_softmax_gaudi2
        deepseek_v41_silu_channel_vector_gaudi2 deepseek_v41_expert_token_wide_unroll_steps_gaudi2 deepseek_v41_logical_main_mirror_gaudi2 deepseek_v41_logical_scale_cache_gaudi2 deepseek_v41_candidate_mirror_keys_gaudi2
        deepseek_v41_expert_token_wide_affine_route_gaudi2 deepseek_v41_logical_scale_bf16_gaudi2 deepseek_v41_expert_token_wide_k64_sat_gaudi2 deepseek_v41_expert_n256_k64_sat_gaudi2 deepseek_v41_selected_mla_stacked_softmax_gaudi2 deepseek_v41_mla_stacked_reduce_bf16_gaudi2)
    set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")
    set(EMBEDDED "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_embedded.o")
    set(INPUT_NORM_DEPENDENCIES)
    if(NAME STREQUAL "deepseek_v41_mhc_deferred_post_gaudi2")
        list(APPEND INPUT_NORM_DEPENDENCIES
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_mhc_mme_post_collapse_gaudi2.c")
    endif()
    if(NAME STREQUAL "deepseek_v41_silu_decode_c6_gaudi2")
        list(APPEND INPUT_NORM_DEPENDENCIES
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_expert_silu_decode_fp8_gaudi2.c")
    endif()
    if(NAME STREQUAL "deepseek_v41_fixed_dense_quant_gaudi2")
        list(APPEND INPUT_NORM_DEPENDENCIES
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_quant_roundtrip_bf16_gaudi2.c")
    endif()
    if(NAME STREQUAL "deepseek_v41_router_shared_scaled_gaudi2")
        list(APPEND INPUT_NORM_DEPENDENCIES
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_router_logits_top6_gaudi2.c")
    endif()
    if(NAME STREQUAL "deepseek_v41_input_norm_bf16_gaudi2")
        list(APPEND INPUT_NORM_DEPENDENCIES
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_attention_norm_bf16_gaudi2.c")
    endif()
    add_custom_command(OUTPUT "${OBJECT}"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2
            -I${TPC_INCLUDE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${NAME}.c" -c -o "${OBJECT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${NAME}.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/block32_roundtrip_inline.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deepseek_v41_coherent_swa_decode_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deepseek_v41_coherent_swa_softmax_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_mhc_gates_post_gaudi2.c"
            ${INPUT_NORM_DEPENDENCIES}
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_expert_n256.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_mhc_gates_f32_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_engram_hash_gather_bf16_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_expert_n256_silu_quant_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deepseek_v41_weighted_mass_bins_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deepseek_v41_weighted_mass_finish_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_selected_kv_vector.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_swa_pack.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_silu_tile_math.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_main_reuse_gather_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_selected_mla_softmax_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_selected_mla_softmax_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/route_metadata.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/route_offsets.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mla_union_codec.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/expert_decode_probe.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_selected_kv_vector.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_fp4_pack.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_fp4_paged_decoded_write_bf16_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_woa_scale_dense_quant_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_fp4_cache_write_bf16_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_index_keys_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_q_scale_rope_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_expert_n256.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_expert_n256_silu_quant_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_score_order.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_selection_bitmap.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_prefill_topk_threshold_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_prefill_topk_emit_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_selected_mla_softmax_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_silu_activate_tile_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_expert_n256_scale_reduce_direct_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_silu_quant_tile_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_peer_post_norm_quant_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_peer_post_collapse_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_peer_post_math.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_swa_pack.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_rope_bf16.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v4_qnorm_rope_kv_pack_bf16.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_kv_norm_rope_publish_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_qkv_norm_publish_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/kernels/deepseek_v41_qnorm_quant_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../flashinfer_gaudi/kernels/norm_math_gaudi2.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../deepseek_v4/include/deepseek_v41_silu_tile_math.h" VERBATIM)
    add_custom_command(OUTPUT "${EMBEDDED}"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            "./${NAME}.o" "${NAME}_embedded.o"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" DEPENDS "${OBJECT}" VERBATIM)
    set_source_files_properties("${EMBEDDED}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND EMBEDDED_OBJECTS "${EMBEDDED}")
endforeach()
add_library(deepseek_v41_unique_kernels SHARED
    ../deepseek_v4/host/deepseek_v41_mhc_gates_post_gaudi2.cpp
    ../deepseek_v4/host/deepseek_v41_mhc_weighted_stats_gaudi2.cpp
    main_batch_publish_host.cpp main_batch_reuse_host.cpp swa_only_reuse_host.cpp host.cpp mtp_dequant_fp8_host.cpp
    ../deepseek_v4/host/deepseek_v41_silu_tile_gaudi2.cpp
    ../deepseek_v4/host/deepseek_v41_peer_post_norm_quant_gaudi2.cpp
    ../deepseek_v4/host/deepseek_v41_peer_post_collapse_gaudi2.cpp
    ../deepseek_v4/host/deepseek_v41_kv_norm_rope_publish_gaudi2.cpp
    ../deepseek_v4/host/deepseek_v41_qkv_norm_publish_gaudi2.cpp ${EMBEDDED_OBJECTS})
target_include_directories(deepseek_v41_unique_kernels PRIVATE /usr/include/habanalabs ${TPC_INCLUDE_DIR})
target_compile_options(deepseek_v41_unique_kernels PRIVATE -Wall -Wextra -Werror)
target_link_options(deepseek_v41_unique_kernels PRIVATE -Wl,-z,noexecstack -Wl,--no-undefined)

target_link_libraries(deepseek_v41_unique_kernels PRIVATE ${CMAKE_DL_LIBS})
