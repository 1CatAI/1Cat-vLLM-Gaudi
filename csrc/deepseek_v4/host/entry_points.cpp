/**********************************************************************
Copyright (c) 2024 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include "deepseek_v41_index_gaudi2.hpp"
#include "deepseek_v41_index_reduce_gaudi2.hpp"
#include "deepseek_v41_index_keys_gaudi2.hpp"
#include "deepseek_v41_quant_roundtrip_gaudi2.hpp"
#include "deepseek_v41_expert_n256_gaudi2.hpp"
#include "deepseek_v41_dynamic_quant_bf16_gaudi2.hpp"
#include "deepseek_v41_ffn_norm_quant_gaudi2.hpp"
#include "deepseek_v41_selected_mla_gaudi2.hpp"
#include "deepseek_v41_selected_kv_gaudi2.hpp"
#include "deepseek_v41_head_attention_gaudi2.hpp"
#include "deepseek_v41_kv_pack_gaudi2.hpp"
#include "deepseek_v41_rope_gaudi2.hpp"
#include "deepseek_v41_prefix_layout_gaudi2.hpp"
#include "deepseek_v41_swa_pack_gaudi2.hpp"
#include "deepseek_v41_fp4_pack_gaudi2.hpp"
#include "deepseek_v41_csa2_prep_gaudi2.hpp"
#include "deepseek_v41_compressor_pair_gaudi2.hpp"
#include "deepseek_v41_decoded_kv_gaudi2.hpp"
#include "deepseek_v41_control_gemv_gaudi2.hpp"
#include "deepseek_v41_control_gemv_rrms_gaudi2.hpp"
#include "deepseek_v41_engram_hash_gather_bf16_gaudi2.hpp"
#include "deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2.hpp"
#include "deepseek_v41_mla_gaudi2.hpp"
#include "deepseek_v41_woa_gaudi2.hpp"
#include "deepseek_v41_dense_gaudi2.hpp"
#include "deepseek_v41_dense_pair_scale_gaudi2.hpp"
#include "deepseek_v41_q_scale_rope_gaudi2.hpp"
#include "deepseek_v41_qnorm_quant_gaudi2.hpp"
#include "deepseek_v41_kv_norm_rope_gaudi2.hpp"
#include "deepseek_v41_attention_norm_gaudi2.hpp"
#include "deepseek_v41_final_collapse_norm_gaudi2.hpp"
#include "deepseek_v41_mhc_gates_gaudi2.hpp"
#include "deepseek_v41_woa_stage_gaudi2.hpp"
#include "deepseek_v41_router_top6_gaudi2.hpp"
#include "deepseek_v41_router_logits_top6_gaudi2.hpp"
#include "deepseek_v4_sparse_attn_bf16_gaudi2.hpp"
#include "deepseek_v4_dequant_gather_bf16_gaudi2.hpp"
#include "deepseek_v4_save_partial_states_f32_gaudi2.hpp"
#include "deepseek_v4_save_compress_norm_c4_f32_gaudi2.hpp"
#include "deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2.hpp"
#include "deepseek_v4_insert_packed_kv_u8_gaudi2.hpp"
#include "deepseek_v4_paged_sparse_attn_fp8_gaudi2.hpp"
#include "deepseek_v4_flashmla_splitkv_gaudi2.hpp"
#include "deepseek_v4_mxfp4_gather_u8_gaudi2.hpp"
#include "deepseek_v4_mxfp4_dequant_fp8_gaudi2.hpp"
#include "deepseek_v4_mxfp4_indexed_moe_gaudi2.hpp"
#include "deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2.hpp"
#include "deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2.hpp"
#include "deepseek_v41_mxfp4_indexed_bf16_gaudi2.hpp"
#include "deepseek_v4_bf16_identity_gaudi2.hpp"
#include "deepseek_v4_mhc_gaudi2.hpp"
#include "deepseek_v4_sinkhorn4_gaudi2.hpp"
#include "deepseek_v4_topk_softplus_sqrt_gaudi2.hpp"
#include "deepseek_v4_fill_short_topk_i32_gaudi2.hpp"

enum KernelIndex {
    GAUDI2_KERNEL_DEEPSEEK_V41_SWA_DECODED_WRITE,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_DECODED_WRITE,
    GAUDI2_KERNEL_DEEPSEEK_V41_DECODED_ATTN,
    GAUDI2_KERNEL_DEEPSEEK_V41_DECODED_ATTN_BLOCK,
    GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PAGED_DECODED_WRITE,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PAGED_DECODED_WRITE,
    GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16_LENGTHS,
    GAUDI2_KERNEL_DEEPSEEK_V41_SPARSE_ATTN_PAIRED_EXP,
    GAUDI2_KERNEL_DEEPSEEK_V41_SPARSE_ATTN_HEAD_PAIR,
    GAUDI2_KERNEL_DEEPSEEK_V4_DEQUANT_GATHER_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_DUAL_DEQUANT_GATHER_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_LOCAL_DUAL_DEQUANT_GATHER_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_PARTIAL_STATES_F32,
    GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_F32,
    GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_F32_ORDERED,
    GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_MIXED,
    GAUDI2_KERNEL_DEEPSEEK_V4_QNORM_ROPE_KV_PACK_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_HYBRID_QNORM_ROPE_KV_PACK_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_INSERT_PACKED_KV_U8,
    GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_LOCAL_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_SEQUENTIAL_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_PAIR_SEQUENTIAL_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_QNORM_PAGED_SPARSE_ATTN_SEQUENTIAL_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SWA_ATTN_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_NATIVE_PAGED_SWA_ATTN_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_NATIVE_PAGED_SPARSE_ATTN_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_FLASHMLA_SPLITKV_PARTIAL_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_FLASHMLA_SPLITKV_PARTIAL_TILED_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_FLASHMLA_SPLITKV_COMBINE,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_GATHER_U8,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_DEQUANT_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_FC1,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_FC2,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_DEQUANT_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_DEQUANT_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_GATE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_GATE_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_UP_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_UP_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_BF16_IDENTITY,
    GAUDI2_KERNEL_DEEPSEEK_V4_MHC_POST_PREPARE,
    GAUDI2_KERNEL_DEEPSEEK_V4_MHC_PRE_EMIT,
    GAUDI2_KERNEL_DEEPSEEK_V4_MHC_PRE_EMIT_NORM,
    GAUDI2_KERNEL_DEEPSEEK_V4_SINKHORN4,
    GAUDI2_KERNEL_DEEPSEEK_V4_TOPK_SOFTPLUS_SQRT,
    GAUDI2_KERNEL_DEEPSEEK_V4_FILL_SHORT_TOPK_I32,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_K128_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_K128_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_PIPE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_BF16_IDENTITY,
    GAUDI2_KERNEL_DEEPSEEK_V41_QUANT_ROUNDTRIP_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC1_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC1_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC2_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC2_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_CACHE_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_CACHE_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VEC_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VEC_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VEC_CACHE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_C1_INDICES_I32,
    GAUDI2_KERNEL_DEEPSEEK_V41_COMPRESSOR_PAIR_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_CACHE_WRITE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK_WRITE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_CONTROL_GEMV_F32,
    GAUDI2_KERNEL_DEEPSEEK_V41_CONTROL_GEMV_RRMS_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V41_WOA_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_WOA_ROPE_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_PRODUCT_ROPE_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_WOA_SCALE,
    GAUDI2_KERNEL_DEEPSEEK_V41_WOA_SCALE_ROUNDTRIP,
    GAUDI2_KERNEL_DEEPSEEK_V41_ROUTER_TOP6,
    GAUDI2_KERNEL_DEEPSEEK_V41_ROUTER_LOGITS_TOP6,
    GAUDI2_KERNEL_DEEPSEEK_V41_WOA_STAGE,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_GATHER,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_SELECTED_PREFIX_GATHER,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_SOFTMAX,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_SHARED_KV,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_EXP_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MLA_NORMALIZE,
    GAUDI2_KERNEL_DEEPSEEK_V41_DENSE_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_DENSE_SCALE,
    GAUDI2_KERNEL_DEEPSEEK_V41_DENSE_PAIR_SCALE,
    GAUDI2_KERNEL_DEEPSEEK_V41_ATTENTION_NORM,
    GAUDI2_KERNEL_DEEPSEEK_V41_FINAL_COLLAPSE_NORM,
    GAUDI2_KERNEL_DEEPSEEK_V41_Q_SCALE_ROPE,
    GAUDI2_KERNEL_DEEPSEEK_V41_QNORM_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_KV_NORM_ROPE,
    GAUDI2_KERNEL_DEEPSEEK_V41_MHC_GATES,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_SHARED_DEQUANT_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_SHARED_DEQUANT_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_K128_DEQUANT_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_K128_DEQUANT_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SPARSE_ATTN_PACKED_EXP,
    GAUDI2_KERNEL_DEEPSEEK_V41_VECTOR_KV_SCALES,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_N512_DEQUANT_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_N512_DEQUANT_NORMAL_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_ATTN_SCORES,
    GAUDI2_KERNEL_DEEPSEEK_V41_ATTN_RECURRENCE,
    GAUDI2_KERNEL_DEEPSEEK_V41_ATTN_VALUES,
    GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PACK_G16,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PACK_G32,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_ROUNDTRIP_G32,
    GAUDI2_KERNEL_DEEPSEEK_V41_ROPE,
    GAUDI2_KERNEL_DEEPSEEK_V41_ROPE_INVERSE,
    GAUDI2_KERNEL_DEEPSEEK_V41_PREFIX_LAYOUT_R1,
    GAUDI2_KERNEL_DEEPSEEK_V41_PREFIX_LAYOUT_R2,
    GAUDI2_KERNEL_DEEPSEEK_V41_N256_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V41_N256_SLOTS_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V41_N256_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_N256_SCALE,
    GAUDI2_KERNEL_DEEPSEEK_V41_N256_SILU_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_N256_SCALE_REDUCE,
    GAUDI2_KERNEL_DEEPSEEK_V41_DYNAMIC_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_FFN_NORM_QUANT,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_MLA_GATHER,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_MLA_SOFTMAX,
    GAUDI2_KERNEL_DEEPSEEK_V41_ENGRAM_HASH_GATHER_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_SCORES,
    GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_THRESHOLD,
    GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_EMIT,
    GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_SCORES_DECODED,
    GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_KEYS,
    GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_REDUCE,
    KERNEL_COUNT
};

namespace {
void* stock_library() {
    static void* handle = dlopen("/usr/lib/habanalabs/libtpc_kernels.so", RTLD_NOW | RTLD_LOCAL);
    return handle;
}
template<typename Function> Function stock_symbol(const char* name) {
    void* handle = stock_library();
    return handle ? reinterpret_cast<Function>(dlsym(handle, name)) : nullptr;
}
bool custom_guid(const char* name) {
    constexpr char prefix[] = "custom_deepseek_v4_";
    constexpr char v41Prefix[] = "custom_deepseek_v41_";
    return std::strncmp(name, prefix, sizeof(prefix) - 1) == 0 ||
           std::strncmp(name, v41Prefix, sizeof(v41Prefix) - 1) == 0;
}
}

namespace tpc_lib_api {
// The optional manipulation payload is opaque here: stock kernels forward it
// unchanged, while custom kernels keep their explicitly qualified layout.
struct _TensorManipulationSuggestion;
}

extern "C" {
uint64_t GetLibVersion() {
    auto stock = stock_symbol<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");
    return stock ? stock() : 0;
}

tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    if (custom_guid(params->guid.name))
        return tpc_lib_api::GLUE_FAILED;
    auto stock = stock_symbol<decltype(&GetSuggestedManipulation)>("GetSuggestedManipulation");
    return stock ? stock(params, suggestion) : tpc_lib_api::GLUE_FAILED;
}

tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count) {
    if (!params || !count) return tpc_lib_api::GLUE_FAILED;
    if (custom_guid(params->guid.name)) {
        *count = 1;
        if (layouts) {
            // An explicit don't-care layout matches the optional-interface
            // default. Zero layouts is a compilation failure in Synapse 1.16.
            for (uint32_t i = 0; i < layouts->inputTensorNr; ++i)
                std::memset(layouts->inputs[i].layout, 'x', sizeof(layouts->inputs[i].layout));
            for (uint32_t i = 0; i < layouts->outputTensorNr; ++i)
                std::memset(layouts->outputs[i].layout, 'x', sizeof(layouts->outputs[i].layout));
            for (uint32_t i = 0; i < layouts->shapeTensorNr; ++i)
                std::memset(layouts->shapeTensors[i].layout, 'x', sizeof(layouts->shapeTensors[i].layout));
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    auto stock = stock_symbol<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
    return stock ? stock(params, layouts, count) : tpc_lib_api::GLUE_FAILED;
}

tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount, tpc_lib_api::GuidInfo* guids) {
    if (!kernelCount) return tpc_lib_api::GLUE_FAILED;
    auto stock = stock_symbol<decltype(&GetKernelGuids)>("GetKernelGuids");
    if (!stock) return tpc_lib_api::GLUE_FAILED;
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) return stock(deviceId, kernelCount, guids);
    uint32_t stock_count = 0;
    auto status = stock(deviceId, &stock_count, nullptr);
    if (status != tpc_lib_api::GLUE_SUCCESS) return status;
    const uint32_t capacity = *kernelCount;
    *kernelCount = KERNEL_COUNT + stock_count;
    if (!guids || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < *kernelCount) return tpc_lib_api::GLUE_FAILED;
    std::memset(guids, 0, KERNEL_COUNT * sizeof(*guids));
    for (auto mode : {DeepseekV41DecodedKVGaudi2::SWA_WRITE,
                      DeepseekV41DecodedKVGaudi2::FP4_WRITE,
                      DeepseekV41DecodedKVGaudi2::ATTENTION,
                      DeepseekV41DecodedKVGaudi2::ATTENTION_BLOCK,
                      DeepseekV41DecodedKVGaudi2::SWA_PAGED_WRITE,
                      DeepseekV41DecodedKVGaudi2::FP4_PAGED_WRITE}) {
        DeepseekV41DecodedKVGaudi2 decoded(mode);
        decoded.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SWA_DECODED_WRITE + mode].name);
    }
    DeepseekV4SparseAttnBF16Gaudi2 pairedExp(DeepseekV4SparseAttnBF16Gaudi2::PAIRED_EXP_LENGTHS);
    pairedExp.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SPARSE_ATTN_PAIRED_EXP].name);
    DeepseekV4SparseAttnBF16Gaudi2 headPair(DeepseekV4SparseAttnBF16Gaudi2::HEAD_PAIR_LENGTHS);
    headPair.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SPARSE_ATTN_HEAD_PAIR].name);
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 legacyK128(
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::LEGACY_K128);
    legacyK128.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_K128_BF16].name);
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 legacyK128Normal(
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::LEGACY_K128, true);
    legacyK128Normal.GetKernelName(
        guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_K128_NORMAL_BF16].name);
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 legacyPipeline(
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::LEGACY_PIPELINE, true);
    legacyPipeline.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_PIPE_BF16].name);
    DeepseekV41SelectedKVGaudi2 selectedOrdered(1), selectedCacheOrdered(2);
    selectedOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_ORDERED_BF16].name);
    selectedCacheOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_CACHE_ORDERED_BF16].name);
    DeepseekV41SelectedKVGaudi2 selectedValid(1, true), selectedValidCache(2, true);
    selectedValid.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_ORDERED_BF16].name);
    selectedValidCache.GetKernelName(
        guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_CACHE_ORDERED_BF16].name);
    DeepseekV41SelectedKVGaudi2 selectedVector(0, false, true), selectedVectorOrdered(1, true, true),
        selectedVectorCache(2, true, true);
    selectedVector.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VEC_BF16].name);
    selectedVectorOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VEC_ORDERED_BF16].name);
    selectedVectorCache.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VEC_CACHE_BF16].name);
    DeepseekV41Csa2PrepGaudi2 c1Indices(false);
    c1Indices.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_C1_INDICES_I32].name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_COMPRESSOR_PAIR_BF16].name,
                DeepseekV41CompressorPairGaudi2::name);
    DeepseekV41Fp4PackGaudi2 fp4Write(0);
    fp4Write.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_FP4_CACHE_WRITE_BF16].name);
    DeepseekV41SwaPackGaudi2 swaWrite(true);
    swaWrite.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK_WRITE_BF16].name);
    DeepseekV41ControlGemvGaudi2 controlGemv;
    controlGemv.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_CONTROL_GEMV_F32].name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_CONTROL_GEMV_RRMS_BF16].name,
                DeepseekV41ControlGemvRrmsGaudi2::name);
    DeepseekV41EngramHashGatherBf16Gaudi2 engramHashGather;
    engramHashGather.GetKernelName(
        guids[GAUDI2_KERNEL_DEEPSEEK_V41_ENGRAM_HASH_GATHER_BF16].name);
    DeepseekV41Mxfp4PreparedDequantFP8Gaudi2 preparedFp8;
    preparedFp8.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_FP8].name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_ATTENTION_NORM].name, DeepseekV41AttentionNormGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_FINAL_COLLAPSE_NORM].name,
                DeepseekV41FinalCollapseNormGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_Q_SCALE_ROPE].name, DeepseekV41QScaleRopeGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_QNORM_QUANT].name, DeepseekV41QNormQuantGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_KV_NORM_ROPE].name,
                DeepseekV41KVNormRopeGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MHC_GATES].name, DeepseekV41MhcGatesGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_DENSE_QUANT].name, DeepseekV41DenseGaudi2::quant_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_DENSE_SCALE].name, DeepseekV41DenseGaudi2::scale_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_DENSE_PAIR_SCALE].name,
                DeepseekV41DensePairScaleGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_WOA_QUANT].name, DeepseekV41WoaGaudi2::quant_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_WOA_ROPE_QUANT].name,
                DeepseekV41WoaGaudi2::rope_quant_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_PRODUCT_ROPE_QUANT].name,
                DeepseekV41WoaGaudi2::product_rope_quant_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_WOA_SCALE].name, DeepseekV41WoaGaudi2::scale_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_WOA_SCALE_ROUNDTRIP].name,
                DeepseekV41WoaGaudi2::scale_roundtrip_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_ROUTER_TOP6].name, DeepseekV41RouterTop6Gaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_ROUTER_LOGITS_TOP6].name,
                DeepseekV41RouterLogitsTop6Gaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_WOA_STAGE].name, DeepseekV41WoaStageGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_GATHER].name, DeepseekV41MlaGaudi2::gather_name);
    std::strcpy(
        guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_SELECTED_PREFIX_GATHER].name,
        DeepseekV41MlaGaudi2::selected_prefix_gather_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_SOFTMAX].name, DeepseekV41MlaGaudi2::softmax_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_SHARED_KV].name, DeepseekV41MlaGaudi2::shared_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_EXP_BF16].name, DeepseekV41MlaGaudi2::exp_name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MLA_NORMALIZE].name, DeepseekV41MlaNormalizeGaudi2::name);
    DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::FP8).GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_N256_FP8].name);
    DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::FP8Slots).GetKernelName(
        guids[GAUDI2_KERNEL_DEEPSEEK_V41_N256_SLOTS_FP8].name);
    DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::BF16).GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_N256_BF16].name);
    DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::Scale).GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_N256_SCALE].name);
    DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::SiluQuant).GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_N256_SILU_QUANT].name);
    DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::ScaleReduce).GetKernelName(
        guids[GAUDI2_KERNEL_DEEPSEEK_V41_N256_SCALE_REDUCE].name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_DYNAMIC_QUANT].name, DeepseekV41DynamicQuantBf16Gaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_FFN_NORM_QUANT].name,
                DeepseekV41FfnNormQuantGaudi2::name);
    for(unsigned i=0;i<4;++i)
        std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_SCORES+i].name,DeepseekV41IndexGaudi2::names[i]);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_KEYS].name,
                DeepseekV41IndexKeysGaudi2::name);
    std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_INDEX_REDUCE].name,
                DeepseekV41IndexReduceGaudi2::name);
    DeepseekV41SelectedMlaGaudi2(true).GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_MLA_GATHER].name);
    DeepseekV41SelectedMlaGaudi2(false).GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_MLA_SOFTMAX].name);

           DeepseekV4SparseAttnBF16Gaudi2 sparseAttnInstance;
           sparseAttnInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16].name);
           DeepseekV4SparseAttnBF16Gaudi2 sparseAttnLengthsInstance(
               DeepseekV4SparseAttnBF16Gaudi2::EXPLICIT_LENGTHS);
           sparseAttnLengthsInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16_LENGTHS]
                   .name);
           DeepseekV4SparseAttnBF16Gaudi2 packedExpInstance(
               DeepseekV4SparseAttnBF16Gaudi2::PACKED_EXP_LENGTHS);
           packedExpInstance.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SPARSE_ATTN_PACKED_EXP].name);
           DeepseekV4DequantGatherBF16Gaudi2 dequantGatherInstance;
           dequantGatherInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_DEQUANT_GATHER_BF16].name);
           DeepseekV4DequantGatherBF16Gaudi2 dualDequantGatherInstance(
               DeepseekV4DequantGatherBF16Gaudi2::DUAL);
           dualDequantGatherInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_DUAL_DEQUANT_GATHER_BF16]
                   .name);
           DeepseekV4DequantGatherBF16Gaudi2 localDualDequantGatherInstance(
               DeepseekV4DequantGatherBF16Gaudi2::LOCAL_DUAL);
           localDualDequantGatherInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_LOCAL_DUAL_DEQUANT_GATHER_BF16]
                   .name);
           DeepseekV4SavePartialStatesF32Gaudi2 savePartialStatesInstance;
           savePartialStatesInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_PARTIAL_STATES_F32].name);
           DeepseekV4SaveCompressNormC4F32Gaudi2 saveCompressNormInstance;
           saveCompressNormInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_F32].name);
           DeepseekV4SaveCompressNormC4F32Gaudi2
               saveCompressNormOrderedInstance(
                   DeepseekV4SaveCompressNormC4F32Gaudi2::ORDERED_F32);
           saveCompressNormOrderedInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_F32_ORDERED]
                   .name);
           DeepseekV4SaveCompressNormC4F32Gaudi2
               saveCompressNormBF16Instance(
                   DeepseekV4SaveCompressNormC4F32Gaudi2::BF16_CONSTANTS);
           saveCompressNormBF16Instance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_BF16]
                   .name);
           DeepseekV4SaveCompressNormC4F32Gaudi2
               saveCompressNormMixedInstance(
                   DeepseekV4SaveCompressNormC4F32Gaudi2::MIXED_CONSTANTS);
           saveCompressNormMixedInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_SAVE_COMPRESS_NORM_C4_MIXED]
                   .name);
           DeepseekV4QnormRopeKvPackBF16Gaudi2 qnormRopeKvPackInstance;
           qnormRopeKvPackInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_QNORM_ROPE_KV_PACK_BF16].name);
           DeepseekV4QnormRopeKvPackBF16Gaudi2 hybridQnormRopeKvPackInstance(
               DeepseekV4QnormRopeKvPackBF16Gaudi2::HYBRID_FP8_BF16_Q);
           hybridQnormRopeKvPackInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_HYBRID_QNORM_ROPE_KV_PACK_BF16]
                   .name);
           DeepseekV4InsertPackedKvU8Gaudi2 insertPackedKvInstance;
           insertPackedKvInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_INSERT_PACKED_KV_U8].name);
           DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSparseAttnInstance;
           pagedSparseAttnInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_FP8].name);
           DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSparseAttnLocalInstance(
               DeepseekV4PagedSparseAttnFP8Gaudi2::LOCAL_BLOCK_TABLE);
           pagedSparseAttnLocalInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_LOCAL_FP8]
                   .name);
           DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSparseAttnSequentialInstance(
               DeepseekV4PagedSparseAttnFP8Gaudi2::SEQUENTIAL_BLOCK_TABLE);
           pagedSparseAttnSequentialInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_SEQUENTIAL_FP8]
                   .name);
           DeepseekV4PagedSparseAttnFP8Gaudi2
               pagedSparseAttnPairSequentialInstance(
                   DeepseekV4PagedSparseAttnFP8Gaudi2::
                       PAIR_SEQUENTIAL_BLOCK_TABLE);
           pagedSparseAttnPairSequentialInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SPARSE_ATTN_PAIR_SEQUENTIAL_FP8]
                   .name);
           DeepseekV4PagedSparseAttnFP8Gaudi2
               qnormPagedSparseAttnSequentialInstance(
                   DeepseekV4PagedSparseAttnFP8Gaudi2::
                       QNORM_SEQUENTIAL_BLOCK_TABLE);
           qnormPagedSparseAttnSequentialInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_QNORM_PAGED_SPARSE_ATTN_SEQUENTIAL_FP8]
                   .name);
           DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSwaAttnInstance(
               DeepseekV4PagedSparseAttnFP8Gaudi2::SWA_ONLY);
           pagedSwaAttnInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_PAGED_SWA_ATTN_FP8]
                   .name);
           DeepseekV4FlashMLASplitKVPartialFP8Gaudi2
               flashmlaSplitKVPartialInstance;
           DeepseekV4PagedSparseAttnFP8Gaudi2 nativeSwa(
               DeepseekV4PagedSparseAttnFP8Gaudi2::SWA_ONLY, true);
           nativeSwa.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V4_NATIVE_PAGED_SWA_ATTN_FP8].name);
           DeepseekV4PagedSparseAttnFP8Gaudi2 nativeSparse(
               DeepseekV4PagedSparseAttnFP8Gaudi2::GLOBAL_SLOTS, true);
           nativeSparse.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V4_NATIVE_PAGED_SPARSE_ATTN_FP8].name);
           flashmlaSplitKVPartialInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_FLASHMLA_SPLITKV_PARTIAL_FP8]
                   .name);
           DeepseekV4FlashMLASplitKVTiledPartialFP8Gaudi2
               flashmlaSplitKVTiledPartialInstance;
           flashmlaSplitKVTiledPartialInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_FLASHMLA_SPLITKV_PARTIAL_TILED_FP8]
                   .name);
           DeepseekV4FlashMLASplitKVCombineGaudi2
               flashmlaSplitKVCombineInstance;
           flashmlaSplitKVCombineInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_FLASHMLA_SPLITKV_COMBINE]
                   .name);
           DeepseekV4Mxfp4GatherU8Gaudi2 mxfp4GatherInstance;
           mxfp4GatherInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_GATHER_U8].name);
           DeepseekV4Mxfp4DequantFp8Gaudi2 mxfp4DequantFp8Instance;
           mxfp4DequantFp8Instance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_DEQUANT_FP8]
                   .name);
           DeepseekV4Mxfp4IndexedMoeGaudi2 mxfp4IndexedFc1Instance(
               DeepseekV4Mxfp4IndexedMoeGaudi2::FC1);
           mxfp4IndexedFc1Instance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_FC1]
                   .name);
           DeepseekV4Mxfp4IndexedMoeGaudi2 mxfp4IndexedFc2Instance(
               DeepseekV4Mxfp4IndexedMoeGaudi2::FC2);
           mxfp4IndexedFc2Instance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_FC2]
                   .name);
           DeepseekV4MHCPostPrepareGaudi2 mhcPostPrepareInstance;
           DeepseekV4Mxfp4IndexedDequantBF16Gaudi2 indexedDequantInstance;
           indexedDequantInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_DEQUANT_BF16].name);
           DeepseekV4Mxfp4IndexedDequantBF16Gaudi2 indexedDequantNormalInstance(true);
           indexedDequantNormalInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_INDEXED_DEQUANT_NORMAL_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantInstance;
           preparedDequantInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantNormalInstance(true);
           preparedDequantNormalInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_NORMAL_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantGateInstance(false, 0);
           preparedDequantGateInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_GATE_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantGateNormalInstance(true, 0);
           preparedDequantGateNormalInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_GATE_NORMAL_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantUpInstance(false, 1);
           preparedDequantUpInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_UP_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantUpNormalInstance(true, 1);
           preparedDequantUpNormalInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MXFP4_PREPARED_DEQUANT_UP_NORMAL_BF16].name);
           DeepseekV4BF16IdentityGaudi2 bf16IdentityInstance;
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedV41(false, -1, true);
           preparedV41.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedV41Normal(true, -1, true);
           preparedV41Normal.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_NORMAL_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 sharedV41(false, -1, true, true);
           sharedV41.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_SHARED_DEQUANT_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 sharedV41Normal(true, -1, true, true);
           sharedV41Normal.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_SHARED_DEQUANT_NORMAL_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 tiledV41(false, -1, true, false, true);
           tiledV41.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_K128_DEQUANT_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 tiledV41Normal(true, -1, true, false, true);
           tiledV41Normal.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_K128_DEQUANT_NORMAL_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 n512V41(false, -1, true, false, true, true);
           n512V41.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_N512_DEQUANT_BF16].name);
           DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 n512V41Normal(true, -1, true, false, true, true);
           n512V41Normal.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_N512_DEQUANT_NORMAL_BF16].name);
           DeepseekV4BF16IdentityGaudi2 v41Identity(true);
           v41Identity.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_BF16_IDENTITY].name);
           DeepseekV41QuantRoundtripGaudi2 v41QuantRoundtrip;
           v41QuantRoundtrip.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_QUANT_ROUNDTRIP_BF16].name);
           DeepseekV41SelectedKVGaudi2 v41SelectedKV;
           v41SelectedKV.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_BF16].name);
           DeepseekV41Mxfp4IndexedBF16Gaudi2 v41IndexedFc1(
               DeepseekV41Mxfp4IndexedBF16Gaudi2::FC1, false);
           v41IndexedFc1.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC1_BF16].name);
           DeepseekV41Mxfp4IndexedBF16Gaudi2 v41IndexedFc1Normal(
               DeepseekV41Mxfp4IndexedBF16Gaudi2::FC1, true);
           v41IndexedFc1Normal.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC1_NORMAL_BF16].name);
           DeepseekV41Mxfp4IndexedBF16Gaudi2 v41IndexedFc2(
               DeepseekV41Mxfp4IndexedBF16Gaudi2::FC2, false);
           v41IndexedFc2.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC2_BF16].name);
           DeepseekV41Mxfp4IndexedBF16Gaudi2 v41IndexedFc2Normal(
               DeepseekV41Mxfp4IndexedBF16Gaudi2::FC2, true);
           v41IndexedFc2Normal.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_INDEXED_FC2_NORMAL_BF16].name);
           bf16IdentityInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_BF16_IDENTITY].name);
           mhcPostPrepareInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MHC_POST_PREPARE].name);
           DeepseekV4MHCPreEmitGaudi2 mhcPreEmitInstance;
           mhcPreEmitInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MHC_PRE_EMIT].name);
           DeepseekV4MHCPreEmitNormGaudi2 mhcPreEmitNormInstance;
           mhcPreEmitNormInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_MHC_PRE_EMIT_NORM].name);
           DeepseekV4TopkSoftplusSqrtGaudi2 routerTopkInstance;
           DeepseekV4Sinkhorn4Gaudi2 sinkhornInstance;
           sinkhornInstance.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V4_SINKHORN4].name);
           routerTopkInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_TOPK_SOFTPLUS_SQRT]
                   .name);
           DeepseekV4FillShortTopkI32Gaudi2 fillShortTopkInstance;
           fillShortTopkInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_FILL_SHORT_TOPK_I32]
                   .name);
    DeepseekV41SelectedKVGaudi2 vectorScales(0, false, false, true);
    vectorScales.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_VECTOR_KV_SCALES].name);
    for (int phase = 0; phase < 3; ++phase) {
        DeepseekV41HeadAttentionGaudi2 part(phase);
        part.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_ATTN_SCORES + phase].name);
    }
    constexpr unsigned codec_modes[] = {0, 16, 32, DeepseekV41KVPackGaudi2::FP4_ROUNDTRIP_G32};
    for (int mode = 0; mode < 4; ++mode) {
        DeepseekV41KVPackGaudi2 part(codec_modes[mode]);
        part.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK + mode].name);
    }
    for (int inverse = 0; inverse < 2; ++inverse) {
        DeepseekV41RopeGaudi2 part(inverse);
        part.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_ROPE + inverse].name);
    }
    for (int ratio = 1; ratio <= 2; ++ratio) {
        DeepseekV41PrefixLayoutGaudi2 part(ratio);
        part.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_PREFIX_LAYOUT_R1 + ratio - 1].name);
    }
    return stock(deviceId, &stock_count, guids + KERNEL_COUNT);
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    char kernelName[tpc_lib_api::MAX_NODE_NAME];
    for (auto mode : {DeepseekV41DecodedKVGaudi2::SWA_WRITE,
                      DeepseekV41DecodedKVGaudi2::FP4_WRITE,
                      DeepseekV41DecodedKVGaudi2::ATTENTION,
                      DeepseekV41DecodedKVGaudi2::ATTENTION_BLOCK,
                      DeepseekV41DecodedKVGaudi2::SWA_PAGED_WRITE,
                      DeepseekV41DecodedKVGaudi2::FP4_PAGED_WRITE}) {
        DeepseekV41DecodedKVGaudi2 decoded(mode);
        decoded.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0)
            return decoded.GetGcDefinitions(params, instance);
    }
    for (auto mode : {DeepseekV4SparseAttnBF16Gaudi2::PAIRED_EXP_LENGTHS,
                      DeepseekV4SparseAttnBF16Gaudi2::HEAD_PAIR_LENGTHS}) {
        DeepseekV4SparseAttnBF16Gaudi2 attention(mode);
        attention.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0)
            return attention.GetGcDefinitions(params, instance);
    }
    for (auto mode : {DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::LEGACY_K128,
                      DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::LEGACY_PIPELINE}) {
        for (bool normal : {false, true}) {
            if (mode == DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::LEGACY_PIPELINE && !normal) continue;
            DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 decoder(mode, normal);
            decoder.GetKernelName(kernelName);
            if (std::strcmp(params->guid.name, kernelName) == 0)
                return decoder.GetGcDefinitions(params, instance);
        }
    }
    for (unsigned order : {0u, 1u, 2u}) {
        for (bool vector : {false, true}) {
            DeepseekV41SelectedKVGaudi2 selected(order, order != 0 && vector, vector);
            selected.GetKernelName(kernelName);
            if (std::strcmp(params->guid.name, kernelName) == 0)
                return selected.GetGcDefinitions(params, instance);
        }
        if (order != 0) {
            DeepseekV41SelectedKVGaudi2 selected(order, true);
            selected.GetKernelName(kernelName);
            if (std::strcmp(params->guid.name, kernelName) == 0)
                return selected.GetGcDefinitions(params, instance);
        }
    }
    DeepseekV41Csa2PrepGaudi2 c1Indices(false);
    c1Indices.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return c1Indices.GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name,
                    DeepseekV41CompressorPairGaudi2::name) == 0)
        return DeepseekV41CompressorPairGaudi2().GetGcDefinitions(
            params, instance);
    DeepseekV41Fp4PackGaudi2 fp4Write(0);
    fp4Write.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return fp4Write.GetGcDefinitions(params, instance);
    DeepseekV41SwaPackGaudi2 swaWrite(true);
    swaWrite.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return swaWrite.GetGcDefinitions(params, instance);
    DeepseekV41ControlGemvGaudi2 controlGemv;
    controlGemv.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return controlGemv.GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name,
                    DeepseekV41ControlGemvRrmsGaudi2::name) == 0)
        return DeepseekV41ControlGemvRrmsGaudi2().GetGcDefinitions(
            params, instance);
    DeepseekV41EngramHashGatherBf16Gaudi2 engramHashGather;
    engramHashGather.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return engramHashGather.GetGcDefinitions(params, instance);
    DeepseekV41Mxfp4PreparedDequantFP8Gaudi2 preparedFp8;
    preparedFp8.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return preparedFp8.GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41AttentionNormGaudi2::name) == 0)
        return DeepseekV41AttentionNormGaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name,
                    DeepseekV41FinalCollapseNormGaudi2::name) == 0)
        return DeepseekV41FinalCollapseNormGaudi2().GetGcDefinitions(
            params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41QScaleRopeGaudi2::name) == 0)
        return DeepseekV41QScaleRopeGaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41QNormQuantGaudi2::name) == 0)
        return DeepseekV41QNormQuantGaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41KVNormRopeGaudi2::name) == 0)
        return DeepseekV41KVNormRopeGaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41DenseGaudi2::quant_name) == 0)
        return DeepseekV41DenseGaudi2(true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41DenseGaudi2::scale_name) == 0)
        return DeepseekV41DenseGaudi2(false).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name,
                    DeepseekV41DensePairScaleGaudi2::name) == 0)
        return DeepseekV41DensePairScaleGaudi2().GetGcDefinitions(params,
                                                                  instance);
    if (std::strcmp(params->guid.name, DeepseekV41WoaGaudi2::quant_name) == 0)
        return DeepseekV41WoaGaudi2(true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41WoaGaudi2::rope_quant_name) == 0)
        return DeepseekV41WoaGaudi2(true, true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41WoaGaudi2::product_rope_quant_name) == 0)
        return DeepseekV41WoaGaudi2(true, true, false, true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41WoaGaudi2::scale_name) == 0)
        return DeepseekV41WoaGaudi2(false).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41WoaGaudi2::scale_roundtrip_name) == 0)
        return DeepseekV41WoaGaudi2(false, false, true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41RouterTop6Gaudi2::name) == 0)
        return DeepseekV41RouterTop6Gaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41RouterLogitsTop6Gaudi2::name) == 0)
        return DeepseekV41RouterLogitsTop6Gaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41WoaStageGaudi2::name) == 0)
        return DeepseekV41WoaStageGaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41MlaGaudi2::gather_name) == 0)
        return DeepseekV41MlaGaudi2(true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name,
                    DeepseekV41MlaGaudi2::selected_prefix_gather_name) == 0)
        return DeepseekV41MlaGaudi2(true, false, true)
            .GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41MlaGaudi2::softmax_name) == 0)
        return DeepseekV41MlaGaudi2(false).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41MlaGaudi2::shared_name) == 0)
        return DeepseekV41MlaGaudi2(true, true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41MlaGaudi2::exp_name) == 0)
        return DeepseekV41MlaGaudi2(false, true).GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41MlaNormalizeGaudi2::name) == 0)
        return DeepseekV41MlaNormalizeGaudi2().GetGcDefinitions(params, instance);
    for (int ratio = 1; ratio <= 2; ++ratio) {
        DeepseekV41PrefixLayoutGaudi2 part(ratio);
        part.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0) return part.GetGcDefinitions(params, instance);
    }
    for (int inverse = 0; inverse < 2; ++inverse) {
        DeepseekV41RopeGaudi2 part(inverse);
        part.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0) return part.GetGcDefinitions(params, instance);
    }
    for (unsigned mode : {0u, 16u, 32u, DeepseekV41KVPackGaudi2::FP4_ROUNDTRIP_G32}) {
        DeepseekV41KVPackGaudi2 part(mode);
        part.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0) return part.GetGcDefinitions(params, instance);
    }
    for (int phase = 0; phase < 3; ++phase) {
        DeepseekV41HeadAttentionGaudi2 part(phase);
        part.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0) return part.GetGcDefinitions(params, instance);
    }
    DeepseekV41SelectedKVGaudi2 vectorScales(0, false, false, true);
    vectorScales.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return vectorScales.GetGcDefinitions(params, instance);
    auto mainFast0 = DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::FP8, true);
    mainFast0.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast0.GetGcDefinitions(params, instance);
    auto mainFastSlots = DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::FP8Slots);
    mainFastSlots.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFastSlots.GetGcDefinitions(params, instance);
    auto mainFast1 = DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::BF16);
    mainFast1.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast1.GetGcDefinitions(params, instance);
    auto mainFast2 = DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::Scale);
    mainFast2.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast2.GetGcDefinitions(params, instance);
    auto mainFast3 = DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::SiluQuant);
    mainFast3.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast3.GetGcDefinitions(params, instance);
    auto mainFast4 = DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::ScaleReduce);
    mainFast4.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast4.GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41MhcGatesGaudi2::name) == 0)
        return DeepseekV41MhcGatesGaudi2().GetGcDefinitions(params, instance);
    auto mainFastDynamicQuant = DeepseekV41DynamicQuantBf16Gaudi2();
    if (std::strcmp(params->guid.name, DeepseekV41DynamicQuantBf16Gaudi2::name) == 0)
        return mainFastDynamicQuant.GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41FfnNormQuantGaudi2::name) == 0)
        return DeepseekV41FfnNormQuantGaudi2().GetGcDefinitions(params, instance);
    auto mainFast5 = DeepseekV41SelectedMlaGaudi2(true);
    mainFast5.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast5.GetGcDefinitions(params, instance);
    auto mainFast6 = DeepseekV41SelectedMlaGaudi2(false);
    mainFast6.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return mainFast6.GetGcDefinitions(params, instance);
    DeepseekV41SelectedKVGaudi2 selectedKV;
    selectedKV.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return selectedKV.GetGcDefinitions(params, instance);
    DeepseekV4SparseAttnBF16Gaudi2 packedExpInstance(
        DeepseekV4SparseAttnBF16Gaudi2::PACKED_EXP_LENGTHS);
    packedExpInstance.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return packedExpInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4SparseAttnBF16Gaudi2 sparseAttnInstance;
    sparseAttnInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return sparseAttnInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4SparseAttnBF16Gaudi2 sparseAttnLengthsInstance(
        DeepseekV4SparseAttnBF16Gaudi2::EXPLICIT_LENGTHS);
    sparseAttnLengthsInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return sparseAttnLengthsInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4DequantGatherBF16Gaudi2 dequantGatherInstance;
    dequantGatherInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return dequantGatherInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4DequantGatherBF16Gaudi2 dualDequantGatherInstance(
        DeepseekV4DequantGatherBF16Gaudi2::DUAL);
    dualDequantGatherInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return dualDequantGatherInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4DequantGatherBF16Gaudi2 localDualDequantGatherInstance(
        DeepseekV4DequantGatherBF16Gaudi2::LOCAL_DUAL);
    localDualDequantGatherInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return localDualDequantGatherInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4SavePartialStatesF32Gaudi2 savePartialStatesInstance;
    savePartialStatesInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return savePartialStatesInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4SaveCompressNormC4F32Gaudi2 saveCompressNormInstance;
    saveCompressNormInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return saveCompressNormInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4SaveCompressNormC4F32Gaudi2 saveCompressNormOrderedInstance(
        DeepseekV4SaveCompressNormC4F32Gaudi2::ORDERED_F32);
    saveCompressNormOrderedInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return saveCompressNormOrderedInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4SaveCompressNormC4F32Gaudi2 saveCompressNormBF16Instance(
        DeepseekV4SaveCompressNormC4F32Gaudi2::BF16_CONSTANTS);
    saveCompressNormBF16Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return saveCompressNormBF16Instance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4SaveCompressNormC4F32Gaudi2 saveCompressNormMixedInstance(
        DeepseekV4SaveCompressNormC4F32Gaudi2::MIXED_CONSTANTS);
    saveCompressNormMixedInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return saveCompressNormMixedInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4QnormRopeKvPackBF16Gaudi2 qnormRopeKvPackInstance;
    qnormRopeKvPackInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return qnormRopeKvPackInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4QnormRopeKvPackBF16Gaudi2 hybridQnormRopeKvPackInstance(
        DeepseekV4QnormRopeKvPackBF16Gaudi2::HYBRID_FP8_BF16_Q);
    hybridQnormRopeKvPackInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return hybridQnormRopeKvPackInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4InsertPackedKvU8Gaudi2 insertPackedKvInstance;
    insertPackedKvInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return insertPackedKvInstance.GetGcDefinitions(params, instance);
    }

    for (auto mode : {DeepseekV4PagedSparseAttnFP8Gaudi2::SWA_ONLY,
                      DeepseekV4PagedSparseAttnFP8Gaudi2::GLOBAL_SLOTS}) {
        DeepseekV4PagedSparseAttnFP8Gaudi2 functionalAttention(mode, true);
        functionalAttention.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0) {
            return functionalAttention.GetGcDefinitions(params, instance);
        }
    }
    DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSparseAttnInstance;
    pagedSparseAttnInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return pagedSparseAttnInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSparseAttnLocalInstance(
        DeepseekV4PagedSparseAttnFP8Gaudi2::LOCAL_BLOCK_TABLE);
    pagedSparseAttnLocalInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return pagedSparseAttnLocalInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSparseAttnSequentialInstance(
        DeepseekV4PagedSparseAttnFP8Gaudi2::SEQUENTIAL_BLOCK_TABLE);
    pagedSparseAttnSequentialInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return pagedSparseAttnSequentialInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4PagedSparseAttnFP8Gaudi2
        pagedSparseAttnPairSequentialInstance(
            DeepseekV4PagedSparseAttnFP8Gaudi2::
                PAIR_SEQUENTIAL_BLOCK_TABLE);
    pagedSparseAttnPairSequentialInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return pagedSparseAttnPairSequentialInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4PagedSparseAttnFP8Gaudi2
        qnormPagedSparseAttnSequentialInstance(
            DeepseekV4PagedSparseAttnFP8Gaudi2::
                QNORM_SEQUENTIAL_BLOCK_TABLE);
    qnormPagedSparseAttnSequentialInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return qnormPagedSparseAttnSequentialInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4PagedSparseAttnFP8Gaudi2 pagedSwaAttnInstance(
        DeepseekV4PagedSparseAttnFP8Gaudi2::SWA_ONLY);
    pagedSwaAttnInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return pagedSwaAttnInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4FlashMLASplitKVPartialFP8Gaudi2
        flashmlaSplitKVPartialInstance;
    flashmlaSplitKVPartialInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return flashmlaSplitKVPartialInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4FlashMLASplitKVTiledPartialFP8Gaudi2
        flashmlaSplitKVTiledPartialInstance;
    flashmlaSplitKVTiledPartialInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return flashmlaSplitKVTiledPartialInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4FlashMLASplitKVCombineGaudi2 flashmlaSplitKVCombineInstance;
    flashmlaSplitKVCombineInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return flashmlaSplitKVCombineInstance.GetGcDefinitions(
            params, instance);
    }

    DeepseekV4Mxfp4GatherU8Gaudi2 mxfp4GatherInstance;
    mxfp4GatherInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mxfp4GatherInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4Mxfp4DequantFp8Gaudi2 mxfp4DequantFp8Instance;
    mxfp4DequantFp8Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mxfp4DequantFp8Instance.GetGcDefinitions(params, instance);
    }

    DeepseekV4Mxfp4IndexedMoeGaudi2 mxfp4IndexedFc1Instance(
        DeepseekV4Mxfp4IndexedMoeGaudi2::FC1);
    mxfp4IndexedFc1Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mxfp4IndexedFc1Instance.GetGcDefinitions(params, instance);
    }

    DeepseekV4Mxfp4IndexedMoeGaudi2 mxfp4IndexedFc2Instance(
        DeepseekV4Mxfp4IndexedMoeGaudi2::FC2);
    mxfp4IndexedFc2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mxfp4IndexedFc2Instance.GetGcDefinitions(params, instance);
    }

    DeepseekV4Mxfp4IndexedDequantBF16Gaudi2 indexedDequantInstance;
    indexedDequantInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return indexedDequantInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4Mxfp4IndexedDequantBF16Gaudi2 indexedDequantNormalInstance(true);
    indexedDequantNormalInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return indexedDequantNormalInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantInstance;
    preparedDequantInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return preparedDequantInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantNormalInstance(true);
    preparedDequantNormalInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return preparedDequantNormalInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantGateInstance(false, 0);
    preparedDequantGateInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return preparedDequantGateInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantGateNormalInstance(true, 0);
    preparedDequantGateNormalInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return preparedDequantGateNormalInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantUpInstance(false, 1);
    preparedDequantUpInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return preparedDequantUpInstance.GetGcDefinitions(params, instance);
    }
    DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedDequantUpNormalInstance(true, 1);
    preparedDequantUpNormalInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return preparedDequantUpNormalInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4BF16IdentityGaudi2 bf16IdentityInstance;
    DeepseekV4BF16IdentityGaudi2 v41Identity(true);
    DeepseekV41QuantRoundtripGaudi2 v41QuantRoundtrip;
    v41QuantRoundtrip.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0)
        return v41QuantRoundtrip.GetGcDefinitions(params, instance);
    for (auto stage : {DeepseekV41Mxfp4IndexedBF16Gaudi2::FC1,
                       DeepseekV41Mxfp4IndexedBF16Gaudi2::FC2}) {
        for (bool normal : {false, true}) {
            DeepseekV41Mxfp4IndexedBF16Gaudi2 indexed(stage, normal);
            indexed.GetKernelName(kernelName);
            if (std::strcmp(params->guid.name, kernelName) == 0)
                return indexed.GetGcDefinitions(params, instance);
        }
    }
    v41Identity.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
        return v41Identity.GetGcDefinitions(params, instance);
    for (bool normal : {false, true}) {
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 sharedV41(normal, -1, true, true);
        sharedV41.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0)
            return sharedV41.GetGcDefinitions(params, instance);
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 tiledV41(normal, -1, true, false, true);
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 n512V41(normal, -1, true, false, true, true);
        n512V41.GetKernelName(kernelName);
        if (std::strcmp(params->guid.name, kernelName) == 0)
            return n512V41.GetGcDefinitions(params, instance);
        tiledV41.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0)
            return tiledV41.GetGcDefinitions(params, instance);
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedV41(normal, -1, true);
        preparedV41.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0)
            return preparedV41.GetGcDefinitions(params, instance);
    }
    bf16IdentityInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return bf16IdentityInstance.GetGcDefinitions(params, instance);
    }

    for(unsigned i=0;i<4;++i)
        if(std::strcmp(params->guid.name,DeepseekV41IndexGaudi2::names[i])==0)
            return DeepseekV41IndexGaudi2(i).GetGcDefinitions(params,instance);
    if (std::strcmp(params->guid.name, DeepseekV41IndexKeysGaudi2::name) == 0)
        return DeepseekV41IndexKeysGaudi2().GetGcDefinitions(params, instance);
    if (std::strcmp(params->guid.name, DeepseekV41IndexReduceGaudi2::name) == 0)
        return DeepseekV41IndexReduceGaudi2().GetGcDefinitions(params, instance);

    DeepseekV4MHCPostPrepareGaudi2 mhcPostPrepareInstance;
    DeepseekV4Sinkhorn4Gaudi2 sinkhornInstance;
    sinkhornInstance.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return sinkhornInstance.GetGcDefinitions(params, instance);
    }
    mhcPostPrepareInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mhcPostPrepareInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4MHCPreEmitGaudi2 mhcPreEmitInstance;
    mhcPreEmitInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mhcPreEmitInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4MHCPreEmitNormGaudi2 mhcPreEmitNormInstance;
    mhcPreEmitNormInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return mhcPreEmitNormInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4TopkSoftplusSqrtGaudi2 routerTopkInstance;
    routerTopkInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return routerTopkInstance.GetGcDefinitions(params, instance);
    }

    DeepseekV4FillShortTopkI32Gaudi2 fillShortTopkInstance;
    fillShortTopkInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return fillShortTopkInstance.GetGcDefinitions(params, instance);
    }

    auto stock = stock_symbol<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
    return stock ? stock(params, instance) : tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId,
    const tpc_lib_api::ShapeInferenceParams* params, tpc_lib_api::ShapeInferenceOutput* output) {
    if (!params || !output) return tpc_lib_api::GLUE_FAILED;
    const auto* guid = params->pGuid ? params->pGuid : &params->guid;
    if (std::strncmp(guid->name, "custom_deepseek_v4_", 19) == 0) return tpc_lib_api::GLUE_SUCCESS;
    auto stock = stock_symbol<decltype(&GetShapeInference)>("GetShapeInference");
    return stock ? stock(deviceId, params, output) : tpc_lib_api::GLUE_SUCCESS;
}
}
