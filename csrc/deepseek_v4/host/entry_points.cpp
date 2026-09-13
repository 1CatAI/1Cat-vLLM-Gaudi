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
#include "deepseek_v4_bf16_identity_gaudi2.hpp"
#include "deepseek_v4_mhc_gaudi2.hpp"
#include "deepseek_v4_sinkhorn4_gaudi2.hpp"
#include "deepseek_v41_quant_roundtrip_gaudi2.hpp"
#include "deepseek_v41_swa_pack_gaudi2.hpp"
#include "deepseek_v41_fp4_pack_gaudi2.hpp"
#include "deepseek_v41_csa2_prep_gaudi2.hpp"
#include "deepseek_v41_selected_kv_gaudi2.hpp"
#include "deepseek_v41_control_gemv_gaudi2.hpp"
#include "deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2.hpp"
#include "deepseek_v41_dynamic_quant_bf16_gaudi2.hpp"
#include "deepseek_v4_topk_softplus_sqrt_gaudi2.hpp"
#include "deepseek_v4_fill_short_topk_i32_gaudi2.hpp"

enum KernelIndex {
    GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16_LENGTHS,
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
    GAUDI2_KERNEL_DEEPSEEK_V41_BF16_IDENTITY,
    GAUDI2_KERNEL_DEEPSEEK_V41_QUANT_ROUNDTRIP_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_ROPE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_C1_INDICES_I32,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PACK_G16_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PACK_G32_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_FP4_CACHE_WRITE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_CACHE_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_CACHE_ORDERED_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK_WRITE_BF16,
    GAUDI2_KERNEL_DEEPSEEK_V41_CONTROL_GEMV_F32,
    GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_FP8,
    GAUDI2_KERNEL_DEEPSEEK_V41_DYNAMIC_QUANT_BF16,
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
           DeepseekV41Mxfp4PreparedDequantFP8Gaudi2 v41fp8;
           v41fp8.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_MXFP4_PREPARED_DEQUANT_FP8].name);
           DeepseekV41DynamicQuantBf16Gaudi2 v41quant;
           std::strcpy(guids[GAUDI2_KERNEL_DEEPSEEK_V41_DYNAMIC_QUANT_BF16].name, DeepseekV41DynamicQuantBf16Gaudi2::name);
           DeepseekV4SparseAttnBF16Gaudi2 sparseAttnInstance;
           sparseAttnInstance.GetKernelName(
               guids[GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16].name);
           DeepseekV4SparseAttnBF16Gaudi2 sparseAttnLengthsInstance(
               DeepseekV4SparseAttnBF16Gaudi2::EXPLICIT_LENGTHS);
           sparseAttnLengthsInstance.GetKernelName(
               guids[
                   GAUDI2_KERNEL_DEEPSEEK_V4_SPARSE_ATTN_BF16_LENGTHS]
                   .name);
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
           DeepseekV4BF16IdentityGaudi2 v41Identity(true);
           DeepseekV41QuantRoundtripGaudi2 v41Quant;
           DeepseekV41ControlGemvGaudi2 v41Control;
           v41Control.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_CONTROL_GEMV_F32].name);
           DeepseekV41Csa2PrepGaudi2 rope(true), indices(false);
           rope.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_ROPE_BF16].name);
           indices.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_C1_INDICES_I32].name);
           DeepseekV41Fp4PackGaudi2 fp4g16(16), fp4g32(32), fp4write(0);
           fp4g16.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PACK_G16_BF16].name);
           fp4g32.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_FP4_PACK_G32_BF16].name);
           fp4write.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_FP4_CACHE_WRITE_BF16].name);
           DeepseekV41SelectedKVGaudi2 validOrdered(1, true), validCacheOrdered(2, true);
           validOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_ORDERED_BF16].name);
           validCacheOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_VALID_CACHE_ORDERED_BF16].name);
           DeepseekV41SelectedKVGaudi2 cacheOrdered(2);
           cacheOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_CACHE_ORDERED_BF16].name);
           DeepseekV41SwaPackGaudi2 v41SwaPack, v41SwaWrite(true);
           v41SwaPack.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK_BF16].name);
           v41SwaWrite.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SWA_PACK_WRITE_BF16].name);
           DeepseekV41SelectedKVGaudi2 v41SelectedOrdered(true);
           v41SelectedOrdered.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_ORDERED_BF16].name);
           DeepseekV41SelectedKVGaudi2 v41Selected;
           v41Selected.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_SELECTED_KV_BF16].name);
           v41Quant.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_QUANT_ROUNDTRIP_BF16].name);
           v41Identity.GetKernelName(guids[GAUDI2_KERNEL_DEEPSEEK_V41_BF16_IDENTITY].name);
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
    return stock(deviceId, &stock_count, guids + KERNEL_COUNT);
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    char kernelName[tpc_lib_api::MAX_NODE_NAME];
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
    DeepseekV41QuantRoundtripGaudi2 v41Quant;
    DeepseekV41ControlGemvGaudi2 v41Control;
    v41Control.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
        return v41Control.GetGcDefinitions(params, instance);
    for (bool rope : {false, true}) {
        DeepseekV41Csa2PrepGaudi2 prep(rope);
        prep.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0) return prep.GetGcDefinitions(params, instance);
    }
    for (unsigned mode : {0u, 16u, 32u}) {
        DeepseekV41Fp4PackGaudi2 fp4(mode);
        fp4.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0) return fp4.GetGcDefinitions(params, instance);
    }
    for (unsigned order : {1u, 2u}) {
        DeepseekV41SelectedKVGaudi2 valid(order, true);
        valid.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0) return valid.GetGcDefinitions(params, instance);
    }
    DeepseekV41SelectedKVGaudi2 cacheOrdered(2);
    cacheOrdered.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) return cacheOrdered.GetGcDefinitions(params, instance);
    for (bool ordered : {false, true}) {
        DeepseekV41SwaPackGaudi2 swaPack(ordered);
        swaPack.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0) return swaPack.GetGcDefinitions(params, instance);
        DeepseekV41SelectedKVGaudi2 selected(ordered);
        selected.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0) return selected.GetGcDefinitions(params, instance);
    }
    DeepseekV41SelectedKVGaudi2 v41Selected;
    v41Selected.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
        return v41Selected.GetGcDefinitions(params, instance);
    v41Quant.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
        return v41Quant.GetGcDefinitions(params, instance);
    v41Identity.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
        return v41Identity.GetGcDefinitions(params, instance);
    for (bool normal : {false, true}) {
        DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 preparedV41(normal, -1, true);
        preparedV41.GetKernelName(kernelName);
        if (strcmp(params->guid.name, kernelName) == 0)
            return preparedV41.GetGcDefinitions(params, instance);
    }
    bf16IdentityInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) {
        return bf16IdentityInstance.GetGcDefinitions(params, instance);
    }

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

    DeepseekV41Mxfp4PreparedDequantFP8Gaudi2 v41fp8;
    v41fp8.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0) return v41fp8.GetGcDefinitions(params, instance);
    DeepseekV41DynamicQuantBf16Gaudi2 v41quant;
    if (strcmp(params->guid.name, DeepseekV41DynamicQuantBf16Gaudi2::name) == 0)
        return v41quant.GetGcDefinitions(params, instance);
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
