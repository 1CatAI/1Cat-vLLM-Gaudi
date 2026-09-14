// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_selected_kv_gaudi2.hpp"
#include <cstring>
#include <algorithm>
extern unsigned char _binary___deepseek_v41_selected_kv_vec_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_vec_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_vec_cache_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_vec_cache_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_ordered_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_ordered_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41SelectedKVGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    if (vector_scales_) {
        std::strcpy(name, "custom_deepseek_v41_selected_kv_vector_scales_bf16_gaudi2");
        return tpc_lib_api::GLUE_SUCCESS;
    }
    if (vector_) {
        static_assert(sizeof("custom_deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2") <= tpc_lib_api::MAX_NODE_NAME);
        std::strcpy(name, ordered_ == 2 ? "custom_deepseek_v41_selected_kv_vec_cache_bf16_gaudi2" :
                          ordered_ ? "custom_deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2" :
                                     "custom_deepseek_v41_selected_kv_vec_bf16_gaudi2");
        return tpc_lib_api::GLUE_SUCCESS;
    }
    if (valid_only_) {
        std::strcpy(name, ordered_ == 2 ? "custom_deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2" :
                                         "custom_deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2");
        return tpc_lib_api::GLUE_SUCCESS;
    }
    std::strcpy(name, ordered_ == 2 ? "custom_deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2" : ordered_ ? "custom_deepseek_v41_selected_kv_ordered_bf16_gaudi2"
                               : "custom_deepseek_v41_selected_kv_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41SelectedKVGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (vector_scales_) {
        if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const auto slots = in->inputTensors[2].geometry.maxSizes[0];
        const TensorDataType types[] = {DATA_U8, DATA_U8, DATA_I32, DATA_BF16, DATA_I32};
        for (unsigned i = 0; i < 5; ++i) {
            auto& g = i < 3 ? in->inputTensors[i].geometry : in->outputTensors[i - 3].geometry;
            if (g.dataType != types[i]) { g.dataType = types[i]; return GLUE_INCOMPATIBLE_DATA_TYPE; }
            bool size = g.dims == 2;
            if (i == 0) size &= g.maxSizes[0] == 528 && g.maxSizes[1] > 0 && g.maxSizes[1] <= 512;
            if (i == 1) size &= g.maxSizes[0] == 288 && g.maxSizes[1] > 0 &&
                               g.maxSizes[1] <= 0x7fffffffULL - 512;
            if (i == 2 || i == 4) size &= slots > 0 && slots <= 4096 &&
                                         g.maxSizes[0] == slots && g.maxSizes[1] == 1;
            if (i == 3) size &= g.maxSizes[0] == 512 && g.maxSizes[1] == slots;
            if (!size) return i < 3 ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            auto& access = i < 3 ? out->inputTensorAccessPattern[i] : out->outputTensorAccessPattern[i - 3];
            if (i < 2) {
                std::memset(&access, 0, sizeof(access));
                access.sparseAccess = true;
                access.mapping[0].indexSpaceDim = 0;
                access.mapping[0].a = 0;
                access.mapping[0].start_b = 0;
                access.mapping[0].end_b = static_cast<float>(g.maxSizes[0] - 1);
                access.mapping[1].indexSpaceDim = 0;
                access.mapping[1].a = 0;
                access.mapping[1].start_b = 0;
                access.mapping[1].end_b = static_cast<float>(g.maxSizes[1] - 1);
            } else {
                for (unsigned d = 0; d < 2; ++d) {
                    auto& map = access.mapping[d];
                    map.indexSpaceDim = 0;
                    map.a = (i == 3 ? d == 1 : d == 0) ? 1 : 0;
                    map.start_b = 0;
                    map.end_b = i == 3 && d == 0 ? 511 : 0;
                }
            }
        }
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = slots;
        out->kernel.paramsNr = 0;
        auto* start = &_binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_start;
        auto* end = &_binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_end;
        const unsigned capacity = out->kernel.elfSize;
        out->kernel.elfSize = end - start;
        if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
        std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
        return GLUE_SUCCESS;
    }
    const unsigned inputs = 3 + ordered_;
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto slots = in->inputTensors[2].geometry.maxSizes[0];
    if (vector_ && ordered_ && !valid_only_) return GLUE_FAILED;
    const TensorDataType types[] = {DATA_U8, DATA_U8, DATA_I32, DATA_BF16, DATA_I32};
    for (unsigned i = 0; i < 5; ++i) {
        auto& g = i < 3 ? in->inputTensors[i].geometry : in->outputTensors[i-3].geometry;
        if (g.dataType != types[i]) { g.dataType = types[i]; return GLUE_INCOMPATIBLE_DATA_TYPE; }
        bool size = g.dims == 2;
        if (i == 0) size &= g.maxSizes[0] == 528 && g.maxSizes[1] > 0 && g.maxSizes[1] <= 512;
        if (i == 1) size &= (g.maxSizes[0] == 288 || g.maxSizes[0] == 528) && g.maxSizes[1] > 0 && g.maxSizes[1] <= 512;
        if (i == 2 || i == 4) size &= slots > 0 && slots <= 1024 && g.maxSizes[0] == slots && g.maxSizes[1] == 1;
        if (i == 3) size &= g.maxSizes[0] == 512 && g.maxSizes[1] == slots;
        if (!size) return i < 3 ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto& access = i < 3 ? out->inputTensorAccessPattern[i] : out->outputTensorAccessPattern[i-3];
        if (i < 2) {
            access.allRequired = true;
        } else {
            for (unsigned d = 0; d < 2; ++d) {
                auto& map = access.mapping[d];
                map.indexSpaceDim = 0;
                map.a = (i == 3 ? d == 1 : d == 0) ? 1 : 0;
                map.start_b = 0;
                map.end_b = i == 3 && d == 0 ? 511 : 0;
                if (vector_ && map.a == 1) map.end_b = ((slots - 1) / 128) * 128;
            }
        }
    }
    if (ordered_) {
        const auto& completion = in->inputTensors[3].geometry;
        if (completion.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (completion.dims != 1 || completion.maxSizes[0] != 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->inputTensorAccessPattern[3].allRequired = true;
    }
    if (ordered_ == 2) {
        const auto& completion = in->inputTensors[4].geometry;
        if (completion.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (completion.dims != 1 || completion.maxSizes[0] != 36) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->inputTensorAccessPattern[4].allRequired = true;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = vector_ ? std::min<uint64_t>(slots, 128) : slots;
    out->kernel.paramsNr = 0;
    auto* start = ordered_ == 2 ? &_binary___deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2_o_start : ordered_ ? &_binary___deepseek_v41_selected_kv_ordered_bf16_gaudi2_o_start
                           : &_binary___deepseek_v41_selected_kv_bf16_gaudi2_o_start;
    auto* end = ordered_ == 2 ? &_binary___deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2_o_end : ordered_ ? &_binary___deepseek_v41_selected_kv_ordered_bf16_gaudi2_o_end
                         : &_binary___deepseek_v41_selected_kv_bf16_gaudi2_o_end;
    if (valid_only_) {
        start = ordered_ == 2 ? &_binary___deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2_o_start :
                                &_binary___deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2_o_start;
        end = ordered_ == 2 ? &_binary___deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2_o_end :
                              &_binary___deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2_o_end;
    }
    if (vector_) {
        start = ordered_ == 2 ? &_binary___deepseek_v41_selected_kv_vec_cache_bf16_gaudi2_o_start :
                ordered_ ? &_binary___deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2_o_start :
                           &_binary___deepseek_v41_selected_kv_vec_bf16_gaudi2_o_start;
        end = ordered_ == 2 ? &_binary___deepseek_v41_selected_kv_vec_cache_bf16_gaudi2_o_end :
              ordered_ ? &_binary___deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2_o_end :
                         &_binary___deepseek_v41_selected_kv_vec_bf16_gaudi2_o_end;
    }
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
