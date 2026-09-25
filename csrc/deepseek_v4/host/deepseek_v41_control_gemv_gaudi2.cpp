// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_control_gemv_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_control_gemv_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_control_gemv_f32_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_control_batch4_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_control_batch4_f32_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_control_prefetch_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_control_prefetch_f32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41ControlGemvGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, prefetch_ ? "custom_deepseek_v41_control_prefetch_f32_gaudi2" : batch4_ ? "custom_deepseek_v41_control_batch4_f32_gaudi2"
                             : "custom_deepseek_v41_control_gemv_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41ControlGemvGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 2) {
        in->inputTensorNr = 2;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto& a = in->inputTensors[0].geometry;
    const auto& w = in->inputTensors[1].geometry;
    const auto& y = in->outputTensors[0].geometry;
    for (unsigned i = 0; i < 3; ++i) {
        auto& g = i < 2 ? in->inputTensors[i].geometry : in->outputTensors[0].geometry;
        if (g.dataType != DATA_F32) {
            g.dataType = DATA_F32;
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (a.dims != 2 || a.maxSizes[0] != 20480 || a.maxSizes[1] < 1 || a.maxSizes[1] > 64 ||
        w.dims != 2 || w.maxSizes[0] != 20480 || w.maxSizes[1] != 24)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (y.dims != 2 || y.maxSizes[0] != 24 || y.maxSizes[1] != a.maxSizes[1])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 24;
    const unsigned tile = batch4_ ? 4 : 1;
    out->indexSpaceGeometry[1] = (a.maxSizes[1] + tile - 1) / tile;
    for (unsigned i = 0; i < 2; ++i) {
        auto& mapping = out->inputTensorAccessPattern[i].mapping;
        mapping[0].indexSpaceDim = 0;
        mapping[0].a = 0;
        mapping[0].start_b = 0;
        mapping[0].end_b = 20479;
        mapping[1].indexSpaceDim = i == 0 ? 1 : 0;
        mapping[1].a = i == 0 ? tile : 1;
        mapping[1].start_b = 0;
        mapping[1].end_b = i == 0 ? tile - 1 : 0;
    }
    auto& mapping = out->outputTensorAccessPattern[0].mapping;
    for (unsigned i = 0; i < 2; ++i) {
        mapping[i].indexSpaceDim = i;
        mapping[i].a = i == 1 ? tile : 1;
        mapping[i].start_b = 0;
        mapping[i].end_b = i == 1 ? tile - 1 : 0;
    }
    out->kernel.paramsNr = 0;
    auto* begin = prefetch_ ? &_binary___deepseek_v41_control_prefetch_f32_gaudi2_o_start : batch4_ ? &_binary___deepseek_v41_control_batch4_f32_gaudi2_o_start
                          : &_binary___deepseek_v41_control_gemv_f32_gaudi2_o_start;
    auto* end = prefetch_ ? &_binary___deepseek_v41_control_prefetch_f32_gaudi2_o_end : batch4_ ? &_binary___deepseek_v41_control_batch4_f32_gaudi2_o_end
                        : &_binary___deepseek_v41_control_gemv_f32_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
