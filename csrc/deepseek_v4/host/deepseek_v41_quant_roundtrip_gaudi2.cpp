// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_quant_roundtrip_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_quant_roundtrip_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_quant_roundtrip_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41QuantRoundtripGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_deepseek_v41_quant_roundtrip_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41QuantRoundtripGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) {
        in->inputTensorNr = 1;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto& shape = in->inputTensors[0].geometry;
    for (unsigned index = 0; index < 2; ++index) {
        auto& geometry = index == 0 ? in->inputTensors[0].geometry : in->outputTensors[0].geometry;
        if (geometry.dataType != DATA_BF16) {
            geometry.dataType = DATA_BF16;
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (geometry.dims != 2 || geometry.maxSizes[0] == 0 || geometry.maxSizes[0] % 32 ||
            geometry.maxSizes[0] > 131072 || geometry.maxSizes[1] == 0 || geometry.maxSizes[1] > 8192 ||
            geometry.maxSizes[0] != shape.maxSizes[0] || geometry.maxSizes[1] != shape.maxSizes[1]) {
            return index == 0 ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        auto& access = index == 0 ? out->inputTensorAccessPattern[0] : out->outputTensorAccessPattern[0];
        access.mapping[0].indexSpaceDim = 0;
        access.mapping[0].a = 32;
        access.mapping[0].start_b = 0;
        access.mapping[0].end_b = 31;
        access.mapping[1].indexSpaceDim = 1;
        access.mapping[1].a = 1;
        access.mapping[1].start_b = 0;
        access.mapping[1].end_b = 0;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = shape.maxSizes[0] / 32;
    out->indexSpaceGeometry[1] = shape.maxSizes[1];
    out->kernel.paramsNr = 0;
    auto* start = &_binary___deepseek_v41_quant_roundtrip_bf16_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_quant_roundtrip_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
