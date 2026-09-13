// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_csa2_prep_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_rope_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_rope_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_c1_indices_i32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_c1_indices_i32_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41Csa2PrepGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, rope_ ? "custom_deepseek_v41_rope_bf16_gaudi2" : "custom_deepseek_v41_c1_indices_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41Csa2PrepGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != (rope_ ? 3u : 2u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (in->outputTensorNr != (rope_ ? 1u : 2u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    for (unsigned i = 0; i < in->inputTensorNr; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    if (rope_) {
        const auto& value = in->inputTensors[0].geometry;
        const auto& pos = in->inputTensors[1].geometry;
        const auto& table = in->inputTensors[2].geometry;
        const auto& result = in->outputTensors[0].geometry;
        if (value.dataType != DATA_BF16 || pos.dataType != DATA_I32 || table.dataType != DATA_F32 ||
            result.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (value.dims != 3 || value.maxSizes[2] != 1 || (value.maxSizes[0] != 128 && value.maxSizes[0] != 512) ||
            value.maxSizes[1] == 0 || value.maxSizes[1] > 32 || pos.dims != 1 || pos.maxSizes[0] != 1 ||
            table.dims != 2 || table.maxSizes[0] != 64 || table.maxSizes[1] != 512)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (result.dims != 3 || std::memcmp(value.maxSizes, result.maxSizes, 3 * sizeof(value.maxSizes[0])))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = value.maxSizes[1];
        out->inputTensorAccessPattern[0].allRequired = false;
        out->inputTensorAccessPattern[0].mapping[0] = {0, 0, 0, static_cast<float>(value.maxSizes[0] - 1)};
        out->inputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
        out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
        out->kernel.paramsNr = 0;
    } else {
        const auto& pos = in->inputTensors[0].geometry;
        const auto& compressed = in->inputTensors[1].geometry;
        const auto& result = in->outputTensors[0].geometry;
        const auto& lengths = in->outputTensors[1].geometry;
        if (pos.dataType != DATA_I32 || compressed.dataType != DATA_I32 || result.dataType != DATA_I32 ||
            lengths.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(int)) return GLUE_NODE_NOT_FOUND;
        const int ratio = *static_cast<const int*>(in->nodeParams.nodeParams);
        if (ratio < 0 || ratio > 2 || pos.dims != 1 || pos.maxSizes[0] != 1 || compressed.dims != 2 ||
            compressed.maxSizes[0] != 512 || compressed.maxSizes[1] != 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (result.dims != 2 || result.maxSizes[0] != (ratio ? 640u : 128u) || result.maxSizes[1] != 1 ||
            lengths.dims != 1 || lengths.maxSizes[0] != 1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = result.maxSizes[0] / 64;
        out->outputTensorAccessPattern[0].mapping[0] = {0, 64, 0, 63};
        out->outputTensorAccessPattern[1].allRequired = true;
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = ratio;
    }
    auto* start = rope_ ? &_binary___deepseek_v41_rope_bf16_gaudi2_o_start :
                          &_binary___deepseek_v41_c1_indices_i32_gaudi2_o_start;
    auto* end = rope_ ? &_binary___deepseek_v41_rope_bf16_gaudi2_o_end :
                        &_binary___deepseek_v41_c1_indices_i32_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
