// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <limits>
#include "silu_mul_quant_bf16_gaudi2.hpp"

extern unsigned char _binary_silu_mul_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_silu_mul_quant_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn SiluMulQuantBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern) return GLUE_FAILED;
    const auto& input = params->inputTensors[0].geometry;
    const auto& silu = params->inputTensors[1].geometry;
    const auto& output = params->outputTensors[0].geometry;
    const auto& scale = params->outputTensors[1].geometry;
    if (input.dataType != DATA_BF16 || silu.dataType != DATA_BF16 ||
        output.dataType != DATA_F8_143 || scale.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto width = output.maxSizes[0];
    const auto rows = input.maxSizes[1];
    if (input.dims != 2 || silu.dims != 2 || output.dims != 2 || scale.dims != 2 || !rows ||
        rows > std::numeric_limits<int32_t>::max() || width < 256 || width > 17408 || width % 128 ||
        input.maxSizes[0] != 2 * width || output.maxSizes[1] != rows ||
        silu.maxSizes[0] != width || silu.maxSizes[1] != rows ||
        scale.maxSizes[0] != 1 || scale.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = rows;
    auto map_row = [](TensorAccessPattern& pattern, int columns) {
        std::memset(&pattern, 0, sizeof(pattern));
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = 0;
        pattern.mapping[0].end_b = columns - 1;
        pattern.mapping[1].indexSpaceDim = 0;
        pattern.mapping[1].a = 1;
    };
    map_row(instance->inputTensorAccessPattern[0], 2 * width);
    instance->inputTensorAccessPattern[0].mapping[0].start_b = width;
    map_row(instance->inputTensorAccessPattern[1], width);
    map_row(instance->outputTensorAccessPattern[0], width);
    map_row(instance->outputTensorAccessPattern[1], 1);
    instance->kernel.paramsNr = 0;
    const unsigned size = &_binary_silu_mul_quant_bf16_gaudi2_o_end - &_binary_silu_mul_quant_bf16_gaudi2_o_start;
    const unsigned capacity = instance->kernel.elfSize;
    instance->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, &_binary_silu_mul_quant_bf16_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
