// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <initializer_list>
#include <limits>
#include "block_fp8_dequant_gaudi2.hpp"

extern unsigned char _binary_block_fp8_dequant_gaudi2_o_start;
extern unsigned char _binary_block_fp8_dequant_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn BlockFp8DequantGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern) return GLUE_FAILED;
    const auto& weight = params->inputTensors[0].geometry;
    const auto& scale = params->inputTensors[1].geometry;
    const auto& output = params->outputTensors[0].geometry;
    if (weight.dataType != DATA_F8_143 || scale.dataType != DATA_F32 || output.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (weight.dims != 2 || scale.dims != 2 || output.dims != 2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned dim = 0; dim < 2; ++dim) {
        const auto size = weight.maxSizes[dim];
        if (!size || size > std::numeric_limits<int32_t>::max() || size % 128 ||
            output.maxSizes[dim] != size || scale.maxSizes[dim] != size / 128)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    instance->indexSpaceRank = 2;
    for (unsigned dim = 0; dim < 2; ++dim) instance->indexSpaceGeometry[dim] = scale.maxSizes[dim];
    instance->indexSpaceGeometry[0] = (scale.maxSizes[0] + 1) / 2;
    auto map = [](TensorAccessPattern& pattern, int tile) {
        std::memset(&pattern, 0, sizeof(pattern));
        for (unsigned dim = 0; dim < 2; ++dim) {
            pattern.mapping[dim].indexSpaceDim = dim;
            pattern.mapping[dim].a = tile;
            pattern.mapping[dim].end_b = tile - 1;
        }
    };
    map(instance->inputTensorAccessPattern[0], 128);
    map(instance->inputTensorAccessPattern[1], 1);
    map(instance->outputTensorAccessPattern[0], 128);
    for (auto* pattern : {&instance->inputTensorAccessPattern[0], &instance->outputTensorAccessPattern[0]}) {
        pattern->mapping[0].a = 256;
        pattern->mapping[0].end_b = 255;
    }
    instance->inputTensorAccessPattern[1].mapping[0].a = 2;
    instance->inputTensorAccessPattern[1].mapping[0].end_b = 1;
    instance->kernel.paramsNr = 0;
    const unsigned size = &_binary_block_fp8_dequant_gaudi2_o_end - &_binary_block_fp8_dequant_gaudi2_o_start;
    const unsigned capacity = instance->kernel.elfSize;
    instance->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, &_binary_block_fp8_dequant_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
