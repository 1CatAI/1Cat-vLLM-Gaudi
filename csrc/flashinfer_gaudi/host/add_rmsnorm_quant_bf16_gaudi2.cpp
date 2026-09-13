// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstring>
#include <limits>
#include "add_rmsnorm_quant_bf16_gaudi2.hpp"

extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_small_o_start;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_small_o_end;

tpc_lib_api::GlueCodeReturn AddRmsNormQuantBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 4) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(AddRmsNormQuantParams)) return GLUE_FAILED;
    AddRmsNormQuantParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const auto& input = params->inputTensors[0].geometry;
    const auto width = input.maxSizes[0], rows = input.maxSizes[1];
    if (input.dims != 2 || !rows || rows > std::numeric_limits<int32_t>::max() ||
        width < 256 || width > 17408 || width % 128 ||
        !std::isnormal(scalar.epsilon) || scalar.epsilon <= 0 ||
        scalar.inverse_width != 1.0f / static_cast<float>(width) ||
        (scalar.inverse_range != 1.0f / 240.0f && scalar.inverse_range != 0.004180908203125f))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (int index = 0; index < 3; ++index) {
        const auto& geometry = params->inputTensors[index].geometry;
        if (geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (geometry.dims != (index == 2 ? 1u : 2u) || geometry.maxSizes[0] != width ||
            (index != 2 && geometry.maxSizes[1] != rows)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (int index = 0; index < 4; ++index) {
        const auto& geometry = params->outputTensors[index].geometry;
        const auto dtype = index == 0 ? DATA_F8_143 : index == 1 ? DATA_F32 : DATA_BF16;
        if (geometry.dataType != dtype) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (geometry.dims != 2 || geometry.maxSizes[0] != (index == 1 ? 1 : width) ||
            geometry.maxSizes[1] != rows) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = rows;
    auto map = [](TensorAccessPattern& pattern, int columns, bool row) {
        std::memset(&pattern, 0, sizeof(pattern));
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].end_b = columns - 1;
        if (row) {
            pattern.mapping[1].indexSpaceDim = 0;
            pattern.mapping[1].a = 1;
        }
    };
    for (int index = 0; index < 3; ++index) map(instance->inputTensorAccessPattern[index], width, index != 2);
    for (int index = 0; index < 4; ++index) map(instance->outputTensorAccessPattern[index], index == 1 ? 1 : width, true);
    instance->kernel.paramsNr = sizeof(scalar) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));
    // The lookup implementation has a smaller VLM budget. Its cache must never
    // serve widths above 8192; both variants keep the identical scalar ABI.
    const auto* begin = width <= 8192 ? &_binary_add_rmsnorm_quant_bf16_gaudi2_small_o_start :
                                       &_binary_add_rmsnorm_quant_bf16_gaudi2_o_start;
    const auto* end = width <= 8192 ? &_binary_add_rmsnorm_quant_bf16_gaudi2_small_o_end :
                                     &_binary_add_rmsnorm_quant_bf16_gaudi2_o_end;
    const unsigned size = end - begin;
    const unsigned capacity = instance->kernel.elfSize;
    instance->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, begin, size);
    return GLUE_SUCCESS;
}
