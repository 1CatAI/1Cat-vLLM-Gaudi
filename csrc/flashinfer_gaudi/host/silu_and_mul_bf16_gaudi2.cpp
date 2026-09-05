// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include "silu_and_mul_bf16_gaudi2.hpp"

extern unsigned char _binary_silu_and_mul_bf16_gaudi2_o_start;
extern unsigned char _binary_silu_and_mul_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn SiluAndMulBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance)
{
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern) return GLUE_FAILED;
    const auto& input = params->inputTensors[0].geometry;
    const auto& output = params->outputTensors[0].geometry;
    if (input.dataType != DATA_BF16 || output.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (input.dims != 2 || output.dims != 2 || !input.maxSizes[1] ||
        !input.maxSizes[0] || input.maxSizes[0] % 256 ||
        input.maxSizes[0] != 2 * output.maxSizes[0] || input.maxSizes[1] != output.maxSizes[1])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    instance->indexSpaceRank = 2;
    instance->indexSpaceGeometry[0] = output.maxSizes[0] / 128;
    instance->indexSpaceGeometry[1] = output.maxSizes[1];
    auto map = [](TensorAccessPattern& pattern, unsigned dim, int a, int end) {
        pattern.mapping[dim].indexSpaceDim = dim;
        pattern.mapping[dim].a = a;
        pattern.mapping[dim].start_b = 0;
        pattern.mapping[dim].end_b = end;
    };
    std::memset(&instance->inputTensorAccessPattern[0], 0, sizeof(TensorAccessPattern));
    std::memset(&instance->outputTensorAccessPattern[0], 0, sizeof(TensorAccessPattern));
    map(instance->inputTensorAccessPattern[0], 0, 128, static_cast<int>(output.maxSizes[0]) + 127);
    map(instance->inputTensorAccessPattern[0], 1, 1, 0);
    map(instance->outputTensorAccessPattern[0], 0, 128, 127);
    map(instance->outputTensorAccessPattern[0], 1, 1, 0);
    instance->kernel.paramsNr = 0;
    const unsigned size = &_binary_silu_and_mul_bf16_gaudi2_o_end - &_binary_silu_and_mul_bf16_gaudi2_o_start;
    const unsigned capacity = instance->kernel.elfSize;
    instance->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, &_binary_silu_and_mul_bf16_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
