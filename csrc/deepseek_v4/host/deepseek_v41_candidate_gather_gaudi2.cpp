// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_candidate_gather_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_candidate_gather_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_candidate_gather_f32_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41CandidateGatherGaudi2::GetGcDefinitions(
        HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(int))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int ratio = *static_cast<const int*>(p->nodeParams.nodeParams);
    const auto& common = p->inputTensors[0].geometry;
    const auto& blocks = p->inputTensors[1].geometry;
    const auto& positions = p->inputTensors[2].geometry;
    if ((ratio != 1 && ratio != 2) || common.dims != 2 || common.dataType != DATA_F32 ||
        !common.maxSizes[0] || common.maxSizes[0] > 32768 || common.maxSizes[0] % 8 ||
        !common.maxSizes[1] || common.maxSizes[1] > 128 || blocks.dims != 2 || blocks.dataType != DATA_I32 ||
        !blocks.maxSizes[0] || blocks.maxSizes[0] > 2048 || blocks.maxSizes[1] != common.maxSizes[1] ||
        positions.dims != 1 || positions.dataType != DATA_I32 || positions.maxSizes[0] != common.maxSizes[1])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i = 0; i < 2; ++i) {
        const auto& tensor = p->outputTensors[i].geometry;
        if (tensor.dims != 2 || tensor.dataType != (i ? DATA_I32 : DATA_F32) ||
            tensor.maxSizes[0] != blocks.maxSizes[0] * 8 || tensor.maxSizes[1] != common.maxSizes[1])
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (blocks.maxSizes[0] + 7) / 8;
    out->indexSpaceGeometry[1] = common.maxSizes[1];
    out->inputTensorAccessPattern[0].allRequired = true;
    out->inputTensorAccessPattern[0].sparseAccess = true;
    out->inputTensorAccessPattern[1].mapping[0] = {0, 8, 0, 7};
    out->inputTensorAccessPattern[1].mapping[1] = {1, 1, 0, 0};
    out->inputTensorAccessPattern[2].mapping[0] = {1, 1, 0, 0};
    for (unsigned i = 0; i < 2; ++i) {
        out->outputTensorAccessPattern[i].mapping[0] = {0, 64, 0, 63};
        out->outputTensorAccessPattern[i].mapping[1] = {1, 1, 0, 0};
    }
    const int parameters[3] = {ratio, int(common.maxSizes[0]), int(blocks.maxSizes[0])};
    out->kernel.paramsNr = 3;
    std::memcpy(out->kernel.scalarParams, parameters, sizeof(parameters));
    const auto size = &_binary___deepseek_v41_candidate_gather_f32_gaudi2_o_end -
                      &_binary___deepseek_v41_candidate_gather_f32_gaudi2_o_start;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary___deepseek_v41_candidate_gather_f32_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
