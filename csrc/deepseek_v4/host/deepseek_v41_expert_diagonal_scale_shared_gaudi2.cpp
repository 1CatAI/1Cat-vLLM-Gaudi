// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_expert_diagonal_scale_shared_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_expert_diagonal_scale_shared_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_expert_diagonal_scale_shared_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertDiagonalScaleSharedGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (in->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorDataType types[] = {DATA_F32, DATA_F32, DATA_I32, DATA_F32, DATA_BF16, DATA_BF16};
    for (unsigned i = 0; i < 6; ++i)
        if (in->inputTensors[i].geometry.dataType != types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& channel = in->inputTensors[4].geometry;
    if (channel.dims != 3 || channel.maxSizes[0] != 256 || channel.maxSizes[1] != 20 ||
        !channel.maxSizes[2] || channel.maxSizes[2] > 384) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i = 0; i < 2; ++i) {
        const auto& p = in->inputTensors[i].geometry;
        if (p.dims != 2 || p.maxSizes[0] != 15360 || p.maxSizes[1] != 3)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const auto& ids = in->inputTensors[2].geometry;
    const auto& scale = in->inputTensors[3].geometry;
    const auto& shared = in->inputTensors[5].geometry;
    if (ids.dims != 2 || ids.maxSizes[0] != 6 || ids.maxSizes[1] != 1 ||
        scale.dims != 2 || scale.maxSizes[0] != 1 || scale.maxSizes[1] != 6 ||
        shared.dims != 2 || shared.maxSizes[0] != 5120 || shared.maxSizes[1] != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& result = in->outputTensors[0].geometry;
    if (result.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (result.dims != 3 || result.maxSizes[0] != 5120 || result.maxSizes[1] != 1 || result.maxSizes[2] != 1)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = 40;
    for (unsigned i = 0; i < 6; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    auto& mapping = out->outputTensorAccessPattern[0];
    mapping.mapping[0].indexSpaceDim = 0; mapping.mapping[0].a = 128;
    mapping.mapping[0].start_b = 0; mapping.mapping[0].end_b = 127;
    for (unsigned i = 1; i < 3; ++i) {
        mapping.mapping[i].indexSpaceDim = 0; mapping.mapping[i].a = 0;
        mapping.mapping[i].start_b = mapping.mapping[i].end_b = 0;
    }
    out->kernel.paramsNr = 0;
    const auto* first = &_binary___deepseek_v41_expert_diagonal_scale_shared_gaudi2_o_start;
    const auto* last = &_binary___deepseek_v41_expert_diagonal_scale_shared_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
