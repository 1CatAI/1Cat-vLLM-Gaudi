// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v4_sinkhorn4_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v4_sinkhorn4_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_sinkhorn4_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV4Sinkhorn4Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_deepseek_v4_sinkhorn4_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV4Sinkhorn4Gaudi2::GetGcDefinitions(
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
    for (unsigned i = 0; i < 2; ++i) {
        auto& tensor = i == 0 ? in->inputTensors[0] : in->outputTensors[0];
        if (tensor.geometry.dataType != DATA_F32) {
            tensor.geometry.dataType = DATA_F32;
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (tensor.geometry.dims != 3 || tensor.geometry.maxSizes[0] != 4 ||
            tensor.geometry.maxSizes[1] != 4 || tensor.geometry.maxSizes[2] < 1 ||
            tensor.geometry.maxSizes[2] > 8192 ||
            tensor.geometry.maxSizes[2] != in->inputTensors[0].geometry.maxSizes[2]) {
            return i == 0 ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        auto& access = i == 0 ? out->inputTensorAccessPattern[0] : out->outputTensorAccessPattern[0];
        for (unsigned dim = 0; dim < 3; ++dim) {
            access.mapping[dim].indexSpaceDim = 0;
            access.mapping[dim].a = dim == 2 ? 1 : 0;
            access.mapping[dim].start_b = 0;
            access.mapping[dim].end_b = dim == 2 ? 0 : 3;
        }
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = in->inputTensors[0].geometry.maxSizes[2];
    out->kernel.paramsNr = 0;
    auto* start = &_binary___deepseek_v4_sinkhorn4_gaudi2_o_start;
    auto* end = &_binary___deepseek_v4_sinkhorn4_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
