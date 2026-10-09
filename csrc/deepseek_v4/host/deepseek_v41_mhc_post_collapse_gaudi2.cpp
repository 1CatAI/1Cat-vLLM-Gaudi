// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mhc_post_collapse_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_mhc_post_collapse_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mhc_post_collapse_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41MhcPostCollapseGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto tokens = in->inputTensors[0].geometry.maxSizes[1];
    if (!tokens || tokens > 6) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto value_dims = in->inputTensors[0].geometry.dims;
    const auto ranks = value_dims == 3 ? in->inputTensors[0].geometry.maxSizes[2] : 1;
    if ((value_dims != 2 && value_dims != 3) || (value_dims == 3 && (ranks < 2 || ranks > 8)))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t sizes[7][3] = {{5120,tokens,ranks},{5120,4,tokens},{4,tokens,0},{4,4,tokens},
                                {4,tokens,0},{5120,4,tokens},{5120,tokens,0}};
    const unsigned dims[7] = {value_dims,3,2,3,2,3,2};
    for (unsigned i = 0; i < 7; ++i) {
        const bool input = i < 5;
        const auto& tensor = input ? in->inputTensors[i] : in->outputTensors[i-5];
        const auto& g = tensor.geometry;
        const auto type = i >= 2 && i <= 4 ? DATA_F32 : DATA_BF16;
        if (g.dataType != type) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (g.dims != dims[i]) return input ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto& ap = input ? out->inputTensorAccessPattern[i] : out->outputTensorAccessPattern[i-5];
        for (unsigned d = 0; d < dims[i]; ++d) {
            if (g.maxSizes[d] != sizes[i][d]) return input ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            if (i == 0 && d == 2) ap.mapping[d] = {0,0,0,float(ranks-1),false};
            else if ((i == 0 && d == 1) || (i != 0 && d == dims[i]-1)) ap.mapping[d] = {1,1,0,0,false};
            else if (d == 0 && type == DATA_BF16) ap.mapping[d] = {0,128,0,127,false};
            else ap.mapping[d] = {0,0,0,float(g.maxSizes[d]-1),false};
        }
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 40;
    out->indexSpaceGeometry[1] = tokens;
    out->kernel.paramsNr = 0;
    auto* first = &_binary___deepseek_v41_mhc_post_collapse_gaudi2_o_start;
    auto* last = &_binary___deepseek_v41_mhc_post_collapse_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = last-first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
