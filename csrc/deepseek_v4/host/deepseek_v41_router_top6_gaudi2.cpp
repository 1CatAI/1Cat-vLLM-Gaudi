// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_router_top6_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_router_top6_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_router_top6_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41RouterTop6Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x = p->inputTensors[0].geometry;
    const auto tokens = x.maxSizes[1];
    if (x.dataType != DATA_F32 || x.dims != 2 || x.maxSizes[0] != 384 || tokens < 1 || tokens > 8192)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (int i = 1; i < 4; ++i) {
        const auto& t = p->inputTensors[i].geometry;
        if (t.dims != 1 || t.maxSizes[0] != (i == 3 ? tokens : 384) ||
            t.dataType != (i == 3 ? DATA_I8 : DATA_F32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (int i = 0; i < 2; ++i) {
        const auto& t = p->outputTensors[i].geometry;
        if (t.dims != 2 || t.maxSizes[0] != 6 || t.maxSizes[1] != tokens ||
            t.dataType != (i == 0 ? DATA_I32 : DATA_F32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = tokens;
    for (int i = 0; i < 4; ++i) {
        auto& t = out->inputTensorAccessPattern[i];
        t.mapping[0].indexSpaceDim = 0; t.mapping[0].a = i == 3 ? 1 : 0;
        t.mapping[0].end_b = i == 3 ? 0 : 383;
        if (i == 0) {t.mapping[1].indexSpaceDim = 0; t.mapping[1].a = 1;}
    }
    for (int i = 0; i < 2; ++i) {
        auto& t = out->outputTensorAccessPattern[i];
        t.mapping[0].indexSpaceDim = 0; t.mapping[0].a = 0; t.mapping[0].end_b = 5;
        t.mapping[1].indexSpaceDim = 0; t.mapping[1].a = 1;
    }
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = &_binary___deepseek_v41_router_top6_gaudi2_o_end - &_binary___deepseek_v41_router_top6_gaudi2_o_start;
    out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary___deepseek_v41_router_top6_gaudi2_o_start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
