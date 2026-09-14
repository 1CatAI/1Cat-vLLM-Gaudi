// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_woa_stage_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_woa_stage_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_woa_stage_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41WoaStageGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    for (const auto* tensor : {&p->inputTensors[0], &p->outputTensors[0]}) {
        const auto& t = tensor->geometry;
        if (t.dims != 3 || t.dataType != DATA_F8_143 || t.maxSizes[0] != 1024 ||
            t.maxSizes[1] != 4096 || t.maxSizes[2] != 4) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = 4; out->indexSpaceGeometry[1] = 256; out->indexSpaceGeometry[2] = 4;
    for (auto* pattern : {&out->inputTensorAccessPattern[0], &out->outputTensorAccessPattern[0]}) {
        for (int dim = 0; dim < 3; ++dim) {
            const int span = dim == 0 ? 256 : dim == 1 ? 16 : 1;
            auto& m = pattern->mapping[dim];
            m.indexSpaceDim = dim; m.a = span; m.start_b = 0; m.end_b = span - 1;
        }
    }
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = &_binary___deepseek_v41_woa_stage_gaudi2_o_end - &_binary___deepseek_v41_woa_stage_gaudi2_o_start;
    out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary___deepseek_v41_woa_stage_gaudi2_o_start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
