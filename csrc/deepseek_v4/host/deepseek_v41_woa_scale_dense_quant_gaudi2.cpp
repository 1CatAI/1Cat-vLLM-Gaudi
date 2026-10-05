// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_woa_scale_dense_quant_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_woa_scale_dense_quant_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_woa_scale_dense_quant_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41WoaScaleDenseQuantGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (in->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    for (unsigned i = 0; i < 3; ++i)
        if (in->inputTensors[i].geometry.dataType != DATA_F32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& p = in->inputTensors[0].geometry;
    const auto& w = in->inputTensors[1].geometry;
    const auto& s = in->inputTensors[2].geometry;
    const auto& q = in->outputTensors[0].geometry;
    const auto& qs = in->outputTensors[1].geometry;
    const auto groups = p.maxSizes[2];
    if (p.dims != 3 || p.maxSizes[0] != 1024 || p.maxSizes[1] != 1 || (groups != 2 && groups != 4) ||
        w.dims != 3 || w.maxSizes[0] != 1024 || w.maxSizes[1] != 1 || w.maxSizes[2] != groups ||
        s.dims != 3 || s.maxSizes[0] != 1 || s.maxSizes[1] != 1 || s.maxSizes[2] != groups)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (q.dataType != DATA_F8_143 || qs.dataType != DATA_F32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (q.dims != 2 || q.maxSizes[0] != groups * 1024 || q.maxSizes[1] != 1 ||
        qs.dims != 2 || qs.maxSizes[0] != 1 || qs.maxSizes[1] != 1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = groups * 8;
    for (unsigned i = 0; i < 3; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    auto& mapping = out->outputTensorAccessPattern[0].mapping[0];
    mapping.indexSpaceDim = 0; mapping.a = 128; mapping.start_b = 0; mapping.end_b = 127;
    out->outputTensorAccessPattern[1].allRequired = true;
    out->kernel.paramsNr = 0;
    auto* first = &_binary___deepseek_v41_woa_scale_dense_quant_gaudi2_o_start;
    auto* last = &_binary___deepseek_v41_woa_scale_dense_quant_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize; out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
