// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_control_gemv_rrms_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char
    _binary___deepseek_v41_control_gemv_rrms_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_control_gemv_rrms_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn
DeepseekV41ControlGemvRrmsGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams ||
        p->nodeParams.nodeParamsSize != 2 * sizeof(float))
        return GLUE_FAILED;
    if (p->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x = p->inputTensors[0].geometry;
    const auto& w = p->inputTensors[1].geometry;
    const auto& y = p->outputTensors[0].geometry;
    if (x.dataType != DATA_BF16 || w.dataType != DATA_F32 ||
        y.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto tokens = x.maxSizes[1];
    if (x.dims != 2 || x.maxSizes[0] != 20480 || tokens < 1 ||
        tokens > 2048 ||
        w.dims != 2 || w.maxSizes[0] != 20480 || w.maxSizes[1] != 24 ||
        y.dims != 2 || y.maxSizes[0] != 25 || y.maxSizes[1] != tokens)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* scalar = static_cast<const float*>(p->nodeParams.nodeParams);
    if (!(scalar[0] > 0.0f) || !std::isfinite(scalar[0]) ||
        scalar[1] != 1.0f / 20480.0f)
        return GLUE_FAILED;

    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 24;
    out->indexSpaceGeometry[1] = tokens;
    for (unsigned i = 0; i < 2; ++i) {
        std::memset(&out->inputTensorAccessPattern[i], 0,
                    sizeof(TensorAccessPattern));
    }
    std::memset(&out->outputTensorAccessPattern[0], 0,
                sizeof(TensorAccessPattern));
    auto& xm = out->inputTensorAccessPattern[0].mapping;
    xm[0].indexSpaceDim = 0;
    xm[0].a = 0;
    xm[0].end_b = 20479;
    xm[1].indexSpaceDim = 1;
    xm[1].a = 1;
    auto& wm = out->inputTensorAccessPattern[1].mapping;
    wm[0].indexSpaceDim = 0;
    wm[0].a = 0;
    wm[0].end_b = 20479;
    wm[1].indexSpaceDim = 0;
    wm[1].a = 1;
    auto& ym = out->outputTensorAccessPattern[0].mapping;
    ym[0].indexSpaceDim = 0;
    ym[0].a = 0;
    ym[0].end_b = 24;
    ym[1].indexSpaceDim = 1;
    ym[1].a = 1;

    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, scalar, 2 * sizeof(float));
    auto* begin =
        &_binary___deepseek_v41_control_gemv_rrms_bf16_gaudi2_o_start;
    auto* end =
        &_binary___deepseek_v41_control_gemv_rrms_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize)
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!out->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
