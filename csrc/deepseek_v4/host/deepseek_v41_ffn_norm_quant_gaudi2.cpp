// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_ffn_norm_quant_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___deepseek_v41_ffn_norm_quant_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_ffn_norm_quant_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41FfnNormQuantGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams ||
        p->nodeParams.nodeParamsSize != 2 * sizeof(float))
        return GLUE_FAILED;
    if (p->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x = p->inputTensors[0].geometry;
    const auto& w = p->inputTensors[1].geometry;
    const auto& y = p->outputTensors[0].geometry;
    const auto& q = p->outputTensors[1].geometry;
    const auto& s = p->outputTensors[2].geometry;
    if (x.dataType != DATA_BF16 || w.dataType != DATA_BF16 ||
        y.dataType != DATA_BF16 || q.dataType != DATA_F8_143 ||
        s.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto width = x.maxSizes[0], rows = x.maxSizes[1];
    if (x.dims != 2 || width != 5120 || rows < 1 || rows > 8192 ||
        w.dims != 1 || w.maxSizes[0] != width ||
        y.dims != 2 || y.maxSizes[0] != width || y.maxSizes[1] != rows ||
        q.dims != 2 || q.maxSizes[0] != width || q.maxSizes[1] != rows ||
        s.dims != 2 || s.maxSizes[0] != 1 || s.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* scalar = static_cast<const float*>(p->nodeParams.nodeParams);
    if (!(scalar[0] > 0) || !std::isnormal(scalar[0]) ||
        scalar[1] != 1.0f / float(width))
        return GLUE_FAILED;

    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = rows;
    auto map = [width](TensorAccessPattern& pattern, unsigned columns,
                       bool row) {
        std::memset(&pattern, 0, sizeof(pattern));
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = 0;
        pattern.mapping[0].end_b = columns - 1;
        if (row) {
            pattern.mapping[1].indexSpaceDim = 0;
            pattern.mapping[1].a = 1;
        }
    };
    map(out->inputTensorAccessPattern[0], width, true);
    map(out->inputTensorAccessPattern[1], width, false);
    map(out->outputTensorAccessPattern[0], width, true);
    map(out->outputTensorAccessPattern[1], width, true);
    map(out->outputTensorAccessPattern[2], 1, true);
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, scalar, 2 * sizeof(float));
    auto* begin = &_binary___deepseek_v41_ffn_norm_quant_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_ffn_norm_quant_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize)
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!out->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
