// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_attention_norm_gaudi2.hpp"
#include <cmath>
#include <cstring>
extern unsigned char _binary___deepseek_v41_attention_norm_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_attention_norm_bf16_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41AttentionNormGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(float)) return GLUE_FAILED;
    if (p->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x = p->inputTensors[0].geometry;
    const auto& w = p->inputTensors[1].geometry;
    const auto& y = p->outputTensors[0].geometry;
    if (x.dataType != DATA_BF16 || w.dataType != DATA_BF16 || y.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto width = x.maxSizes[0], rows = x.maxSizes[1];
    if (x.dims != 2 || w.dims != 1 || y.dims != 2 || (width != 512 && width != 1280) ||
        rows < 1 || rows > 512 || w.maxSizes[0] != width || y.maxSizes[0] != width || y.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* params = static_cast<const float*>(p->nodeParams.nodeParams);
    if (!(params[0] > 0) || !std::isnormal(params[0])) return GLUE_FAILED;
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = rows;
    auto map = [width](TensorAccessPattern& pattern, bool is_weight) {
        pattern.mapping[0].indexSpaceDim = 0; pattern.mapping[0].a = 0;
        pattern.mapping[0].start_b = 0; pattern.mapping[0].end_b = width - 1;
        if (!is_weight) {
            pattern.mapping[1].indexSpaceDim = 0; pattern.mapping[1].a = 1;
            pattern.mapping[1].start_b = pattern.mapping[1].end_b = 0;
        }
    };
    map(out->inputTensorAccessPattern[0], false); map(out->inputTensorAccessPattern[1], true);
    map(out->outputTensorAccessPattern[0], false);
    // Shape-derived parameters cannot be cached as scalar-only eager node parameters.
    const float kernel_params[] = {params[0], 1.0f / float(width)};
    out->kernel.paramsNr = 2; std::memcpy(out->kernel.scalarParams, kernel_params, sizeof(kernel_params));
    auto* begin = &_binary___deepseek_v41_attention_norm_bf16_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_attention_norm_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize; out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
