// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_final_collapse_norm_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char
    _binary___deepseek_v41_final_collapse_norm_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_final_collapse_norm_bf16_gaudi2_o_end;

namespace {
void map(tpc_lib_api::TensorAccessPattern& p, int dim, int index, int a,
         int first, int last) {
    p.mapping[dim].indexSpaceDim = index;
    p.mapping[dim].a = a;
    p.mapping[dim].start_b = first;
    p.mapping[dim].end_b = last;
}
}

tpc_lib_api::GlueCodeReturn
DeepseekV41FinalCollapseNormGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams ||
        p->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;
    if (p->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& residual = p->inputTensors[0].geometry;
    const auto& pre_mix = p->inputTensors[1].geometry;
    const auto& weight = p->inputTensors[2].geometry;
    const auto& output = p->outputTensors[0].geometry;
    if (residual.dataType != DATA_BF16 || pre_mix.dataType != DATA_F32 ||
        weight.dataType != DATA_BF16 || output.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto tokens = residual.maxSizes[2];
    if (residual.dims != 3 || residual.maxSizes[0] != 5120 ||
        residual.maxSizes[1] != 4 || tokens < 1 || tokens > 2 ||
        pre_mix.dims != 2 || pre_mix.maxSizes[0] != 4 ||
        pre_mix.maxSizes[1] != tokens || weight.dims != 1 ||
        weight.maxSizes[0] != 5120 || output.dims != 2 ||
        output.maxSizes[0] != 5120 || output.maxSizes[1] != tokens)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* params = static_cast<const float*>(p->nodeParams.nodeParams);
    if (!(params[0] > 0) || !std::isnormal(params[0])) return GLUE_FAILED;

    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = tokens;
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, 5119);
    map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 3);
    map(out->inputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, 0, 0, 0, 3);
    map(out->inputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, 5119);
    map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 5119);
    map(out->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);

    const float kernel_params[] = {params[0], 1.0f / 5120.0f};
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, kernel_params,
                sizeof(kernel_params));
    auto* begin =
        &_binary___deepseek_v41_final_collapse_norm_bf16_gaudi2_o_start;
    auto* end =
        &_binary___deepseek_v41_final_collapse_norm_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize)
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!out->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
