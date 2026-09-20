// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_kv_norm_rope_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___deepseek_v41_kv_norm_rope_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_kv_norm_rope_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41KVNormRopeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams ||
        p->nodeParams.nodeParamsSize != 2 * sizeof(float))
        return GLUE_FAILED;
    if (p->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x = p->inputTensors[0].geometry;
    const auto& w = p->inputTensors[1].geometry;
    const auto& pos = p->inputTensors[2].geometry;
    const auto& phase = p->inputTensors[3].geometry;
    const auto& y = p->outputTensors[0].geometry;
    const auto rows = x.maxSizes[1];
    if (x.dataType != DATA_BF16 || w.dataType != DATA_BF16 ||
        pos.dataType != DATA_I32 || phase.dataType != DATA_F32 ||
        y.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (x.dims != 2 || x.maxSizes[0] != 512 || rows < 1 || rows > 512 ||
        w.dims != 1 || w.maxSizes[0] != 512 ||
        pos.dims != 1 || pos.maxSizes[0] != rows ||
        phase.dims != 2 || phase.maxSizes[0] != 64 ||
        phase.maxSizes[1] < 1 || phase.maxSizes[1] > 1048576 ||
        y.dims != 2 || y.maxSizes[0] != 512 || y.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* scalar = static_cast<const float*>(p->nodeParams.nodeParams);
    if (!(scalar[0] > 0) || !std::isnormal(scalar[0]) ||
        scalar[1] != 1.0f / 512.0f)
        return GLUE_FAILED;

    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = rows;
    auto rows_map = [](TensorAccessPattern& pattern, unsigned width) {
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = 0;
        pattern.mapping[0].start_b = 0;
        pattern.mapping[0].end_b = width - 1;
        pattern.mapping[1].indexSpaceDim = 0;
        pattern.mapping[1].a = 1;
        pattern.mapping[1].start_b = pattern.mapping[1].end_b = 0;
    };
    rows_map(out->inputTensorAccessPattern[0], 512);
    rows_map(out->outputTensorAccessPattern[0], 512);
    out->inputTensorAccessPattern[1].allRequired = true;
    out->inputTensorAccessPattern[2].mapping[0] = {0, 1, 0, 0};
    // Runtime positions select phase rows, so retain the complete bounded
    // table as the accurate dynamic access contract.
    out->inputTensorAccessPattern[3].allRequired = true;
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, scalar, 2 * sizeof(float));
    auto* begin = &_binary___deepseek_v41_kv_norm_rope_bf16_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_kv_norm_rope_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
