// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_reindex_compact_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_reindex_compact_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_reindex_compact_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41ReindexCompactGaudi2::GetGcDefinitions(
    HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(int))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int ratio = *static_cast<const int*>(p->nodeParams.nodeParams);
    const auto& pool = p->inputTensors[0].geometry;
    const auto& pos = p->inputTensors[1].geometry;
    if ((ratio != 1 && ratio != 2) || pool.dims != 2 || pool.dataType != DATA_I32 ||
        pool.maxSizes[0] != 2048 || pool.maxSizes[1] < 1 || pool.maxSizes[1] > 64 ||
        pos.dims != 1 || pos.dataType != DATA_I32 || pos.maxSizes[0] != pool.maxSizes[1])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i = 0; i < 3; ++i) {
        const auto& g = p->outputTensors[i].geometry;
        if (g.dataType != DATA_I32 || g.dims != (i < 2 ? 2u : 1u) ||
            g.maxSizes[0] != (i < 2 ? 2048u : pool.maxSizes[1]) ||
            (i < 2 && g.maxSizes[1] != pool.maxSizes[1])) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = pool.maxSizes[1];
    out->inputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 2047};
    out->inputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
    out->inputTensorAccessPattern[1].mapping[0] = {0, 1, 0, 0};
    for (unsigned i = 0; i < 2; ++i) {
        out->outputTensorAccessPattern[i].mapping[0] = {0, 0, 0, 2047};
        out->outputTensorAccessPattern[i].mapping[1] = {0, 1, 0, 0};
    }
    out->outputTensorAccessPattern[2].mapping[0] = {0, 1, 0, 0};
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &ratio, sizeof(ratio));
    const auto size = &_binary___deepseek_v41_reindex_compact_gaudi2_o_end -
                      &_binary___deepseek_v41_reindex_compact_gaudi2_o_start;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary___deepseek_v41_reindex_compact_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
