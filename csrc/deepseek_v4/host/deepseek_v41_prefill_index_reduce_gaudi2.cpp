// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_index_reduce_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_prefill_index_reduce_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_index_reduce_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillIndexReduceGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != 2 * sizeof(int))
        return GLUE_FAILED;
    const auto* parameters = static_cast<const int*>(p->nodeParams.nodeParams);
    const int ratio = parameters[0], local_heads = parameters[1];
    if (local_heads != 8 && local_heads != 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (ratio != 1 && ratio != 2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (p->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& y = p->outputTensors[0].geometry;
    if (y.dims != 2 || y.dataType != DATA_F32 || y.maxSizes[0] < 128 ||
        y.maxSizes[0] > (y.maxSizes[1] <= 6 ? 32768 : 2048) || y.maxSizes[0] % 128 ||
        y.maxSizes[1] < 1 || y.maxSizes[1] > 16384) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const unsigned columns = y.maxSizes[0], tokens = y.maxSizes[1];
    for (unsigned i = 0; i < 4; ++i) {
        const auto& x = p->inputTensors[i].geometry;
        auto& access = out->inputTensorAccessPattern[i];
        access.allRequired = false;
        if (x.dataType != (i < 2 ? DATA_BF16 : DATA_I32)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (i == 0) {
            if (x.dims != 3 || x.maxSizes[0] != columns || x.maxSizes[1] != 32 || x.maxSizes[2] != tokens)
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            access.mapping[0] = {0, 128, 0, 127};
            access.mapping[1] = {0, 0, 0, 31};
            access.mapping[2] = {1, 1, 0, 0};
        } else if (i == 1) {
            if (x.dims != 2 || x.maxSizes[0] != 32 || x.maxSizes[1] != tokens)
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            access.mapping[0] = {0, 0, 0, 31};
            access.mapping[1] = {1, 1, 0, 0};
        } else {
            const bool query_rows = i == 3 && tokens <= 6 && x.dims == 2 && x.maxSizes[1] == tokens;
            if ((x.dims != 1 && !query_rows) || x.maxSizes[0] != (i == 2 ? tokens : columns))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if (i == 2) access.mapping[0] = {1, 1, 0, 0};
            else {
                access.mapping[0] = {0, 128, 0, 127};
                if (query_rows) access.mapping[1] = {1, 1, 0, 0};
            }
        }
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = columns / 128;
    out->indexSpaceGeometry[1] = tokens;
    out->outputTensorAccessPattern[0].allRequired = false;
    out->outputTensorAccessPattern[0].mapping[0] = {0, 128, 0, 127};
    out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, p->nodeParams.nodeParams, 2 * sizeof(int));
    const auto* begin = &_binary___deepseek_v41_prefill_index_reduce_gaudi2_o_start;
    const auto* end = &_binary___deepseek_v41_prefill_index_reduce_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
