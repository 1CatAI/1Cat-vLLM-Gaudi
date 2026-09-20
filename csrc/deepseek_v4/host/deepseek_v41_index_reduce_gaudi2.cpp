// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_index_reduce_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_index_reduce_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_index_reduce_bf16_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41IndexReduceGaudi2::GetGcDefinitions(
    HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(int))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int ratio = *static_cast<const int*>(p->nodeParams.nodeParams);
    const auto& raw = p->inputTensors[0].geometry;
    const auto& weights = p->inputTensors[1].geometry;
    const auto& positions = p->inputTensors[2].geometry;
    if ((ratio != 1 && ratio != 2) || raw.dims != 3 || raw.dataType != DATA_BF16 ||
        raw.maxSizes[1] != 32 || raw.maxSizes[0] < 512 || raw.maxSizes[0] > 16384 ||
        raw.maxSizes[0] % 64 || weights.dims != 2 || weights.dataType != DATA_BF16 ||
        weights.maxSizes[0] != 32 || weights.maxSizes[1] != raw.maxSizes[2] ||
        positions.dims != 1 || positions.dataType != DATA_I32 ||
        positions.maxSizes[0] != raw.maxSizes[2] || !raw.maxSizes[2] || raw.maxSizes[2] > 64)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& output = p->outputTensors[0].geometry;
    if (output.dims != 2 || output.dataType != DATA_F32 ||
        output.maxSizes[0] != raw.maxSizes[0] || output.maxSizes[1] != raw.maxSizes[2])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const uint64_t batch = raw.maxSizes[2];
    out->indexSpaceRank = batch == 1 ? 1 : 2;
    out->indexSpaceGeometry[0] = raw.maxSizes[0] / 64;
    if (batch > 1) out->indexSpaceGeometry[1] = batch;
    for (unsigned i = 0; i < p->inputTensorNr; ++i) {
        out->inputTensorAccessPattern[i].allRequired = true;
        out->inputTensorAccessPattern[i].sparseAccess = true;
    }
    out->outputTensorAccessPattern[0].allRequired = true;
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &ratio, sizeof(ratio));
    const auto size = &_binary___deepseek_v41_index_reduce_bf16_gaudi2_o_end -
                      &_binary___deepseek_v41_index_reduce_bf16_gaudi2_o_start;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,
                &_binary___deepseek_v41_index_reduce_bf16_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
