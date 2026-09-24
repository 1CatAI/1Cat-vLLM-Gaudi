// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_main_decode_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_prefill_main_decode_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_main_decode_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41PrefillMainDecodeGaudi2::GetGcDefinitions(
    HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(int))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int ratio = *static_cast<const int*>(p->nodeParams.nodeParams);
    const auto& cache = p->inputTensors[0].geometry;
    const auto& pages = p->inputTensors[1].geometry;
    const auto& rows = p->inputTensors[2].geometry;
    const auto& result = p->outputTensors[0].geometry;
    if ((ratio != 0 && ratio != 1 && ratio != 2) || cache.dims != 2 || cache.maxSizes[0] != 288 ||
        cache.dataType != DATA_U8 || !cache.maxSizes[1] || cache.maxSizes[1] > 1048704 ||
        pages.dims != 1 || pages.dataType != DATA_I32 || !pages.maxSizes[0] ||
        pages.maxSizes[0] > 8193 || rows.dims != 1 || rows.dataType != DATA_I32 ||
        !rows.maxSizes[0] || rows.maxSizes[0] > 65536 || result.dims != 2 ||
        result.dataType != DATA_BF16 || result.maxSizes[0] != 512 ||
        result.maxSizes[1] != rows.maxSizes[0] ||
        (ratio == 0 && cache.maxSizes[1] < rows.maxSizes[0]))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = rows.maxSizes[0];
    if (ratio == 0) {
        out->inputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 287};
        out->inputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
    } else {
        out->inputTensorAccessPattern[0].allRequired = true;
        out->inputTensorAccessPattern[0].sparseAccess = true;
        out->inputTensorAccessPattern[1].allRequired = true;
        out->inputTensorAccessPattern[1].sparseAccess = true;
    }
    out->inputTensorAccessPattern[2].mapping[0] = {0, 1, 0, 0};
    out->outputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 511};
    out->outputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &ratio, sizeof(ratio));
    const auto size = &_binary___deepseek_v41_prefill_main_decode_gaudi2_o_end -
                      &_binary___deepseek_v41_prefill_main_decode_gaudi2_o_start;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,
                &_binary___deepseek_v41_prefill_main_decode_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
