// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_kv_norm_rope_publish_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___deepseek_v41_kv_norm_rope_publish_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_kv_norm_rope_publish_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41KVNormRopePublishGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams ||
        p->nodeParams.nodeParamsSize != 2 * sizeof(float) + sizeof(int))
        return GLUE_FAILED;
    if (p->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
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
    if (x.dims != 2 || x.maxSizes[0] != 512 || rows != 1 ||
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

    const auto& cache = p->inputTensors[4].geometry;
    const auto& decoded = p->inputTensors[5].geometry;
    const auto& completion = p->outputTensors[1].geometry;
    const int offset = *reinterpret_cast<const int*>(scalar + 2);
    if (cache.dataType != DATA_U8 || decoded.dataType != DATA_BF16 || completion.dataType != DATA_I32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (cache.dims != 2 || cache.maxSizes[0] != 528 || cache.maxSizes[1] < 256 ||
        decoded.dims != 2 || decoded.maxSizes[0] != 512 || decoded.maxSizes[1] < 512 ||
        completion.dims != 1 || completion.maxSizes[0] != 16 ||
        (offset != -1 && (offset < 0 || offset % 512 || static_cast<uint64_t>(offset) + 512 > decoded.maxSizes[1])))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = 16;
    out->inputTensorAccessPattern[0].allRequired = true;
    out->outputTensorAccessPattern[0].mapping[0] = {0, 32, 0, 31};
    out->outputTensorAccessPattern[0].mapping[1] = {0, 0, 0, 0};
    out->inputTensorAccessPattern[1].allRequired = true;
    out->inputTensorAccessPattern[2].allRequired = true;
    // Runtime positions select phase rows, so retain the complete bounded
    // table as the accurate dynamic access contract.
    out->inputTensorAccessPattern[3].allRequired = true;
    out->inputTensorAccessPattern[4].allRequired = true;
    out->inputTensorAccessPattern[5].allRequired = true;
    out->outputTensorAccessPattern[1].mapping[0] = {0, 1, 0, 0};
    out->kernel.paramsNr = 3;
    std::memcpy(out->kernel.scalarParams, scalar, 2 * sizeof(float) + sizeof(int));
    auto* begin = &_binary___deepseek_v41_kv_norm_rope_publish_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_kv_norm_rope_publish_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
