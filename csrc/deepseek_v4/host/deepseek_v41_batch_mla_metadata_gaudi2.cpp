// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_batch_mla_metadata_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_batch_mla_metadata_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_batch_mla_metadata_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41BatchMlaMetadataGaudi2::GetGcDefinitions(
        HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out || !p->nodeParams) return GLUE_FAILED;
    if (p->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto params = *static_cast<Dsv41BatchMlaMetadataParams*>(p->nodeParams);
    const auto& selected = p->inputTensors[0].geometry;
    const auto& pages = p->inputTensors[1].geometry;
    const auto batch = selected.maxSizes[1];
    if (selected.dims != 2 || selected.maxSizes[0] != 512 || batch < 1 || batch > 64 ||
        pages.dims != 2 || pages.maxSizes[0] < 1 || pages.maxSizes[1] != batch ||
        params.ratio < 0 || params.ratio > 2 || params.swa_rows < 256 ||
        params.swa_rows % 256 || (params.tile != 4 && params.tile != 8 && params.tile != 16))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i = 0; i < 6; ++i) {
        const auto& g = p->inputTensors[i].geometry;
        if (g.dataType != DATA_I32 || (i >= 2 && (g.dims != 1 || g.maxSizes[0] != batch)))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->inputTensorAccessPattern[i].allRequired = true;
    }
    out->inputTensorAccessPattern[1].sparseAccess = true;
    const unsigned width = params.ratio ? 640 : 128;
    for (unsigned i = 0; i < 3; ++i) {
        const auto& g = p->outputTensors[i].geometry;
        if (g.dataType != DATA_I32 || (i < 2 ?
            (g.dims != 2 || g.maxSizes[0] != width || g.maxSizes[1] != batch) :
            (g.dims != 1 || g.maxSizes[0] != batch)))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = width / 16;
    out->indexSpaceGeometry[1] = batch;
    for (unsigned i = 0; i < 2; ++i) {
        out->outputTensorAccessPattern[i].mapping[0] = {0, 16, 0, 15};
        out->outputTensorAccessPattern[i].mapping[1] = {1, 1, 0, 0};
    }
    out->outputTensorAccessPattern[2].mapping[0] = {1, 1, 0, 0};
    out->kernel.paramsNr = sizeof(params) / sizeof(int);
    std::memcpy(out->kernel.scalarParams, &params, sizeof(params));
    const auto size = &_binary___deepseek_v41_batch_mla_metadata_gaudi2_o_end -
                      &_binary___deepseek_v41_batch_mla_metadata_gaudi2_o_start;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary___deepseek_v41_batch_mla_metadata_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
