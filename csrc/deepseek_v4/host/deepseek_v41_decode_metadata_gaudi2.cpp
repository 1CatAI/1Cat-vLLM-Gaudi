// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_decode_metadata_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_decode_metadata_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_decode_metadata_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41DecodeMetadataGaudi2::GetKernelName(char out[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(out, name);
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41DecodeMetadataGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 4) { in->outputTensorNr = 4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    for (unsigned i = 0; i < 3; ++i)
        if (in->inputTensors[i].geometry.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (in->outputTensors[0].geometry.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& pos = in->inputTensors[0].geometry;
    const auto& ids = in->inputTensors[1].geometry;
    const auto& pages = in->inputTensors[2].geometry;
    const auto tokens = pos.maxSizes[0];
    if (pos.dims != 1 || !tokens || tokens > 6 || ids.dims != 1 || ids.maxSizes[0] != tokens ||
        pages.dims != 1 || !pages.maxSizes[0]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& result = in->outputTensors[0].geometry;
    if (result.dims != 2 || result.maxSizes[0] != 192 || result.maxSizes[1] != tokens)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    for (unsigned i = 1; i < 4; ++i) {
        const auto& field = in->outputTensors[i].geometry;
        if (field.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (field.dims != 1 || field.maxSizes[0] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->outputTensorAccessPattern[i].mapping[0] = {1, 1, 0, 0, false};
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 1;
    out->indexSpaceGeometry[1] = tokens;
    for (unsigned i = 0; i < 2; ++i) out->inputTensorAccessPattern[i].mapping[0] = {1, 1, 0, 0, false};
    out->inputTensorAccessPattern[2].allRequired = true;
    out->outputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 191, false};
    out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0, false};
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___deepseek_v41_decode_metadata_gaudi2_o_start;
    const auto* end = &_binary___deepseek_v41_decode_metadata_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
