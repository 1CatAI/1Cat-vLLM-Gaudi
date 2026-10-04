// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_ordered_peer_sum_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_ordered_peer_sum_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_ordered_peer_sum_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41OrderedPeerSumGaudi2::GetKernelName(char out[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(out, name);
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41OrderedPeerSumGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (in->inputTensors[0].geometry.dataType != DATA_BF16 ||
        in->outputTensors[0].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& shards = in->inputTensors[0].geometry;
    const auto& result = in->outputTensors[0].geometry;
    const auto width = shards.maxSizes[0];
    if (shards.dims != 2 || !width || width % 128 || width > 32768 ||
        shards.maxSizes[1] < 2 || shards.maxSizes[1] > 8 || result.dims != 2 ||
        result.maxSizes[0] != width || result.maxSizes[1] != 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = width / 128;
    out->inputTensorAccessPattern[0].mapping[0] = {0, 128, 0, 127, false};
    out->inputTensorAccessPattern[0].mapping[1] = {0, 0, 0, float(shards.maxSizes[1] - 1), false};
    out->outputTensorAccessPattern[0].mapping[0] = {0, 128, 0, 127, false};
    out->outputTensorAccessPattern[0].mapping[1] = {0, 0, 0, 0, false};
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___deepseek_v41_ordered_peer_sum_gaudi2_o_start;
    const auto* end = &_binary___deepseek_v41_ordered_peer_sum_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
