// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_candidate_coordinates_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_candidate_coordinates_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_candidate_coordinates_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41CandidateCoordinatesGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (p->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(int)) return GLUE_FAILED;
    const int maximum = *static_cast<const int*>(p->nodeParams.nodeParams);
    if (maximum < 0) return GLUE_FAILED;
    const auto& blocks = p->inputTensors[0].geometry;
    if (blocks.dataType != DATA_I32 || blocks.dims != 2 || !blocks.maxSizes[0] ||
        blocks.maxSizes[0] > 2048 || !blocks.maxSizes[1] || blocks.maxSizes[1] > 6)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i = 0; i < 2; ++i) {
        const auto& value = p->outputTensors[i].geometry;
        if (value.dataType != DATA_I32 || value.dims != 2 || value.maxSizes[0] != blocks.maxSizes[0] * 8 ||
            value.maxSizes[1] != blocks.maxSizes[1]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (whole_) {
            // Tiny metadata is intentionally materialized once. Its consumers
            // retain their own independently sliced gather/MME SRAM bundles.
            out->outputTensorAccessPattern[i].allRequired = true;
        } else {
            out->outputTensorAccessPattern[i].mapping[0] = {0, 64, 0, 63};
            out->outputTensorAccessPattern[i].mapping[1] = {1, 1, 0, 0};
        }
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (blocks.maxSizes[0] + 7) / 8;
    out->indexSpaceGeometry[1] = blocks.maxSizes[1];
    out->inputTensorAccessPattern[0].mapping[0] = {0, 8, 0, 7};
    out->inputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
    const int scalars[] = {maximum, int(blocks.maxSizes[0])};
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, scalars, sizeof(scalars));
    const auto* first = &_binary___deepseek_v41_candidate_coordinates_gaudi2_o_start;
    const auto* last = &_binary___deepseek_v41_candidate_coordinates_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
