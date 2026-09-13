// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include "gdn_mtp_prepared_f32_gaudi2.hpp"

extern unsigned char _binary_gdn_mtp_prepared_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_prepared_f32_gaudi2_o_end;

namespace {
#ifndef MTP_PREPARED_ROWS
#define MTP_PREPARED_ROWS 4
#endif
constexpr unsigned kRows = MTP_PREPARED_ROWS;
void Map(tpc_lib_api::TensorAccessPattern& pattern, unsigned dim, unsigned indexDim,
         int a, int first, int last)
{
    pattern.mapping[dim].indexSpaceDim = indexDim;
    pattern.mapping[dim].a = a;
    pattern.mapping[dim].start_b = first;
    pattern.mapping[dim].end_b = last;
}
bool Shape(const tpc_lib_api::Tensor& tensor, unsigned rank)
{
    return tensor.geometry.dataType == tpc_lib_api::DATA_F32 && tensor.geometry.dims == rank;
}
}

tpc_lib_api::GlueCodeReturn GdnMtpPreparedF32Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(name, "flashinfer_gaudi_gdn_mtp_prepared_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnMtpPreparedF32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 4) {
        in->inputTensorNr = 4;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto& state = in->inputTensors[0];
    const auto& packed = in->inputTensors[1];
    const uint64_t batch = state.geometry.maxSizes[3];
    if (!Shape(state, 4) || !Shape(packed, 3) ||
        !Shape(in->inputTensors[2], 3) || !Shape(in->inputTensors[3], 3)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (batch == 0 || batch > 16 || state.geometry.maxSizes[0] != 128 ||
        state.geometry.maxSizes[1] != 128 || state.geometry.maxSizes[2] != 48 ||
        packed.geometry.maxSizes[0] != 10240 || packed.geometry.maxSizes[1] != 8 ||
        packed.geometry.maxSizes[2] != batch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (unsigned i = 2; i < 4; ++i) {
        const auto& gate = in->inputTensors[i].geometry;
        if (gate.maxSizes[0] != 48 || gate.maxSizes[1] != 8 || gate.maxSizes[2] != batch)
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const uint64_t outputSizes[2][5] = {{128, 48, 8, batch, 0}, {128, 128, 48, 8, batch}};
    bool wrongOutput = false;
    for (unsigned i = 0; i < 2; ++i) {
        auto& geometry = in->outputTensors[i].geometry;
        wrongOutput |= !Shape(in->outputTensors[i], 4 + i);
        geometry.dataType = tpc_lib_api::DATA_F32;
        geometry.dims = 4 + i;
        for (unsigned d = 0; d < geometry.dims; ++d) {
            wrongOutput |= geometry.maxSizes[d] != outputSizes[i][d];
            geometry.maxSizes[d] = outputSizes[i][d];
        }
    }
    if (wrongOutput) return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;

    out->indexSpaceRank = 4;
    out->indexSpaceGeometry[0] = 1;
    out->indexSpaceGeometry[1] = 128 / kRows;
    out->indexSpaceGeometry[2] = 48;
    out->indexSpaceGeometry[3] = batch;
    Map(out->inputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    Map(out->inputTensorAccessPattern[0], 1, 1, kRows, 0, kRows - 1);
    Map(out->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    Map(out->inputTensorAccessPattern[0], 3, 3, 1, 0, 0);
    Map(out->inputTensorAccessPattern[1], 0, 2, 0, 0, 10239);
    Map(out->inputTensorAccessPattern[1], 1, 0, 0, 0, 7);
    Map(out->inputTensorAccessPattern[1], 2, 3, 1, 0, 0);
    for (unsigned i = 2; i < 4; ++i) {
        Map(out->inputTensorAccessPattern[i], 0, 2, 1, 0, 63);
        Map(out->inputTensorAccessPattern[i], 1, 0, 0, 0, 7);
        Map(out->inputTensorAccessPattern[i], 2, 3, 1, 0, 0);
    }
    Map(out->outputTensorAccessPattern[0], 0, 1, kRows, 0, kRows - 1);
    Map(out->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[0], 2, 0, 0, 0, 7);
    Map(out->outputTensorAccessPattern[0], 3, 3, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    Map(out->outputTensorAccessPattern[1], 1, 1, kRows, 0, kRows - 1);
    Map(out->outputTensorAccessPattern[1], 2, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 3, 0, 0, 0, 7);
    Map(out->outputTensorAccessPattern[1], 4, 3, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const unsigned size = &_binary_gdn_mtp_prepared_f32_gaudi2_o_end -
                          &_binary_gdn_mtp_prepared_f32_gaudi2_o_start;
    const unsigned provided = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (provided < size) return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary_gdn_mtp_prepared_f32_gaudi2_o_start, size);
    return tpc_lib_api::GLUE_SUCCESS;
}
