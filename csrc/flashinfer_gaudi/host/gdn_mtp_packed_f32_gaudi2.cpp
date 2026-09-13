// SPDX-License-Identifier: Apache-2.0
/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "gdn_mtp_packed_f32_gaudi2.hpp"

extern unsigned char _binary_gdn_mtp_packed_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_packed_f32_gaudi2_o_end;

namespace {

constexpr uint64_t kKeyDim = 128;
constexpr uint64_t kValueDim = 128;
constexpr uint64_t kKeyHeads = 16;
constexpr uint64_t kValueHeads = 48;
constexpr uint64_t kTokens = 8;
constexpr uint64_t kPackedWidth =
    (2 * kKeyHeads + kValueHeads) * kKeyDim;

void MapDimension(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensorDim,
    unsigned indexDim,
    int a,
    int startB,
    int endB)
{
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = a;
    pattern.mapping[tensorDim].start_b = startB;
    pattern.mapping[tensorDim].end_b = endB;
}

bool HasTypeAndRank(
    const tpc_lib_api::Tensor& tensor,
    tpc_lib_api::TensorDataType type,
    unsigned dims)
{
    return tensor.geometry.dataType == type && tensor.geometry.dims == dims;
}

}  // namespace

tpc_lib_api::GlueCodeReturn GdnMtpPackedF32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(kernelName, "flashinfer_gaudi_gdn_mtp_packed_bf16_f32state_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnMtpPackedF32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 7;
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const auto& state = inDefs->inputTensors[0];
    const auto& packed = inDefs->inputTensors[1];
    const auto& decay = inDefs->inputTensors[2];
    const auto& beta = inDefs->inputTensors[3];
    const auto& indices = inDefs->inputTensors[4];
    const auto& accepted = inDefs->inputTensors[5];
    const auto& queryLengths = inDefs->inputTensors[6];
    auto& output = inDefs->outputTensors[0];

    const bool inputsValid =
        HasTypeAndRank(state, tpc_lib_api::DATA_F32, 4) &&
        HasTypeAndRank(packed, tpc_lib_api::DATA_BF16, 3) &&
        HasTypeAndRank(decay, tpc_lib_api::DATA_F32, 3) &&
        HasTypeAndRank(beta, tpc_lib_api::DATA_BF16, 3) &&
        HasTypeAndRank(indices, tpc_lib_api::DATA_I32, 2) &&
        HasTypeAndRank(accepted, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(queryLengths, tpc_lib_api::DATA_I32, 1);
    if (!inputsValid) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t keyDim = state.geometry.maxSizes[0];
    const uint64_t valueDim = state.geometry.maxSizes[1];
    const uint64_t valueHeads = state.geometry.maxSizes[2];
    const uint64_t stateSlots = state.geometry.maxSizes[3];
    const uint64_t packedWidth = packed.geometry.maxSizes[0];
    const uint64_t numTokens = packed.geometry.maxSizes[1];
    const uint64_t batchSize = packed.geometry.maxSizes[2];
    if (keyDim != kKeyDim || valueDim != kValueDim || valueHeads != kValueHeads ||
        stateSlots == 0 || batchSize == 0 || batchSize > 16 || numTokens != kTokens ||
        packedWidth != kPackedWidth) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t keyHeads = (packedWidth - valueHeads * valueDim) / (2 * keyDim);
    const bool shapesMatch =
        keyHeads == kKeyHeads && valueHeads % keyHeads == 0 &&
        packedWidth == 2 * keyHeads * keyDim + valueHeads * valueDim &&
        decay.geometry.maxSizes[0] == valueHeads && decay.geometry.maxSizes[1] == numTokens &&
        decay.geometry.maxSizes[2] == batchSize &&
        beta.geometry.maxSizes[0] == valueHeads && beta.geometry.maxSizes[1] == numTokens &&
        beta.geometry.maxSizes[2] == batchSize &&
        indices.geometry.maxSizes[0] == numTokens && indices.geometry.maxSizes[1] == batchSize &&
        accepted.geometry.maxSizes[0] == batchSize && queryLengths.geometry.maxSizes[0] == batchSize;
    if (!shapesMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches =
        HasTypeAndRank(output, tpc_lib_api::DATA_BF16, 4) &&
        output.geometry.maxSizes[0] == valueDim && output.geometry.maxSizes[1] == valueHeads &&
        output.geometry.maxSizes[2] == numTokens && output.geometry.maxSizes[3] == batchSize;
    if (!outputMatches) {
        output.geometry.dataType = tpc_lib_api::DATA_BF16;
        output.geometry.dims = 4;
        output.geometry.maxSizes[0] = valueDim;
        output.geometry.maxSizes[1] = valueHeads;
        output.geometry.maxSizes[2] = numTokens;
        output.geometry.maxSizes[3] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 4;
    outDefs->indexSpaceGeometry[0] = 1;
    outDefs->indexSpaceGeometry[1] = 2;
    outDefs->indexSpaceGeometry[2] = valueHeads;
    outDefs->indexSpaceGeometry[3] = batchSize;

    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 1, 64, 0, 63);
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[0], 3, 3, 0, 0, static_cast<int>(stateSlots - 1));

    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 2, 0, 0, static_cast<int>(packedWidth - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 1, 0, 0, 0, static_cast<int>(numTokens - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 2, 3, 1, 0, 0);
    for (unsigned input = 2; input <= 3; ++input) {
        MapDimension(outDefs->inputTensorAccessPattern[input], 0, 2, 1, 0, 0);
        MapDimension(outDefs->inputTensorAccessPattern[input], 1, 0, 0, 0, static_cast<int>(numTokens - 1));
        MapDimension(outDefs->inputTensorAccessPattern[input], 2, 3, 1, 0, 0);
    }
    MapDimension(outDefs->inputTensorAccessPattern[4], 0, 0, 0, 0, static_cast<int>(numTokens - 1));
    MapDimension(outDefs->inputTensorAccessPattern[4], 1, 3, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[5], 0, 3, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[6], 0, 3, 1, 0, 0);

    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 1, 64, 0, 63);
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 2, 0, 0, 0, static_cast<int>(numTokens - 1));
    MapDimension(outDefs->outputTensorAccessPattern[0], 3, 3, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary_gdn_mtp_packed_f32_gaudi2_o_end - &_binary_gdn_mtp_packed_f32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(outDefs->kernel.kernelElf, &_binary_gdn_mtp_packed_f32_gaudi2_o_start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
