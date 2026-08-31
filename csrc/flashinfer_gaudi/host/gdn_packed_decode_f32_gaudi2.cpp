/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "gdn_packed_decode_f32_gaudi2.hpp"

extern unsigned char _binary_gdn_packed_decode_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_packed_decode_f32_gaudi2_o_end;

namespace {

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

tpc_lib_api::GlueCodeReturn GdnPackedDecodeF32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(kernelName, "flashinfer_gaudi_gdn_decode_packed_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnPackedDecodeF32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 5;
    constexpr unsigned kOutputCount = 2;
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
    auto& newState = inDefs->outputTensors[0];
    auto& output = inDefs->outputTensors[1];

    const bool inputsValid =
        HasTypeAndRank(state, tpc_lib_api::DATA_F32, 4) &&
        HasTypeAndRank(packed, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(decay, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(beta, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(indices, tpc_lib_api::DATA_I32, 1);
    if (!inputsValid) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t keyDim = state.geometry.maxSizes[0];
    const uint64_t valueDim = state.geometry.maxSizes[1];
    const uint64_t valueHeads = state.geometry.maxSizes[2];
    const uint64_t stateSlots = state.geometry.maxSizes[3];
    const uint64_t batchSize = packed.geometry.maxSizes[1];
    const uint64_t packedWidth = packed.geometry.maxSizes[0];
    if (keyDim != 128 || valueDim != 128 || valueHeads == 0 || stateSlots == 0 || batchSize == 0 ||
        packedWidth <= valueHeads * valueDim) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t keyHeads = (packedWidth - valueHeads * valueDim) / (2 * keyDim);
    const bool shapesMatch =
        keyHeads > 0 && valueHeads % keyHeads == 0 &&
        packedWidth == 2 * keyHeads * keyDim + valueHeads * valueDim &&
        decay.geometry.maxSizes[0] == valueHeads && decay.geometry.maxSizes[1] == batchSize &&
        beta.geometry.maxSizes[0] == valueHeads && beta.geometry.maxSizes[1] == batchSize &&
        indices.geometry.maxSizes[0] == batchSize;
    if (!shapesMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool stateOutputMatches =
        HasTypeAndRank(newState, tpc_lib_api::DATA_F32, 4) &&
        newState.geometry.maxSizes[0] == keyDim && newState.geometry.maxSizes[1] == valueDim &&
        newState.geometry.maxSizes[2] == valueHeads && newState.geometry.maxSizes[3] == batchSize;
    if (!stateOutputMatches) {
        newState.geometry.dataType = tpc_lib_api::DATA_F32;
        newState.geometry.dims = 4;
        newState.geometry.maxSizes[0] = keyDim;
        newState.geometry.maxSizes[1] = valueDim;
        newState.geometry.maxSizes[2] = valueHeads;
        newState.geometry.maxSizes[3] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    const bool outputMatches =
        HasTypeAndRank(output, tpc_lib_api::DATA_F32, 3) &&
        output.geometry.maxSizes[0] == valueDim && output.geometry.maxSizes[1] == valueHeads &&
        output.geometry.maxSizes[2] == batchSize;
    if (!outputMatches) {
        output.geometry.dataType = tpc_lib_api::DATA_F32;
        output.geometry.dims = 3;
        output.geometry.maxSizes[0] = valueDim;
        output.geometry.maxSizes[1] = valueHeads;
        output.geometry.maxSizes[2] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 4;
    outDefs->indexSpaceGeometry[0] = 1;
    outDefs->indexSpaceGeometry[1] = 4;
    outDefs->indexSpaceGeometry[2] = valueHeads;
    outDefs->indexSpaceGeometry[3] = batchSize;

    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 1, 32, 0, 31);
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[0], 3, 3, 0, 0, static_cast<int>(stateSlots - 1));

    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 2, 0, 0, static_cast<int>(packedWidth - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 1, 3, 1, 0, 0);
    for (unsigned input = 2; input <= 3; ++input) {
        MapDimension(outDefs->inputTensorAccessPattern[input], 0, 2, 1, 0, 0);
        MapDimension(outDefs->inputTensorAccessPattern[input], 1, 3, 1, 0, 0);
    }
    MapDimension(outDefs->inputTensorAccessPattern[4], 0, 3, 1, 0, 0);

    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 1, 32, 0, 31);
    MapDimension(outDefs->outputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 3, 3, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[1], 0, 1, 32, 0, 31);
    MapDimension(outDefs->outputTensorAccessPattern[1], 1, 2, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[1], 2, 3, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary_gdn_packed_decode_f32_gaudi2_o_end - &_binary_gdn_packed_decode_f32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(outDefs->kernel.kernelElf, &_binary_gdn_packed_decode_f32_gaudi2_o_start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}

