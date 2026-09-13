/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_save_partial_states_f32_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_save_partial_states_f32_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_save_partial_states_f32_gaudi2_o_end;

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

tpc_lib_api::GlueCodeReturn
DeepseekV4SavePartialStatesF32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName, "custom_deepseek_v4_save_partial_states_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4SavePartialStatesF32Gaudi2::GetGcDefinitions(
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

    const auto& stateStorage = inDefs->inputTensors[0];
    const auto& stateGeometry = inDefs->inputTensors[1];
    const auto& kv = inDefs->inputTensors[2];
    const auto& score = inDefs->inputTensors[3];
    const auto& ape = inDefs->inputTensors[4];
    const auto& positions = inDefs->inputTensors[5];
    const auto& slotMapping = inDefs->inputTensors[6];
    auto& completion = inDefs->outputTensors[0];

    const bool typesAndRanks =
        HasTypeAndRank(stateStorage, tpc_lib_api::DATA_F32, 1) &&
        HasTypeAndRank(stateGeometry, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(kv, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(score, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(ape, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(positions, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(slotMapping, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(completion, tpc_lib_api::DATA_F32, 2);
    if (!typesAndRanks) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t storageElements = stateStorage.geometry.maxSizes[0];
    const uint64_t stateWidth = kv.geometry.maxSizes[0];
    const uint64_t kvTokens = kv.geometry.maxSizes[1];
    const uint64_t numTokens = slotMapping.geometry.maxSizes[0];
    const uint64_t compressRatio = ape.geometry.maxSizes[1];
    const uint64_t chunks = stateWidth / 64;
    const bool inputsMatch =
        storageElements > 0 && stateGeometry.geometry.maxSizes[0] == 5 &&
        stateWidth >= 64 && stateWidth <= 1024 && stateWidth % 64 == 0 &&
        numTokens > 0 && kvTokens >= numTokens &&
        score.geometry.maxSizes[0] == stateWidth &&
        score.geometry.maxSizes[1] >= numTokens &&
        ape.geometry.maxSizes[0] == stateWidth &&
        (compressRatio == 4 || compressRatio == 128) &&
        positions.geometry.maxSizes[0] >= numTokens;
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches =
        completion.geometry.maxSizes[0] == chunks &&
        completion.geometry.maxSizes[1] == numTokens;
    if (!outputMatches) {
        completion.geometry.dataType = tpc_lib_api::DATA_F32;
        completion.geometry.dims = 2;
        completion.geometry.maxSizes[0] = chunks;
        completion.geometry.maxSizes[1] = numTokens;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 2;
    outDefs->indexSpaceGeometry[0] = chunks;
    outDefs->indexSpaceGeometry[1] = numTokens;

    // The state slot is selected indirectly; conservatively expose the whole
    // backing storage to the graph compiler.
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        0,
        0,
        0,
        0,
        static_cast<int>(storageElements - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, 4);
    for (unsigned input = 2; input <= 3; ++input) {
        MapDimension(
            outDefs->inputTensorAccessPattern[input], 0, 0, 64, 0, 63);
        MapDimension(
            outDefs->inputTensorAccessPattern[input], 1, 1, 1, 0, 0);
    }
    MapDimension(outDefs->inputTensorAccessPattern[4], 0, 0, 64, 0, 63);
    MapDimension(
        outDefs->inputTensorAccessPattern[4],
        1,
        1,
        0,
        0,
        static_cast<int>(compressRatio - 1));
    MapDimension(outDefs->inputTensorAccessPattern[5], 0, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[6], 0, 1, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary___deepseek_v4_save_partial_states_f32_gaudi2_o_end -
        &_binary___deepseek_v4_save_partial_states_f32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary___deepseek_v4_save_partial_states_f32_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
