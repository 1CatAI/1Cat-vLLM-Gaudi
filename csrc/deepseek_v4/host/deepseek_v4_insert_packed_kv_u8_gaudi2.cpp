/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_insert_packed_kv_u8_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_insert_packed_kv_u8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_insert_packed_kv_u8_gaudi2_o_end;

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
    return tensor.geometry.dataType == type &&
        tensor.geometry.dims == dims;
}

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV4InsertPackedKvU8Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        "custom_deepseek_v4_insert_packed_kv_u8_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4InsertPackedKvU8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 4;
    constexpr unsigned kOutputCount = 1;
    constexpr uint64_t kPayloadBytes = 584;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const auto& cacheStorage = inDefs->inputTensors[0];
    const auto& cacheGeometry = inDefs->inputTensors[1];
    const auto& packed = inDefs->inputTensors[2];
    const auto& slots = inDefs->inputTensors[3];
    auto& output = inDefs->outputTensors[0];
    const bool typesAndRanks =
        HasTypeAndRank(cacheStorage, tpc_lib_api::DATA_U8, 1) &&
        HasTypeAndRank(cacheGeometry, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(packed, tpc_lib_api::DATA_U8, 2) &&
        HasTypeAndRank(slots, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(output, tpc_lib_api::DATA_U8, 1);
    if (!typesAndRanks) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t numTokens = packed.geometry.maxSizes[1];
    const uint64_t storageBytes = cacheStorage.geometry.maxSizes[0];
    const bool inputsMatch =
        storageBytes >= kPayloadBytes &&
        cacheGeometry.geometry.maxSizes[0] == 4 &&
        packed.geometry.maxSizes[0] == kPayloadBytes &&
        numTokens > 0 && slots.geometry.maxSizes[0] >= numTokens;
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    if (output.geometry.maxSizes[0] != numTokens) {
        output.geometry.dataType = tpc_lib_api::DATA_U8;
        output.geometry.dims = 1;
        output.geometry.maxSizes[0] = numTokens;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = numTokens;
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        0,
        0,
        0,
        0,
        static_cast<int>(storageBytes - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, 3);
    MapDimension(
        outDefs->inputTensorAccessPattern[2],
        0,
        0,
        0,
        0,
        static_cast<int>(kPayloadBytes - 1));
    MapDimension(outDefs->inputTensorAccessPattern[2], 1, 0, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[3], 0, 0, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary___deepseek_v4_insert_packed_kv_u8_gaudi2_o_end -
        &_binary___deepseek_v4_insert_packed_kv_u8_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary___deepseek_v4_insert_packed_kv_u8_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
