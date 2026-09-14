/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>
#include <limits>

#include "deepseek_v41_engram_hash_gather_bf16_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v41_engram_hash_gather_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_engram_hash_gather_bf16_gaudi2_o_end;

namespace {

constexpr uint64_t kHeads = 12;
constexpr uint64_t kHistory = 3;
constexpr uint64_t kParameters = 47;
constexpr uint64_t kWeightWidth = 256;
constexpr uint64_t kScaleWidth = 8;

void MapDimension(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensorDim,
    unsigned indexDim,
    int coefficient,
    int start,
    int end)
{
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = coefficient;
    pattern.mapping[tensorDim].start_b = start;
    pattern.mapping[tensorDim].end_b = end;
}

bool HasDataType(
    tpc_lib_api::Tensor& tensor,
    tpc_lib_api::TensorDataType expected)
{
    if (tensor.geometry.dataType == expected) {
        return true;
    }
    tensor.geometry.dataType = expected;
    return false;
}

bool HasShape2(
    const tpc_lib_api::Tensor& tensor,
    uint64_t dim0,
    uint64_t dim1)
{
    return tensor.geometry.dims == 2 &&
        tensor.geometry.maxSizes[0] == dim0 &&
        tensor.geometry.maxSizes[1] == dim1;
}

void SetShape2(
    tpc_lib_api::Tensor& tensor,
    uint64_t dim0,
    uint64_t dim1)
{
    tensor.geometry.dims = 2;
    tensor.geometry.maxSizes[0] = dim0;
    tensor.geometry.maxSizes[1] = dim1;
}

void MapAll(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensorDim,
    int end)
{
    MapDimension(pattern, tensorDim, 0, 0, 0, end);
}

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV41EngramHashGatherBf16Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(kernelName, name);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV41EngramHashGatherBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 6;
    constexpr unsigned kOutputCount = 2;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned index = 0; index < 4; ++index) {
        if (!HasDataType(
                inDefs->inputTensors[index], tpc_lib_api::DATA_I32)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    for (unsigned index = 4; index < kInputCount; ++index) {
        if (!HasDataType(
                inDefs->inputTensors[index], tpc_lib_api::DATA_U8)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_BF16) ||
        !HasDataType(inDefs->outputTensors[1], tpc_lib_api::DATA_I32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& tokenMap = inDefs->inputTensors[2];
    const auto& weights = inDefs->inputTensors[4];
    if (!HasShape2(inDefs->inputTensors[0], 1, 1) ||
        !HasShape2(inDefs->inputTensors[1], kHistory, 1) ||
        tokenMap.geometry.dims != 2 || tokenMap.geometry.maxSizes[0] == 0 ||
        tokenMap.geometry.maxSizes[0] >
            static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        tokenMap.geometry.maxSizes[1] != 1 ||
        !HasShape2(inDefs->inputTensors[3], kParameters, 1) ||
        weights.geometry.dims != 2 ||
        weights.geometry.maxSizes[0] != kWeightWidth ||
        weights.geometry.maxSizes[1] == 0 ||
        weights.geometry.maxSizes[1] >
            static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        !HasShape2(
            inDefs->inputTensors[5],
            kScaleWidth,
            weights.geometry.maxSizes[1])) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!HasShape2(inDefs->outputTensors[0], kWeightWidth, kHeads) ||
        !HasShape2(inDefs->outputTensors[1], kHistory, 1)) {
        SetShape2(inDefs->outputTensors[0], kWeightWidth, kHeads);
        SetShape2(inDefs->outputTensors[1], kHistory, 1);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = kHeads;
    MapAll(outDefs->inputTensorAccessPattern[0], 0, 0);
    MapAll(outDefs->inputTensorAccessPattern[0], 1, 0);
    MapAll(outDefs->inputTensorAccessPattern[1], 0, kHistory - 1);
    MapAll(outDefs->inputTensorAccessPattern[1], 1, 0);
    MapAll(
        outDefs->inputTensorAccessPattern[2],
        0,
        static_cast<int>(tokenMap.geometry.maxSizes[0] - 1));
    MapAll(outDefs->inputTensorAccessPattern[2], 1, 0);
    MapAll(outDefs->inputTensorAccessPattern[3], 0, kParameters - 1);
    MapAll(outDefs->inputTensorAccessPattern[3], 1, 0);
    const int lastRow = static_cast<int>(weights.geometry.maxSizes[1] - 1);
    MapAll(outDefs->inputTensorAccessPattern[4], 0, kWeightWidth - 1);
    MapAll(outDefs->inputTensorAccessPattern[4], 1, lastRow);
    MapAll(outDefs->inputTensorAccessPattern[5], 0, kScaleWidth - 1);
    MapAll(outDefs->inputTensorAccessPattern[5], 1, lastRow);
    MapDimension(
        outDefs->outputTensorAccessPattern[0],
        0,
        0,
        0,
        0,
        kWeightWidth - 1);
    MapDimension(
        outDefs->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    MapAll(outDefs->outputTensorAccessPattern[1], 0, kHistory - 1);
    MapAll(outDefs->outputTensorAccessPattern[1], 1, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary___deepseek_v41_engram_hash_gather_bf16_gaudi2_o_end -
        &_binary___deepseek_v41_engram_hash_gather_bf16_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary___deepseek_v41_engram_hash_gather_bf16_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
