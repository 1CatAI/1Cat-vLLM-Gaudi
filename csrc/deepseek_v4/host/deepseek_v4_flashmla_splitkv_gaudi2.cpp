/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_flashmla_splitkv_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_flashmla_splitkv_partial_tiled_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_flashmla_splitkv_partial_tiled_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_flashmla_splitkv_combine_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_flashmla_splitkv_combine_gaudi2_o_end;

namespace {

constexpr uint64_t kHeadDim = 512;
constexpr uint64_t kBytesPerToken = 584;

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

tpc_lib_api::GlueCodeReturn CopyElf(
    const unsigned char* start,
    const unsigned char* end,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    const unsigned isaSize = end - start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(outDefs->kernel.kernelElf, start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV4FlashMLASplitKVPartialFP8Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        "custom_deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4FlashMLASplitKVTiledPartialFP8Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        "custom_deepseek_v4_flashmla_splitkv_partial_tiled_fp8_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

namespace {

tpc_lib_api::GlueCodeReturn GetFlashMLAPartialGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs,
    const unsigned char* elfStart,
    const unsigned char* elfEnd)
{
    constexpr unsigned kInputCount = 13;
    constexpr unsigned kOutputCount = 2;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const tpc_lib_api::TensorDataType inputTypes[kInputCount] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
    };
    for (unsigned input = 0; input < kInputCount; ++input) {
        if (!HasDataType(inDefs->inputTensors[input], inputTypes[input])) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    for (unsigned output = 0; output < kOutputCount; ++output) {
        if (!HasDataType(
                inDefs->outputTensors[output], tpc_lib_api::DATA_F32)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }

    const auto& q = inDefs->inputTensors[0];
    const auto& compressedStorage = inDefs->inputTensors[1];
    const auto& compressedGeometry = inDefs->inputTensors[2];
    const auto& topkShape = inDefs->inputTensors[3];
    const auto& splitShape = inDefs->inputTensors[4];
    const auto& tokenToReq = inDefs->inputTensors[5];
    const auto& blockTable = inDefs->inputTensors[6];
    const auto& validToken = inDefs->inputTensors[7];
    const auto& seqLens = inDefs->inputTensors[8];
    const auto& swaStorage = inDefs->inputTensors[9];
    const auto& swaGeometry = inDefs->inputTensors[10];
    const auto& swaIndices = inDefs->inputTensors[11];
    const auto& swaLens = inDefs->inputTensors[12];
    auto& partialOutput = inDefs->outputTensors[0];
    auto& partialStats = inDefs->outputTensors[1];

    const bool ranksMatch =
        q.geometry.dims == 3 && compressedStorage.geometry.dims == 1 &&
        compressedGeometry.geometry.dims == 1 &&
        topkShape.geometry.dims == 2 && splitShape.geometry.dims == 1 &&
        tokenToReq.geometry.dims == 1 && blockTable.geometry.dims == 2 &&
        validToken.geometry.dims == 1 && seqLens.geometry.dims == 1 &&
        swaStorage.geometry.dims == 1 && swaGeometry.geometry.dims == 1 &&
        swaIndices.geometry.dims == 2 && swaLens.geometry.dims == 1 &&
        partialOutput.geometry.dims == 4 && partialStats.geometry.dims == 4;
    if (!ranksMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t headCount = q.geometry.maxSizes[1];
    const uint64_t batchSize = q.geometry.maxSizes[2];
    const uint64_t splitCount = splitShape.geometry.maxSizes[0];
    const bool inputsMatch =
        q.geometry.maxSizes[0] == kHeadDim && headCount > 0 &&
        batchSize > 0 && splitCount > 0 && splitCount <= 16 &&
        compressedStorage.geometry.maxSizes[0] >= kBytesPerToken &&
        compressedGeometry.geometry.maxSizes[0] == 4 &&
        topkShape.geometry.maxSizes[0] > 0 &&
        topkShape.geometry.maxSizes[1] == batchSize &&
        tokenToReq.geometry.maxSizes[0] == batchSize &&
        blockTable.geometry.maxSizes[0] > 0 &&
        blockTable.geometry.maxSizes[1] > 0 &&
        validToken.geometry.maxSizes[0] == batchSize &&
        seqLens.geometry.maxSizes[0] > 0 &&
        swaStorage.geometry.maxSizes[0] >= kBytesPerToken &&
        swaGeometry.geometry.maxSizes[0] == 4 &&
        swaIndices.geometry.maxSizes[0] > 0 &&
        swaIndices.geometry.maxSizes[1] == batchSize &&
        swaLens.geometry.maxSizes[0] == batchSize;
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputsMatch =
        partialOutput.geometry.maxSizes[0] == kHeadDim &&
        partialOutput.geometry.maxSizes[1] == headCount &&
        partialOutput.geometry.maxSizes[2] == splitCount &&
        partialOutput.geometry.maxSizes[3] == batchSize &&
        partialStats.geometry.maxSizes[0] == 2 &&
        partialStats.geometry.maxSizes[1] == headCount &&
        partialStats.geometry.maxSizes[2] == splitCount &&
        partialStats.geometry.maxSizes[3] == batchSize;
    if (!outputsMatch) {
        partialOutput.geometry.dataType = tpc_lib_api::DATA_F32;
        partialOutput.geometry.dims = 4;
        partialOutput.geometry.maxSizes[0] = kHeadDim;
        partialOutput.geometry.maxSizes[1] = headCount;
        partialOutput.geometry.maxSizes[2] = splitCount;
        partialOutput.geometry.maxSizes[3] = batchSize;
        partialStats.geometry.dataType = tpc_lib_api::DATA_F32;
        partialStats.geometry.dims = 4;
        partialStats.geometry.maxSizes[0] = 2;
        partialStats.geometry.maxSizes[1] = headCount;
        partialStats.geometry.maxSizes[2] = splitCount;
        partialStats.geometry.maxSizes[3] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 3;
    outDefs->indexSpaceGeometry[0] = splitCount;
    outDefs->indexSpaceGeometry[1] = (headCount + 1) / 2;
    outDefs->indexSpaceGeometry[2] = batchSize;

    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 0, 0, 511);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 1, 2, 0, 1);
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    const unsigned storageInputs[2] = {1, 9};
    for (unsigned input : storageInputs) {
        const auto size = inDefs->inputTensors[input].geometry.maxSizes[0];
        MapDimension(
            outDefs->inputTensorAccessPattern[input],
            0,
            0,
            0,
            0,
            static_cast<int>(size - 1));
    }
    const unsigned geometryInputs[2] = {2, 10};
    for (unsigned input : geometryInputs) {
        MapDimension(
            outDefs->inputTensorAccessPattern[input], 0, 0, 0, 0, 3);
    }
    MapDimension(
        outDefs->inputTensorAccessPattern[3],
        0,
        0,
        0,
        0,
        static_cast<int>(topkShape.geometry.maxSizes[0] - 1));
    MapDimension(outDefs->inputTensorAccessPattern[3], 1, 2, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[4],
        0,
        0,
        0,
        0,
        static_cast<int>(splitCount - 1));
    MapDimension(outDefs->inputTensorAccessPattern[5], 0, 2, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[6],
        0,
        0,
        0,
        0,
        static_cast<int>(blockTable.geometry.maxSizes[0] - 1));
    MapDimension(
        outDefs->inputTensorAccessPattern[6],
        1,
        0,
        0,
        0,
        static_cast<int>(blockTable.geometry.maxSizes[1] - 1));
    MapDimension(outDefs->inputTensorAccessPattern[7], 0, 2, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[8],
        0,
        0,
        0,
        0,
        static_cast<int>(seqLens.geometry.maxSizes[0] - 1));
    MapDimension(
        outDefs->inputTensorAccessPattern[11],
        0,
        0,
        0,
        0,
        static_cast<int>(swaIndices.geometry.maxSizes[0] - 1));
    MapDimension(outDefs->inputTensorAccessPattern[11], 1, 2, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[12], 0, 2, 1, 0, 0);

    for (unsigned output = 0; output < 2; ++output) {
        MapDimension(
            outDefs->outputTensorAccessPattern[output],
            0,
            0,
            0,
            0,
            output == 0 ? 511 : 1);
        MapDimension(
            outDefs->outputTensorAccessPattern[output], 1, 1, 2, 0, 1);
        MapDimension(
            outDefs->outputTensorAccessPattern[output], 2, 0, 1, 0, 0);
        MapDimension(
            outDefs->outputTensorAccessPattern[output], 3, 2, 1, 0, 0);
    }

    outDefs->kernel.paramsNr = 0;
    return CopyElf(elfStart, elfEnd, outDefs);
}

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV4FlashMLASplitKVPartialFP8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    return GetFlashMLAPartialGcDefinitions(
        inDefs,
        outDefs,
        &_binary___deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2_o_start,
        &_binary___deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn
DeepseekV4FlashMLASplitKVTiledPartialFP8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    return GetFlashMLAPartialGcDefinitions(
        inDefs,
        outDefs,
        &_binary___deepseek_v4_flashmla_splitkv_partial_tiled_fp8_gaudi2_o_start,
        &_binary___deepseek_v4_flashmla_splitkv_partial_tiled_fp8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn
DeepseekV4FlashMLASplitKVCombineGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName, "custom_deepseek_v4_flashmla_splitkv_combine_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4FlashMLASplitKVCombineGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 3;
    constexpr unsigned kOutputCount = 2;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const tpc_lib_api::TensorDataType inputTypes[kInputCount] = {
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_F32,
    };
    for (unsigned input = 0; input < kInputCount; ++input) {
        if (!HasDataType(inDefs->inputTensors[input], inputTypes[input])) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(
            inDefs->outputTensors[0], tpc_lib_api::DATA_BF16) ||
        !HasDataType(
            inDefs->outputTensors[1], tpc_lib_api::DATA_F32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& partialOutput = inDefs->inputTensors[0];
    const auto& partialStats = inDefs->inputTensors[1];
    const auto& sink = inDefs->inputTensors[2];
    auto& output = inDefs->outputTensors[0];
    auto& combineStats = inDefs->outputTensors[1];
    const bool ranksMatch =
        partialOutput.geometry.dims == 4 && partialStats.geometry.dims == 4 &&
        sink.geometry.dims == 1 && output.geometry.dims == 3 &&
        combineStats.geometry.dims == 2;
    if (!ranksMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t headCount = partialOutput.geometry.maxSizes[1];
    const uint64_t splitCount = partialOutput.geometry.maxSizes[2];
    const uint64_t batchSize = partialOutput.geometry.maxSizes[3];
    const bool inputsMatch =
        partialOutput.geometry.maxSizes[0] == kHeadDim && headCount > 0 &&
        splitCount > 0 && splitCount <= 16 && batchSize > 0 &&
        partialStats.geometry.maxSizes[0] == 2 &&
        partialStats.geometry.maxSizes[1] == headCount &&
        partialStats.geometry.maxSizes[2] == splitCount &&
        partialStats.geometry.maxSizes[3] == batchSize &&
        sink.geometry.maxSizes[0] >= headCount;
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (output.geometry.maxSizes[0] != kHeadDim ||
        output.geometry.maxSizes[1] < headCount ||
        output.geometry.maxSizes[2] != batchSize ||
        combineStats.geometry.maxSizes[0] != headCount ||
        combineStats.geometry.maxSizes[1] != batchSize) {
        output.geometry.dataType = tpc_lib_api::DATA_BF16;
        output.geometry.dims = 3;
        output.geometry.maxSizes[0] = kHeadDim;
        output.geometry.maxSizes[1] = headCount;
        output.geometry.maxSizes[2] = batchSize;
        combineStats.geometry.dataType = tpc_lib_api::DATA_F32;
        combineStats.geometry.dims = 2;
        combineStats.geometry.maxSizes[0] = headCount;
        combineStats.geometry.maxSizes[1] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 2;
    outDefs->indexSpaceGeometry[0] = headCount;
    outDefs->indexSpaceGeometry[1] = batchSize;
    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 0, 0, 511);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        2,
        0,
        0,
        0,
        static_cast<int>(splitCount - 1));
    MapDimension(outDefs->inputTensorAccessPattern[0], 3, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, 1);
    MapDimension(outDefs->inputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[1],
        2,
        0,
        0,
        0,
        static_cast<int>(splitCount - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 3, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[2], 0, 0, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 0, 0, 511);
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 2, 1, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[1], 0, 0, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[1], 1, 1, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    return CopyElf(
        &_binary___deepseek_v4_flashmla_splitkv_combine_gaudi2_o_start,
        &_binary___deepseek_v4_flashmla_splitkv_combine_gaudi2_o_end,
        outDefs);
}
