/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_save_compress_norm_c4_f32_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_f32_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_f32_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_bf16_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_mixed_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_save_compress_norm_c4_mixed_gaudi2_o_end;

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
DeepseekV4SaveCompressNormC4F32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    if (constantsType_ == ORDERED_F32) {
        std::strcpy(
            kernelName,
            "custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2");
    } else if (constantsType_ == BF16_CONSTANTS) {
        std::strcpy(
            kernelName,
            "custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2");
    } else if (constantsType_ == MIXED_CONSTANTS) {
        std::strcpy(
            kernelName,
            "custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2");
    } else {
        std::strcpy(
            kernelName,
            "custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2");
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4SaveCompressNormC4F32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    const bool ordered = constantsType_ == ORDERED_F32;
    const unsigned kInputCount = ordered ? 15 : 14;
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const auto& storage = inDefs->inputTensors[0];
    const auto& stateGeometry = inDefs->inputTensors[1];
    const auto& cacheGeometry = inDefs->inputTensors[2];
    const auto& kv = inDefs->inputTensors[3];
    const auto& score = inDefs->inputTensors[4];
    const auto& ape = inDefs->inputTensors[5];
    const auto& positions = inDefs->inputTensors[6];
    const auto& slotMapping = inDefs->inputTensors[7];
    const auto& tokenToReq = inDefs->inputTensors[8];
    const auto& blockTable = inDefs->inputTensors[9];
    const auto& rmsWeight = inDefs->inputTensors[10];
    const auto& rmsEps = inDefs->inputTensors[11];
    const auto& cosSinCache = inDefs->inputTensors[12];
    const auto& kvSlotMapping = inDefs->inputTensors[13];
    const auto* completionInput = ordered
        ? &inDefs->inputTensors[14]
        : nullptr;
    auto& mutationToken = inDefs->outputTensors[0];

    const bool usesBF16Inputs = constantsType_ == BF16_CONSTANTS;
    const bool usesBF16Constants =
        constantsType_ == BF16_CONSTANTS ||
        constantsType_ == MIXED_CONSTANTS;

    const bool typesAndRanks =
        HasTypeAndRank(storage, tpc_lib_api::DATA_U8, 1) &&
        HasTypeAndRank(stateGeometry, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(cacheGeometry, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(
            kv,
            usesBF16Inputs
                ? tpc_lib_api::DATA_BF16
                : tpc_lib_api::DATA_F32,
            2) &&
        HasTypeAndRank(
            score,
            usesBF16Inputs
                ? tpc_lib_api::DATA_BF16
                : tpc_lib_api::DATA_F32,
            2) &&
        HasTypeAndRank(ape, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(positions, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(slotMapping, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(tokenToReq, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(blockTable, tpc_lib_api::DATA_I32, 2) &&
        HasTypeAndRank(
            rmsWeight,
            usesBF16Constants
                ? tpc_lib_api::DATA_BF16
                : tpc_lib_api::DATA_F32,
            1) &&
        HasTypeAndRank(rmsEps, tpc_lib_api::DATA_F32, 1) &&
        HasTypeAndRank(cosSinCache, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(kvSlotMapping, tpc_lib_api::DATA_I32, 1) &&
        (!ordered || HasTypeAndRank(
            *completionInput, tpc_lib_api::DATA_I32, 1)) &&
        HasTypeAndRank(
            mutationToken,
            ordered ? tpc_lib_api::DATA_I32 : tpc_lib_api::DATA_U8,
            1);
    if (!typesAndRanks) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t storageBytes = storage.geometry.maxSizes[0];
    const uint64_t stateWidth = kv.geometry.maxSizes[0];
    const uint64_t headDim = rmsWeight.geometry.maxSizes[0];
    const uint64_t compressRatio = ape.geometry.maxSizes[1];
    const uint64_t coff = compressRatio == 4 ? 2 : 1;
    const uint64_t kvTokens = kv.geometry.maxSizes[1];
    const uint64_t numTokens = slotMapping.geometry.maxSizes[0];
    const uint64_t numRequests = blockTable.geometry.maxSizes[1];
    const uint64_t tableWidth = blockTable.geometry.maxSizes[0];
    const bool inputsMatch =
        storageBytes > 0 && stateGeometry.geometry.maxSizes[0] == 5 &&
        cacheGeometry.geometry.maxSizes[0] == 4 &&
        (headDim == 128 || headDim == 512) &&
        stateWidth == coff * headDim &&
        (compressRatio == 4 ||
         (compressRatio == 128 && headDim == 512)) &&
        numTokens > 0 && kvTokens >= numTokens &&
        score.geometry.maxSizes[0] == stateWidth &&
        score.geometry.maxSizes[1] >= numTokens &&
        ape.geometry.maxSizes[0] == stateWidth &&
        positions.geometry.maxSizes[0] >= numTokens &&
        kvSlotMapping.geometry.maxSizes[0] >= numTokens &&
        tokenToReq.geometry.maxSizes[0] >= numTokens &&
        numRequests > 0 && tableWidth > 0 &&
        rmsWeight.geometry.maxSizes[0] == headDim &&
        rmsEps.geometry.maxSizes[0] == 1 &&
        cosSinCache.geometry.maxSizes[0] == 64 &&
        cosSinCache.geometry.maxSizes[1] > 0 &&
        (!ordered || completionInput->geometry.maxSizes[0] >= numTokens);
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches = mutationToken.geometry.maxSizes[0] == numTokens;
    if (!outputMatches) {
        mutationToken.geometry.dataType = ordered
            ? tpc_lib_api::DATA_I32
            : tpc_lib_api::DATA_U8;
        mutationToken.geometry.dims = 1;
        mutationToken.geometry.maxSizes[0] = numTokens;
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
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, 4);
    MapDimension(outDefs->inputTensorAccessPattern[2], 0, 0, 0, 0, 3);
    for (unsigned input = 3; input <= 4; ++input) {
        MapDimension(
            outDefs->inputTensorAccessPattern[input],
            0,
            0,
            0,
            0,
            static_cast<int>(stateWidth - 1));
        MapDimension(outDefs->inputTensorAccessPattern[input], 1, 0, 1, 0, 0);
    }
    MapDimension(
        outDefs->inputTensorAccessPattern[5],
        0,
        0,
        0,
        0,
        static_cast<int>(stateWidth - 1));
    MapDimension(
        outDefs->inputTensorAccessPattern[5],
        1,
        0,
        0,
        0,
        static_cast<int>(compressRatio - 1));
    for (unsigned input = 6; input <= 8; ++input) {
        MapDimension(outDefs->inputTensorAccessPattern[input], 0, 0, 1, 0, 0);
    }
    MapDimension(
        outDefs->inputTensorAccessPattern[9],
        0,
        0,
        0,
        0,
        static_cast<int>(tableWidth - 1));
    MapDimension(
        outDefs->inputTensorAccessPattern[9],
        1,
        0,
        0,
        0,
        static_cast<int>(numRequests - 1));
    MapDimension(
        outDefs->inputTensorAccessPattern[10],
        0,
        0,
        0,
        0,
        static_cast<int>(headDim - 1));
    MapDimension(outDefs->inputTensorAccessPattern[11], 0, 0, 0, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[12], 0, 0, 0, 0, 63);
    MapDimension(
        outDefs->inputTensorAccessPattern[12],
        1,
        0,
        0,
        0,
        static_cast<int>(cosSinCache.geometry.maxSizes[1] - 1));
    MapDimension(outDefs->inputTensorAccessPattern[13], 0, 0, 1, 0, 0);
    if (ordered) {
        MapDimension(
            outDefs->inputTensorAccessPattern[14], 0, 0, 1, 0, 0);
    }
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize = constantsType_ == ORDERED_F32
        ? &_binary___deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2_o_end -
              &_binary___deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2_o_start
        : constantsType_ == BF16_CONSTANTS
        ? &_binary___deepseek_v4_save_compress_norm_c4_bf16_gaudi2_o_end -
              &_binary___deepseek_v4_save_compress_norm_c4_bf16_gaudi2_o_start
        : constantsType_ == MIXED_CONSTANTS
            ? &_binary___deepseek_v4_save_compress_norm_c4_mixed_gaudi2_o_end -
                  &_binary___deepseek_v4_save_compress_norm_c4_mixed_gaudi2_o_start
            : &_binary___deepseek_v4_save_compress_norm_c4_f32_gaudi2_o_end -
                  &_binary___deepseek_v4_save_compress_norm_c4_f32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    const unsigned char* isa = constantsType_ == ORDERED_F32
        ? &_binary___deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2_o_start
        : constantsType_ == BF16_CONSTANTS
        ? &_binary___deepseek_v4_save_compress_norm_c4_bf16_gaudi2_o_start
        : constantsType_ == MIXED_CONSTANTS
            ? &_binary___deepseek_v4_save_compress_norm_c4_mixed_gaudi2_o_start
            : &_binary___deepseek_v4_save_compress_norm_c4_f32_gaudi2_o_start;
    std::memcpy(outDefs->kernel.kernelElf, isa, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
