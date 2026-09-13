/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2_o_end;

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
DeepseekV4QnormRopeKvPackBF16Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        mode_ == HYBRID_FP8_BF16_Q
            ? "custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2"
            : "custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4QnormRopeKvPackBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 7;
    constexpr unsigned kOutputCount = 1;
    constexpr uint64_t kHeadDim = 512;
    constexpr uint64_t kRopeDim = 64;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const auto& q = inDefs->inputTensors[0];
    const auto& kv = inDefs->inputTensors[1];
    const auto& cacheStorage = inDefs->inputTensors[2];
    const auto& cacheGeometry = inDefs->inputTensors[3];
    const auto& slots = inDefs->inputTensors[4];
    const auto& positions = inDefs->inputTensors[5];
    const auto& cosSinCache = inDefs->inputTensors[6];
    auto& mutationToken = inDefs->outputTensors[0];
    const bool typesAndRanks =
        HasTypeAndRank(q, tpc_lib_api::DATA_BF16, 3) &&
        HasTypeAndRank(kv, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(cacheStorage, tpc_lib_api::DATA_U8, 1) &&
        HasTypeAndRank(cacheGeometry, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(slots, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(positions, tpc_lib_api::DATA_I32, 1) &&
        HasTypeAndRank(cosSinCache, tpc_lib_api::DATA_F32, 2) &&
        HasTypeAndRank(mutationToken, tpc_lib_api::DATA_U8, 1);
    if (!typesAndRanks) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t numTokens = q.geometry.maxSizes[2];
    const uint64_t numHeads = q.geometry.maxSizes[1];
    const uint64_t storageBytes = cacheStorage.geometry.maxSizes[0];
    const bool inputsMatch =
        q.geometry.maxSizes[0] == kHeadDim &&
        numHeads > 0 && numTokens > 0 &&
        kv.geometry.maxSizes[0] == kHeadDim &&
        kv.geometry.maxSizes[1] >= numTokens &&
        storageBytes >= 584 &&
        cacheGeometry.geometry.maxSizes[0] == 4 &&
        slots.geometry.maxSizes[0] >= numTokens &&
        positions.geometry.maxSizes[0] >= numTokens &&
        cosSinCache.geometry.maxSizes[0] == kRopeDim &&
        cosSinCache.geometry.maxSizes[1] > 0;
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches =
        mutationToken.geometry.maxSizes[0] == numTokens;
    if (!outputMatches) {
        mutationToken.geometry.dataType = tpc_lib_api::DATA_U8;
        mutationToken.geometry.dims = 1;
        mutationToken.geometry.maxSizes[0] = numTokens;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 2;
    outDefs->indexSpaceGeometry[0] = numTokens;
    outDefs->indexSpaceGeometry[1] = numHeads;
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        0,
        1,
        0,
        0,
        static_cast<int>(kHeadDim - 1));
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[1],
        0,
        1,
        0,
        0,
        static_cast<int>(kHeadDim - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[2],
        0,
        0,
        0,
        0,
        static_cast<int>(storageBytes - 1));
    MapDimension(outDefs->inputTensorAccessPattern[3], 0, 0, 0, 0, 3);
    MapDimension(outDefs->inputTensorAccessPattern[4], 0, 0, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[5], 0, 0, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[6],
        0,
        1,
        0,
        0,
        static_cast<int>(kRopeDim - 1));
    MapDimension(
        outDefs->inputTensorAccessPattern[6],
        1,
        1,
        0,
        0,
        static_cast<int>(cosSinCache.geometry.maxSizes[1] - 1));
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const bool hybrid = mode_ == HYBRID_FP8_BF16_Q;
    const unsigned char* isaStart =
        hybrid
            ? &_binary___deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2_o_start
            : &_binary___deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2_o_start;
    const unsigned char* isaEnd =
        hybrid
            ? &_binary___deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2_o_end
            : &_binary___deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2_o_end;
    const unsigned isaSize = isaEnd - isaStart;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        isaStart,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
