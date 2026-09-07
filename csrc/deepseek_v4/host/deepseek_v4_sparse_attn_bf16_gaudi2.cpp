/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_sparse_attn_bf16_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_sparse_attn_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_sparse_attn_bf16_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_sparse_attn_bf16_lengths_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_sparse_attn_bf16_lengths_gaudi2_o_end;

namespace {

constexpr uint64_t kHeadDim = 512;

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

}  // namespace

tpc_lib_api::GlueCodeReturn DeepseekV4SparseAttnBF16Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        m_mode == EXPLICIT_LENGTHS
            ? "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2"
            : "custom_deepseek_v4_sparse_attn_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4SparseAttnBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    const unsigned inputCount = m_mode == EXPLICIT_LENGTHS ? 6 : 5;
    constexpr unsigned kOutputCount = 3;
    if (inDefs->inputTensorNr != inputCount) {
        inDefs->inputTensorNr = inputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const tpc_lib_api::TensorDataType inputTypes[6] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_I32,
    };
    for (unsigned input = 0; input < inputCount; ++input) {
        if (!HasDataType(inDefs->inputTensors[input], inputTypes[input])) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(
            inDefs->outputTensors[0], tpc_lib_api::DATA_BF16) ||
        !HasDataType(
            inDefs->outputTensors[1], tpc_lib_api::DATA_F32) ||
        !HasDataType(
            inDefs->outputTensors[2], tpc_lib_api::DATA_F32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& q = inDefs->inputTensors[0];
    const auto& kv = inDefs->inputTensors[1];
    const auto& indices = inDefs->inputTensors[2];
    const auto& sink = inDefs->inputTensors[3];
    const auto& scale = inDefs->inputTensors[4];
    const auto* topkLengths = m_mode == EXPLICIT_LENGTHS
        ? &inDefs->inputTensors[5]
        : nullptr;
    auto& output = inDefs->outputTensors[0];
    auto& maxLogits = inDefs->outputTensors[1];
    auto& softmaxLse = inDefs->outputTensors[2];

    const bool ranksMatch =
        q.geometry.dims == 3 && kv.geometry.dims == 2 &&
        indices.geometry.dims == 2 && sink.geometry.dims == 1 &&
        scale.geometry.dims == 1 && output.geometry.dims == 3 &&
        maxLogits.geometry.dims == 2 && softmaxLse.geometry.dims == 2;
    const bool lengthsRankMatches =
        topkLengths == nullptr || topkLengths->geometry.dims == 1;
    if (!ranksMatch || !lengthsRankMatches) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t headCount = q.geometry.maxSizes[1];
    const uint64_t batchSize = q.geometry.maxSizes[2];
    const uint64_t sequenceLength = kv.geometry.maxSizes[1];
    const uint64_t topkWidth = indices.geometry.maxSizes[0];
    const bool inputsMatch =
        q.geometry.maxSizes[0] == kHeadDim &&
        kv.geometry.maxSizes[0] == kHeadDim && sequenceLength > 0 &&
        topkWidth > 0 && indices.geometry.maxSizes[1] == batchSize &&
        sink.geometry.maxSizes[0] == headCount &&
        scale.geometry.maxSizes[0] == 1 &&
        (topkLengths == nullptr ||
         topkLengths->geometry.maxSizes[0] == batchSize);
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches =
        output.geometry.maxSizes[0] == kHeadDim &&
        output.geometry.maxSizes[1] == headCount &&
        output.geometry.maxSizes[2] == batchSize &&
        maxLogits.geometry.maxSizes[0] == headCount &&
        maxLogits.geometry.maxSizes[1] == batchSize &&
        softmaxLse.geometry.maxSizes[0] == headCount &&
        softmaxLse.geometry.maxSizes[1] == batchSize;
    if (!outputMatches) {
        output.geometry.dims = 3;
        output.geometry.maxSizes[0] = kHeadDim;
        output.geometry.maxSizes[1] = headCount;
        output.geometry.maxSizes[2] = batchSize;
        maxLogits.geometry.dims = 2;
        maxLogits.geometry.maxSizes[0] = headCount;
        maxLogits.geometry.maxSizes[1] = batchSize;
        softmaxLse.geometry.dims = 2;
        softmaxLse.geometry.maxSizes[0] = headCount;
        softmaxLse.geometry.maxSizes[1] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 3;
    outDefs->indexSpaceGeometry[0] = 1;
    outDefs->indexSpaceGeometry[1] = headCount;
    outDefs->indexSpaceGeometry[2] = batchSize;

    // q/output: [D, H, B]
    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 512, 0, 511);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 512, 0, 511);
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    for (unsigned statsOutput = 1; statsOutput <= 2; ++statsOutput) {
        MapDimension(
            outDefs->outputTensorAccessPattern[statsOutput],
            0,
            1,
            1,
            0,
            0);
        MapDimension(
            outDefs->outputTensorAccessPattern[statsOutput],
            1,
            2,
            1,
            0,
            0);
    }

    // kv: [D, S], where sparse indices make S an indirect full-range access.
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 512, 0, 511);
    MapDimension(
        outDefs->inputTensorAccessPattern[1],
        1,
        0,
        0,
        0,
        static_cast<int>(sequenceLength - 1));

    // indices: [K, B]
    MapDimension(
        outDefs->inputTensorAccessPattern[2],
        0,
        0,
        0,
        0,
        static_cast<int>(topkWidth - 1));
    MapDimension(outDefs->inputTensorAccessPattern[2], 1, 2, 1, 0, 0);

    // sink: [H], scale: [1]
    MapDimension(outDefs->inputTensorAccessPattern[3], 0, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[4], 0, 0, 0, 0, 0);
    if (topkLengths != nullptr) {
        // topk_lengths: [B]
        MapDimension(
            outDefs->inputTensorAccessPattern[5], 0, 2, 1, 0, 0);
    }

    outDefs->kernel.paramsNr = 0;
    const unsigned char* isaStart =
        m_mode == EXPLICIT_LENGTHS
        ? &_binary___deepseek_v4_sparse_attn_bf16_lengths_gaudi2_o_start
        : &_binary___deepseek_v4_sparse_attn_bf16_gaudi2_o_start;
    const unsigned char* isaEnd =
        m_mode == EXPLICIT_LENGTHS
        ? &_binary___deepseek_v4_sparse_attn_bf16_lengths_gaudi2_o_end
        : &_binary___deepseek_v4_sparse_attn_bf16_gaudi2_o_end;
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
