// SPDX-License-Identifier: Apache-2.0
/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "dflash2_score_select_i64_bf16_f32_gaudi2.hpp"

extern unsigned char _binary_dflash2_score_select_i64_bf16_f32_gaudi2_o_start;
extern unsigned char _binary_dflash2_score_select_i64_bf16_f32_gaudi2_o_end;

namespace {

constexpr uint64_t kVocabSize = 248320;
constexpr uint64_t kRank = 256;
constexpr uint64_t kSteps = 7;
constexpr uint64_t kTopK = 16;
constexpr uint64_t kMaxBatch = 16;

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

tpc_lib_api::GlueCodeReturn DFlash2ScoreSelectI64Bf16F32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(kernelName, "flashinfer_gaudi_dflash2_score_select_i32_bf16_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DFlash2ScoreSelectI64Bf16F32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 6;
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const auto& predecessor = inDefs->inputTensors[0];
    const auto& successor = inDefs->inputTensors[1];
    const auto& candidates = inDefs->inputTensors[2];
    const auto& unary = inDefs->inputTensors[3];
    const auto& hidden = inDefs->inputTensors[4];
    const auto& anchors = inDefs->inputTensors[5];
    auto& output = inDefs->outputTensors[0];

    const bool inputsValid =
        HasTypeAndRank(predecessor, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(successor, tpc_lib_api::DATA_BF16, 2) &&
        HasTypeAndRank(candidates, tpc_lib_api::DATA_I32, 3) &&
        HasTypeAndRank(unary, tpc_lib_api::DATA_F32, 3) &&
        HasTypeAndRank(hidden, tpc_lib_api::DATA_BF16, 3) &&
        HasTypeAndRank(anchors, tpc_lib_api::DATA_I32, 1);
    if (!inputsValid) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t topK = candidates.geometry.maxSizes[0];
    const uint64_t steps = candidates.geometry.maxSizes[1];
    const uint64_t batchSize = candidates.geometry.maxSizes[2];
    const bool shapesMatch =
        predecessor.geometry.maxSizes[0] == kRank &&
        predecessor.geometry.maxSizes[1] == kVocabSize &&
        successor.geometry.maxSizes[0] == kRank &&
        successor.geometry.maxSizes[1] == kVocabSize &&
        topK == kTopK && steps == kSteps && batchSize > 0 && batchSize <= kMaxBatch &&
        unary.geometry.maxSizes[0] == topK && unary.geometry.maxSizes[1] == steps &&
        unary.geometry.maxSizes[2] == batchSize &&
        hidden.geometry.maxSizes[0] == kRank && hidden.geometry.maxSizes[1] == steps &&
        hidden.geometry.maxSizes[2] == batchSize && anchors.geometry.maxSizes[0] == batchSize;
    if (!shapesMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches =
        HasTypeAndRank(output, tpc_lib_api::DATA_I32, 2) &&
        output.geometry.maxSizes[0] == steps && output.geometry.maxSizes[1] == batchSize;
    if (!outputMatches) {
        output.geometry.dataType = tpc_lib_api::DATA_I32;
        output.geometry.dims = 2;
        output.geometry.maxSizes[0] = steps;
        output.geometry.maxSizes[1] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = batchSize;

    for (unsigned input = 0; input <= 1; ++input) {
        MapDimension(outDefs->inputTensorAccessPattern[input], 0, 0, 0, 0, static_cast<int>(kRank - 1));
        MapDimension(outDefs->inputTensorAccessPattern[input], 1, 0, 0, 0, static_cast<int>(kVocabSize - 1));
    }
    for (unsigned input = 2; input <= 3; ++input) {
        MapDimension(outDefs->inputTensorAccessPattern[input], 0, 0, 0, 0, static_cast<int>(topK - 1));
        MapDimension(outDefs->inputTensorAccessPattern[input], 1, 0, 0, 0, static_cast<int>(steps - 1));
        MapDimension(outDefs->inputTensorAccessPattern[input], 2, 0, 1, 0, 0);
    }
    MapDimension(outDefs->inputTensorAccessPattern[4], 0, 0, 0, 0, static_cast<int>(kRank - 1));
    MapDimension(outDefs->inputTensorAccessPattern[4], 1, 0, 0, 0, static_cast<int>(steps - 1));
    MapDimension(outDefs->inputTensorAccessPattern[4], 2, 0, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[5], 0, 0, 1, 0, 0);

    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(steps - 1));
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize = &_binary_dflash2_score_select_i64_bf16_f32_gaudi2_o_end -
        &_binary_dflash2_score_select_i64_bf16_f32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary_dflash2_score_select_i64_bf16_f32_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
